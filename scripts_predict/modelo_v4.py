"""Versão 4 do modelo de preço: histórico simulado com valorização PRÓPRIA de cada imóvel.

Na v3, o preço de cada ano é o preço de hoje dividido pelo quanto o índice IPCA+INCC subiu:
todos os imóveis têm o mesmo passado em %. Na v4, cada imóvel tem uma taxa própria:

    taxa do imóvel = taxa do índice IPCA+INCC (ano a ano) + ajuste fuzzy (p.p. ao ano)

    preço simulado em jun/Y = preço de hoje / (fator do índice de jun/Y a hoje
                                               x (1 + ajuste/100) ^ anos entre jun/Y e hoje)

O ajuste vem das regras fuzzy de `fuzzy_valorizacao.py` (preço/m² do bairro, área do
imóvel, tendência do bairro nos anúncios, tipo) e é centralizado por tipo: na média, a
valorização continua igual à do índice.

Todo o resto é igual à v3, para a comparação ser justa: mesma base, preços levados a
ago/2026, mesma separação treino/teste (semente 42), mesma limpeza aprendida só no treino,
mesmos candidatos, mesma validação cruzada agrupada por imóvel, erro só nas linhas reais e
mesmo critério de escolha (MdAPE, desempate pelo MAE). As régua da fuzzy (medianas do
bairro, área, tendência) também são tiradas só dos imóveis de treino.

    python scripts_predict/modelo_v4.py
"""

import argparse
import json
import sqlite3
import warnings
from datetime import datetime
from pathlib import Path

import fuzzy_valorizacao as fz
import numpy as np
import pandas as pd
from imoveis_ml_hibrido import SEGMENT_TYPES, prepare_segment_frame, split_and_prepare_segment
from indice_ipca_incc import fator, nivel, resumo
from model_io import load_model, save_model
from modelo_v3 import (
    ANOS_SIMULADOS,
    CANDIDATOS,
    ENSEMBLE,
    FEATURES_V3,
    MEIO_DO_ANO,
    avaliar,
    chave,
    escolher,
    linha_cv,
    linha_ensemble,
    mdape,
    modelo_api,
    montar,
    prever,
    treinar_ensemble,
)
from modelo_v3 import carregar_reais as carregar_reais_v3
from sklearn.model_selection import GroupKFold, cross_val_predict

warnings.filterwarnings("ignore")

DEFAULT_NORMALIZED_DB = Path("imoveis_normalizados.db")
DEFAULT_UNIFIED_DB = Path("imoveis_unificado.db")
DEFAULT_MODEL_PATH = Path("modelos/preco_imovel_modelo_v4.pkl")
DEFAULT_REPORT = Path("docs/reports/modelo_v4.md")
DEFAULT_DATASET_DB = Path("imoveis_simulados_v4.db")
DEFAULT_DATASET_CSV = Path("dados/dataset_simulado_v4.csv")
DEFAULT_TAXAS = Path("dados/taxas_fuzzy_v4.json")
PASTA_ESTIMADORES = "v4"
KIND = "preco_v4_historico_fuzzy"


# ------------------------------------------------------------------ dados
def carregar_reais(normalized_db=DEFAULT_NORMALIZED_DB):
    """Mesmos imóveis e preços da v3, com os códigos dos anúncios (para medir a tendência)."""
    df = carregar_reais_v3(normalized_db)
    with sqlite3.connect(normalized_db) as conn:
        ids = pd.read_sql_query(
            "SELECT preco, origem_ids FROM imoveis_normalizados WHERE preco IS NOT NULL AND preco > 0", conn
        )
    ids = ids.loc[df.index]  # carregar_reais_v3 filtra sem renumerar: mesmo índice, mesma ordem
    df["origem_ids"] = ids["origem_ids"].map(lambda s: json.loads(s) if s else [])
    return df


def codigos(df):
    return {c for lista in df["origem_ids"] for c in lista}


def taxas_fuzzy(treino_real, unified_db=DEFAULT_UNIFIED_DB):
    """Régua e ajuste fuzzy de cada imóvel, tudo medido só com os imóveis de `treino_real`."""
    tendencias = fz.tendencias_bairros(unified_db, codigos(treino_real))
    ref = fz.referencias(treino_real, tendencias)
    return ref, fz.ajustes(treino_real, ref)


def simular(reais, ajuste, anos=ANOS_SIMULADOS):
    """Registros simulados: preço levado a jun de cada ano pelo índice E pelo ajuste do imóvel."""
    indice = nivel()
    ajuste = np.asarray(ajuste, dtype=float)
    linhas = []
    for ano in anos:
        bloco = reais.copy()
        f = np.array([fator(f"{ano}-06", m, indice) for m in bloco["mes"]])
        anos_ate_hoje = bloco["ano"].to_numpy() - (ano + MEIO_DO_ANO)
        bloco["preco"] = bloco["preco"] / (f * (1 + ajuste / 100) ** anos_ate_hoje)
        bloco["ano"] = ano + MEIO_DO_ANO
        bloco["mes"] = f"{ano}-06"
        bloco["simulado"] = 1
        linhas.append(bloco)
    return pd.concat(linhas, ignore_index=True)


def dados_segmento(df, tipo, seed=42, unified_db=DEFAULT_UNIFIED_DB):
    """Mesma separação da v3; o histórico simulado usa o ajuste fuzzy de cada imóvel."""
    split = split_and_prepare_segment(df, tipo, seed=seed)
    ref, ajuste = taxas_fuzzy(split["train"], unified_db)
    split["train"] = split["train"].assign(ajuste_pp=ajuste)
    split["ref_fuzzy"] = ref
    treino = pd.concat([split["train"], simular(split["train"], ajuste)], ignore_index=True)
    return split, treino, treino["imovel"].to_numpy(), (treino["simulado"] == 0).to_numpy()


# ----------------------------------------------------------------- modelo
def resultados_tabpfn(tipo, imoveis_reais):
    """Tentativas do TabPFN no protocolo da v4 (`tabpfn_v4.py`) e as previsões que terminaram."""
    from tabpfn_model import TABPFN_VARIANTS
    from tabpfn_v4 import RESUMO, arquivo

    if not RESUMO.exists():
        return [], {}
    tentativas = [t for t in json.loads(RESUMO.read_text(encoding="utf-8"))["tentativas"] if t["tipo"] == tipo]
    previsoes = {}
    for t in tentativas:
        caminho = arquivo(tipo, t["modelo"])
        if t["status"] != "ok" or not caminho.exists():
            continue
        dados = json.loads(caminho.read_text(encoding="utf-8"))
        if dados["cv_imovel"] != [int(i) for i in imoveis_reais]:
            raise SystemExit(f"Previsões do {t['modelo']} ({tipo}) não batem com a separação atual: rode tabpfn_v4.py.")
        previsoes[t["modelo"]] = {**dados, "api": TABPFN_VARIANTS.get(t["modelo"]) == "api"}
    return tentativas, previsoes


def passado_previsto(prever_fn, teste):
    """Quanto o modelo diz que cada imóvel de teste valia em 2016 e 2021 (fração de hoje).

    É a valorização que o modelo APRENDEU com o histórico simulado. Na v3 ela é quase igual
    para todos; na v4 deve variar de imóvel para imóvel. Não é medida de acerto.
    """
    hoje = np.asarray(prever_fn(teste), dtype=float)
    saida = {}
    for ano in (2016, 2021):
        r = np.asarray(prever_fn(teste.assign(ano=ano + MEIO_DO_ANO)), dtype=float) / hoje
        anos = teste["ano"].to_numpy() - (ano + MEIO_DO_ANO)
        taxa = (r ** (-1 / anos) - 1) * 100
        saida[str(ano)] = {
            "razao_mediana": float(np.median(r)),
            "taxa_anual_p10": float(np.percentile(taxa, 10)),
            "taxa_anual_mediana": float(np.median(taxa)),
            "taxa_anual_p90": float(np.percentile(taxa, 90)),
        }
    return saida


def segmento(df, tipo, seed=42):
    split, treino, grupos, real = dados_segmento(df, tipo, seed)
    treino_real = split["train"]

    y = treino["preco"].to_numpy()[real]
    ranking, previsoes = [], {}
    cv = GroupKFold(n_splits=5)
    for nome in CANDIDATOS:
        for usa_log in (False, True):
            alvo = "log(preco)" if usa_log else "preco"
            pred = cross_val_predict(montar(nome, usa_log, seed), treino[FEATURES_V3], treino["preco"], cv=cv, groups=grupos)
            previsoes[(nome, alvo)] = pred[real]
            ranking.append(linha_cv(nome, alvo, y, pred[real]))
            print(f"  [{tipo}] {nome} ({alvo}): MdAPE CV {ranking[-1]['mdape']:.2f}% MAE {ranking[-1]['mae']:,.0f}", flush=True)
    tentativas_tabpfn, tabpfn = resultados_tabpfn(tipo, treino.loc[real, "imovel"])
    for nome, dados in tabpfn.items():
        ranking.append(linha_cv(nome, "preco", y, np.asarray(dados["cv_previsto"]), no_site=dados["api"]))
        ranking[-1]["linhas_treino"] = dados["linhas_treino"]
        if dados["api"]:
            ranking[-1]["api"] = True
    ensemble, _ = linha_ensemble(ranking, previsoes, y)
    ranking.append(ensemble)
    ranking.sort(key=chave)
    escolhido, _ = escolher(ranking)
    reserva, membros = escolher(ranking, api=False)
    print(f"  [{tipo}] escolhido: {escolhido['model']} ({escolhido['target']}), MdAPE CV {escolhido['mdape']:.2f}%", flush=True)

    entradas = treinar_ensemble(treino, membros, seed)
    teste_tip, teste_comp = split["test_typical"], split["test_full"]
    metricas_reserva = avaliar(teste_tip["preco"].to_numpy(), prever(entradas, teste_tip))
    metricas, metricas_comp = metricas_reserva, avaliar(teste_comp["preco"].to_numpy(), prever(entradas, teste_comp))
    passado = passado_previsto(lambda X: prever(entradas, X), teste_tip)
    if escolhido.get("api"):
        dados = tabpfn[escolhido["model"]]
        metricas = avaliar(teste_tip["preco"].to_numpy(), np.asarray(dados["teste_tipico"]))
        metricas_comp = avaliar(teste_comp["preco"].to_numpy(), np.asarray(dados["teste_completo"]))
        api = modelo_api(treino)
        passado = passado_previsto(api.predict, teste_tip)

    # Modelo final: todos os imóveis do tipo + simulados com o ajuste fuzzy da base inteira.
    final_real, rare, clip = prepare_segment_frame(df, tipo)
    ref_final, ajuste_final = taxas_fuzzy(final_real)
    final = pd.concat([final_real, simular(final_real, ajuste_final)], ignore_index=True)
    entradas_finais = treinar_ensemble(final, membros, seed)
    return {
        "api": {"model": escolhido["model"], "treino": final[FEATURES_V3 + ["preco"]].reset_index(drop=True)}
        if escolhido.get("api") else None,
        "metrics_reserva": metricas_reserva,
        "linhas_reais_treino": int(len(treino_real)),
        "linhas_simuladas_treino": int((treino["simulado"] == 1).sum()),
        "teste_tipico": int(len(teste_tip)),
        "ranking": ranking,
        "escolhido": {k: escolhido[k] for k in ("model", "target", "mdape", "mae")},
        "reserva": {k: reserva[k] for k in ("model", "target", "mdape", "mae")},
        "tabpfn_tentativas": tentativas_tabpfn,
        "metrics": metricas,
        "metrics_full": metricas_comp,
        "passado_previsto": passado,
        "ajuste_treino": {"imovel": treino_real["imovel"].astype(int).tolist(),
                          "bairro": treino_real["bairro"].tolist(),
                          "ajuste_pp": [float(a) for a in treino_real["ajuste_pp"]]},
        "ref_fuzzy": ref_final,
        "rare_bairros": sorted(rare),
        "clip_bounds": {k: list(v) for k, v in clip.items()},
        "ensemble": entradas_finais,
        "selected": [{k: e[k] for k in ("model_name", "target", "weight")} for e in entradas_finais],
    }


def separar_estimadores(art, pasta):
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    for antigo in pasta.glob("*.pkl*"):
        antigo.unlink()
    for tipo, seg in art["segments"].items():
        for i, e in enumerate(seg["ensemble"]):
            nome = f"{tipo.lower()}_{i}.pkl"
            save_model(e.pop("estimator"), pasta / nome, compress=3)
            e["arquivo"] = nome


def load_v4(path=DEFAULT_MODEL_PATH):
    art = load_model(path)
    if not isinstance(art, dict) or art.get("kind") != KIND:
        raise SystemExit("Arquivo do modelo v4 inválido. Rode: python scripts_predict/modelo_v4.py")
    art["pasta_estimadores"] = Path(path).parent / PASTA_ESTIMADORES
    return art


# ------------------------------------------------------------------ saídas
def exportar_dataset(df, db_path=DEFAULT_DATASET_DB, csv_path=DEFAULT_DATASET_CSV):
    """Dataset v4 completo (reais + simulados), com o ajuste fuzzy e a taxa de cada imóvel."""
    partes = []
    for tipo in SEGMENT_TYPES:
        base = df[df["tipo_imovel"] == tipo].copy()
        ref, ajuste = taxas_fuzzy(base)
        base["ajuste_pp"] = ajuste
        partes += [base, simular(base, ajuste)]
    completo = pd.concat(partes, ignore_index=True)
    indice = nivel()
    reais = completo["simulado"] == 0
    completo["preco_anunciado"] = np.where(
        reais,
        completo["preco"] / np.array([fator(m, r, indice) for m, r in zip(completo["mes_anuncio"], completo["mes"])]),
        np.nan,
    )
    completo.loc[~reais, "mes_anuncio"] = None
    colunas = ["imovel", "simulado", "ano", "mes", "tipo_imovel", "bairro", "area_total", "area_privada",
               "quartos", "banheiros", "vagas", "preco", "mes_anuncio", "preco_anunciado", "ajuste_pp"]
    completo = completo[colunas].sort_values(["imovel", "ano"])
    completo["preco"] = completo["preco"].round(0)
    completo["preco_anunciado"] = completo["preco_anunciado"].round(0)
    completo["ajuste_pp"] = completo["ajuste_pp"].round(3)
    with sqlite3.connect(db_path) as conn:
        completo.to_sql("imoveis_v4", conn, if_exists="replace", index=False)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    completo.to_csv(csv_path, index=False, encoding="utf-8")
    return completo


def resumo_taxas(dataset, indice):
    """Taxa de cada imóvel (índice médio de 10 anos + ajuste), por tipo e por bairro."""
    reais = dataset[dataset["simulado"] == 0].copy()
    reais["taxa"] = indice["media_anual_pct"] + reais["ajuste_pp"]

    def estat(s):
        return {"n": int(len(s)), "min": float(s.min()), "p10": float(s.quantile(0.1)), "mediana": float(s.median()),
                "media": float(s.mean()), "p90": float(s.quantile(0.9)), "max": float(s.max())}

    saida = {"indice_media_anual_pct": indice["media_anual_pct"], "por_tipo": {}, "por_bairro": {}}
    for tipo, g in reais.groupby("tipo_imovel"):
        saida["por_tipo"][tipo] = {"ajuste_pp": estat(g["ajuste_pp"]), "taxa_pct": estat(g["taxa"])}
        saida["por_bairro"][tipo] = {
            b: estat(gb["taxa"]) for b, gb in g.groupby("bairro") if len(gb) >= fz.MIN_IMOVEIS_BAIRRO
        }
    return saida


def relatorio(art):
    idx = art["indice"]
    linhas = [
        "# Modelo de preço v4: histórico simulado com ajuste fuzzy por imóvel",
        "",
        "Gerado por `scripts_predict/modelo_v4.py`. Cada imóvel real ganha os anos",
        f"{ANOS_SIMULADOS[0]} a {ANOS_SIMULADOS[-1]} com o índice IPCA+INCC (média de 10 anos:",
        f"{idx['media_anual_pct']:.2f}% ao ano) mais um ajuste fuzzy próprio (`fuzzy_valorizacao.py`),",
        "centralizado por tipo. Erros medidos só nos preços reais do teste.",
        "",
        "| Tipo | Modelo escolhido | Linhas reais (treino) | Linhas simuladas | MAE | RMSE | R² | MdAPE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for tipo, s in art["segments"].items():
        m = s["metrics"]
        linhas.append(
            f"| {tipo} | {s['escolhido']['model']} ({s['escolhido']['target']}) | {s['linhas_reais_treino']} | "
            f"{s['linhas_simuladas_treino']} | R$ {m['mae']:,.0f} | R$ {m['rmse']:,.0f} | {m['r2']:.3f} | "
            f"{m['erro_mediano_pct']:.1f}% |".replace(",", ".")
        )
    linhas += ["", "## Valorização aprendida pelo modelo (imóveis de teste; não é medida de acerto)", "",
               "| Tipo | Período | Valor no início / hoje (mediana) | Taxa anual p10 | mediana | p90 |",
               "|---|---|---:|---:|---:|---:|"]
    for tipo, s in art["segments"].items():
        for ano, p in s["passado_previsto"].items():
            linhas.append(f"| {tipo} | {ano}–2026 | {p['razao_mediana']:.3f} | {p['taxa_anual_p10']:.2f}% | "
                          f"{p['taxa_anual_mediana']:.2f}% | {p['taxa_anual_p90']:.2f}% |")
    linhas += ["", "## Escolha do modelo (validação cruzada por imóvel, erro só nas linhas reais)", "",
               "Mesmo critério da v3: menor MdAPE; empate (1 casa decimal) decidido pelo menor MAE.", ""]
    for tipo, s in art["segments"].items():
        e = s["escolhido"]
        linhas += [f"### {tipo}", "", "| Posição | Modelo | Alvo | MdAPE CV | MAE CV | No site |",
                   "|---:|---|---|---:|---:|---|"]
        for pos, r in enumerate(s["ranking"], start=1):
            nome = r["model"] + (" ★" if (r["model"], r["target"]) == (e["model"], e["target"]) else "")
            if "linhas_treino" in r:
                nome += f" ({r['linhas_treino']:,} linhas de treino)".replace(",", ".")
            linhas.append(f"| {pos} | {nome} | {r['target']} | {r['mdape']:.1f}% | "
                          f"R$ {r['mae']:,.0f} | {'sim' if r['no_site'] else 'não'} |".replace(",", "."))
        linhas += ["", "Reserva que roda no site: " + ", ".join(
            f"{x['model_name']} ({x['target']}, peso {x['weight']:.2f})" for x in s["selected"]), ""]
        if s["tabpfn_tentativas"]:
            linhas += ["Tentativas do TabPFN (`tabpfn_v4.py`):", "",
                       "| Variante | Linhas de treino | Resultado | Tempo |", "|---|---|---|---:|"]
            for t in s["tabpfn_tentativas"]:
                status = {"ok": "terminou", "cortado": "cortado por tempo", "erro": "erro", "pulado": "pulado"}[t["status"]]
                if t["detalhe"]:
                    status += f": {t['detalhe']}"
                linhas.append(f"| {t['modelo']} | {t['linhas'] or '—'} | {status} | {t['minutos']:.0f} min |")
            linhas.append("")
    return "\n".join(linhas) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Modelo de preço v4 (histórico simulado com ajuste fuzzy).")
    parser.add_argument("--normalized-db", default=str(DEFAULT_NORMALIZED_DB))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--so-dataset", action="store_true", help="só gera o dataset e as taxas, sem treinar")
    args = parser.parse_args()

    df = carregar_reais(args.normalized_db)
    dataset = exportar_dataset(df)
    taxas = resumo_taxas(dataset, resumo())
    DEFAULT_TAXAS.write_text(json.dumps(taxas, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Dataset v4: {int((dataset.simulado == 0).sum())} reais + {int((dataset.simulado == 1).sum())} simulados.")
    if args.so_dataset:
        return

    segmentos = {}
    for tipo in SEGMENT_TYPES:
        print(f"Treinando {tipo}...", flush=True)
        segmentos[tipo] = segmento(df, tipo)
    art = {"kind": KIND, "indice": resumo(), "anos_simulados": ANOS_SIMULADOS, "segments": segmentos,
           "regras_fuzzy": fz.REGRAS, "entradas_fuzzy": fz.ENTRADAS, "trained_at": datetime.now().isoformat()}
    separar_estimadores(art, Path(args.model_path).parent / PASTA_ESTIMADORES)
    save_model(art, Path(args.model_path))
    Path(DEFAULT_REPORT).write_text(relatorio(art), encoding="utf-8")
    for tipo, s in segmentos.items():
        m = s["metrics"]
        print(f"[{tipo}] v4 {s['escolhido']['model']}: MdAPE {m['erro_mediano_pct']:.1f}% MAE {m['mae']:,.0f} R2 {m['r2']:.3f}")
    print(json.dumps({"report": str(DEFAULT_REPORT)}))


if __name__ == "__main__":
    import modelo_v4

    modelo_v4.main()
