"""Versão 3 do modelo de preço: anúncios reais + histórico SIMULADO pelo índice IPCA+INCC.

Pedido da orientação: como 5 meses de anúncios não mostram a valorização, o dataset
ganha um passado simulado. Para cada imóvel real, são criados registros de 2016 a 2025
com o preço "deflacionado" pelo índice IPCA+INCC realizado (`indice_ipca_incc.py`):

    preço simulado no ano Y = preço anunciado / (quanto o índice subiu de jun/Y até o mês do anúncio)

Cada registro simulado leva `simulado = 1`. O modelo recebe o ano como variável e
aprende a relação preço x tempo; com isso estima quanto um imóvel valia em anos
anteriores, coerente com o índice.

Cuidados (para a banca):
- a simulação usa índices REAIS de cada ano, mas o passado de cada imóvel é uma
  HIPÓTESE (o imóvel teria acompanhado o índice), não um dado observado;
- treino/teste separados por imóvel ANTES de simular: os simulados de um imóvel de teste
  nunca entram no treino;
- o erro é medido SÓ nos preços reais coletados; medir nos simulados seria circular (o
  modelo acertaria a própria regra que gerou os dados);
- a validação cruzada agrupa por imóvel (GroupKFold) e mede o erro só nas linhas reais;
- o modelo publicado é o de menor erro percentual mediano (MdAPE) na validação cruzada,
  com o ensemble concorrendo como mais um candidato; empate é desempatado pelo MAE.

    python scripts_predict/modelo_v3.py
"""

import argparse
import json
import sqlite3
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from imoveis_ml_hibrido import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    SEGMENT_TYPES,
    apply_numeric_clip_bounds,
    apply_rare_bairros,
    calculate_metrics,
    canonical_tipo,
    make_model_factories,
    normalize_text,
    prepare_segment_frame,
    safe_expm1,
    split_and_prepare_segment,
)
from indice_ipca_incc import fator, nivel, resumo
from model_io import load_model, save_model
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

DEFAULT_NORMALIZED_DB = Path("imoveis_normalizados.db")
DEFAULT_MODEL_PATH = Path("modelos/preco_imovel_modelo_v3.pkl")
DEFAULT_REPORT = Path("docs/reports/modelo_v3.md")
DEFAULT_DATASET_DB = Path("imoveis_simulados_v3.db")  # fora do padrão das coletas
DEFAULT_DATASET_CSV = Path("dados/dataset_simulado_v3.csv")
ANOS_SIMULADOS = list(range(2016, 2026))
MEIO_DO_ANO = 5.5 / 12  # meio de junho, na mesma convenção dos reais: ano + (mês - 0,5) / 12
NUM_V3 = NUMERIC_COLUMNS + ["ano"]
FEATURES_V3 = NUM_V3 + CATEGORICAL_COLUMNS
# MLP/SVM e as regressões lineares sem penalização ficam de fora (piores no v1). O Random
# Forest completo passa de 200 MB em memória, mas comprimido fica abaixo de 80 MB por arquivo.
CANDIDATOS = ["CatBoost", "XGBoost", "Gradient Boosting", "Random Forest", "Ridge"]
# Cada estimador do site fica num arquivo próprio e comprimido (o GitHub aceita até 100 MB por
# arquivo); o arquivo principal guarda só o resto do artefato.
PASTA_ESTIMADORES = "v3"
# Random Forest com 100 árvores (em vez de 400): o MdAPE na validação cruzada mudou menos de
# 0,2 ponto percentual e o modelo ocupa 1/4 da memória, o que cabe no site gratuito.
ARVORES_RF = 100
KIND = "preco_v3_historico_simulado"
TOP_ENSEMBLE = 3
ENSEMBLE = "Ensemble"


# ------------------------------------------------------------------ dados
def carregar_reais(normalized_db):
    """Mesmos filtros e ordem de `load_training_frame` (mesma separação do v1) + data."""
    conn = sqlite3.connect(normalized_db)
    df = pd.read_sql_query(
        "SELECT preco, bairro, tipo_imovel, area_total, area_privada, quartos, banheiros, "
        "vagas, data_insercao FROM imoveis_normalizados WHERE preco IS NOT NULL AND preco > 0",
        conn,
    )
    conn.close()
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df = df[(df["area_total"] > 0) | (df["area_privada"] > 0)].copy()
    df["bairro"] = df["bairro"].fillna("").map(normalize_text)
    df["tipo_imovel"] = df["tipo_imovel"].fillna("").map(canonical_tipo)
    # Todos os preços reais são levados ao mês da última coleta (ago/2026) pelo índice.
    # Sem isso, o modelo confundia a diferença entre imóveis anunciados em meses diferentes
    # de 2026 (composição) com valorização. O ajuste é pequeno (no máximo ~2,5%).
    mes = pd.to_datetime(df["data_insercao"]).dt.to_period("M")
    referencia = mes.max()
    indice = nivel()
    df["mes_anuncio"] = mes.astype(str)
    df["preco"] = df["preco"] * np.array([fator(m, str(referencia), indice) for m in df["mes_anuncio"]])
    df["mes"] = str(referencia)
    df["ano"] = referencia.year + (referencia.month - 0.5) / 12
    df["simulado"] = 0
    df["imovel"] = np.arange(len(df))
    return df.drop(columns="data_insercao")


def simular(reais, anos=ANOS_SIMULADOS):
    """Registros simulados de cada imóvel real: preço levado a jun de cada ano pelo índice."""
    indice = nivel()
    linhas = []
    for ano in anos:
        bloco = reais.copy()
        f = np.array([fator(f"{ano}-06", m, indice) for m in bloco["mes"]])
        bloco["preco"] = bloco["preco"] / f
        bloco["ano"] = ano + MEIO_DO_ANO
        bloco["mes"] = f"{ano}-06"
        bloco["simulado"] = 1
        linhas.append(bloco)
    return pd.concat(linhas, ignore_index=True)


# ----------------------------------------------------------------- modelo
def preprocessador():
    num = Pipeline([("i", SimpleImputer(strategy="constant", fill_value=0)), ("s", StandardScaler())])
    cat = Pipeline(
        [
            ("i", SimpleImputer(strategy="constant", fill_value="")),
            ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer([("num", num, NUM_V3), ("cat", cat, CATEGORICAL_COLUMNS)])


def montar(nome, usa_log, seed=42):
    modelo = make_model_factories(seed=seed)[nome]()
    if nome == "Random Forest":
        modelo.set_params(n_estimators=ARVORES_RF)
    pipe = Pipeline([("pre", preprocessador()), ("modelo", modelo)])
    if usa_log:
        return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=safe_expm1, check_inverse=False)
    return pipe


def treinar_ensemble(treino, membros, seed=42):
    """Treina os membros escolhidos (um só, com peso 1, quando o melhor é um modelo isolado)."""
    entradas = []
    for r in membros:
        modelo = montar(r["model"], r["target"] == "log(preco)", seed).fit(treino[FEATURES_V3], treino["preco"])
        entradas.append({"model_name": r["model"], "target": r["target"], "weight": float(r["weight"]), "estimator": modelo})
    return entradas


def prever(entradas, X):
    return sum(e["weight"] * np.asarray(e["estimator"].predict(X[FEATURES_V3]), dtype=float) for e in entradas)


def mdape(y, pred):
    """Erro percentual absoluto mediano (MdAPE): metade das previsões erra menos que isso."""
    return float(np.median(np.abs(np.asarray(pred) - y) / y) * 100)


def avaliar(y, pred):
    m = calculate_metrics(y, pred)
    m.pop("mape", None)
    m["erro_mediano_pct"] = mdape(y, pred)
    return m


# ------------------------------------------------------------- escolha
def chave(r):
    """Critério de escolha: menor MdAPE na validação cruzada (só linhas reais).

    Empate = mesmo MdAPE com 1 casa decimal (como aparece nas tabelas); desempata o menor MAE.
    """
    return (round(r["mdape"], 1), r["mae"])


def linha_cv(nome, alvo, y, pred, no_site=True):
    return {"model": nome, "target": alvo, "mdape": mdape(y, pred),
            "mae": float(np.mean(np.abs(pred - y))), "no_site": no_site}


def linha_ensemble(ranking, previsoes, y):
    """Ensemble dos 3 melhores (por MdAPE) entre os que cabem no site, como mais um candidato.

    Peso de cada membro = 1/MdAPE (normalizado). A previsão do ensemble na validação cruzada
    é a média ponderada das previsões fora da amostra dos membros, então ele é comparado
    com os modelos isolados nas mesmas linhas.
    """
    top = sorted((r for r in ranking if r["no_site"] and not r.get("api")), key=chave)[:TOP_ENSEMBLE]
    pesos = np.array([1 / r["mdape"] for r in top])
    pesos = pesos / pesos.sum()
    pred = sum(p * previsoes[(r["model"], r["target"])] for p, r in zip(pesos, top))
    linha = linha_cv(ENSEMBLE, "ensemble", y, pred)
    linha["membros"] = [{"model": r["model"], "target": r["target"], "weight": float(p)} for p, r in zip(pesos, top)]
    return linha, pred


def escolher(ranking, api=True):
    """Modelo publicado: o melhor pelo critério entre os que cabem no site (inclui o ensemble).

    Com `api=False`, só os modelos que rodam no próprio site: é a reserva usada quando o
    escolhido é o TabPFN pela API e a API não responde.
    """
    melhor = min((r for r in ranking if r["no_site"] and (api or not r.get("api"))), key=chave)
    if melhor["model"] == ENSEMBLE:
        return melhor, melhor["membros"]
    return melhor, [{"model": melhor["model"], "target": melhor["target"], "weight": 1.0}]


def resultados_tabpfn(tipo, imoveis_reais):
    """Tentativas do TabPFN (`tabpfn_v3.py`) e as previsões das que terminaram dentro do corte.

    Cada previsão traz `api=True` quando a variante roda pela API da Prior Labs (cabe no site).
    """
    from tabpfn_model import TABPFN_VARIANTS
    from tabpfn_v3 import RESUMO, arquivo

    if not RESUMO.exists():
        return [], {}
    resumo = json.loads(RESUMO.read_text(encoding="utf-8"))
    tentativas = [t for t in resumo["tentativas"] if t["tipo"] == tipo]
    previsoes = {}
    for t in tentativas:
        caminho = arquivo(tipo, t["modelo"])
        if t["status"] != "ok" or not caminho.exists():
            continue
        dados = json.loads(caminho.read_text(encoding="utf-8"))
        if dados["cv_imovel"] != [int(i) for i in imoveis_reais]:
            raise SystemExit(f"Previsões do {t['modelo']} ({tipo}) não batem com a separação atual: rode tabpfn_v3.py.")
        previsoes[t["modelo"]] = {**dados, "api": TABPFN_VARIANTS.get(t["modelo"]) == "api"}
    return tentativas, previsoes


def dados_segmento(df, tipo, seed=42):
    """Separação treino/teste do tipo e treino com o histórico simulado (igual em todos os scripts)."""
    split = split_and_prepare_segment(df, tipo, seed=seed)
    treino = pd.concat([split["train"], simular(split["train"])], ignore_index=True)
    return split, treino, treino["imovel"].to_numpy(), (treino["simulado"] == 0).to_numpy()


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
            previsoes[(nome, alvo)] = pred[real]  # erro só nas linhas reais
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

    # Coerência com o índice (NÃO é medida de acerto): quanto o modelo diz que os imóveis
    # de teste valiam em jun/2021, comparado ao que o índice diria.
    hoje = prever(entradas, teste_tip)
    em2021 = prever(entradas, teste_tip.assign(ano=2021 + MEIO_DO_ANO))
    if escolhido.get("api"):
        # Teste do TabPFN pela API: previsões já gravadas por tabpfn_v3.py (treino completo).
        dados = tabpfn[escolhido["model"]]
        metricas = avaliar(teste_tip["preco"].to_numpy(), np.asarray(dados["teste_tipico"]))
        metricas_comp = avaliar(teste_comp["preco"].to_numpy(), np.asarray(dados["teste_completo"]))
        api = modelo_api(treino)
        hoje = np.asarray(dados["teste_tipico"])
        em2021 = api.predict(teste_tip.assign(ano=2021 + MEIO_DO_ANO))
    indice = nivel()
    esperado = np.array([1 / fator("2021-06", m, indice) for m in teste_tip["mes"]])
    coerencia = {
        "razao_modelo_2021_sobre_hoje": float(np.median(em2021 / hoje)),
        "razao_indice_2021_sobre_hoje": float(np.median(esperado)),
    }

    # Modelo final: todos os imóveis do tipo (limpeza com a base inteira) + simulados.
    final_real, rare, clip = prepare_segment_frame(df, tipo)
    final = pd.concat([final_real, simular(final_real)], ignore_index=True)
    entradas_finais = treinar_ensemble(final, membros, seed)
    return {
        # Treino do TabPFN pela API (enviado ao servidor da Prior Labs na hora da previsão).
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
        "coerencia_indice": coerencia,
        "rare_bairros": sorted(rare),
        "clip_bounds": {k: list(v) for k, v in clip.items()},
        "ensemble": entradas_finais,
        "selected": [{k: e[k] for k in ("model_name", "target", "weight")} for e in entradas_finais],
    }


# ---------------------------------------------------------------- previsão
def modelo_api(treino):
    """TabPFN 3.5 pela API da Prior Labs, com todas as linhas de treino (token em TABPFN_TOKEN)."""
    from tabpfn_model import TABPFN_VARIANTS, make_tabpfn

    versao = next(v for v in TABPFN_VARIANTS.values() if v == "api")
    modelo = make_tabpfn(NUM_V3, CATEGORICAL_COLUMNS, version=versao)
    modelo.max_train_rows = len(treino)
    return modelo.fit(treino, treino["preco"].to_numpy())


def separar_estimadores(art, pasta):
    """Grava cada estimador num arquivo comprimido e deixa no artefato só o nome do arquivo."""
    pasta = Path(pasta)
    for antigo in pasta.glob("*.pkl*"):
        antigo.unlink()
    for tipo, seg in art["segments"].items():
        for i, e in enumerate(seg["ensemble"]):
            nome = f"{tipo.lower()}_{i}.pkl"
            save_model(e.pop("estimator"), pasta / nome, compress=3)
            e["arquivo"] = nome


def load_v3(path=DEFAULT_MODEL_PATH):
    """Artefato sem os estimadores; `carregar_segmento` lê os de um tipo quando precisa."""
    art = load_model(path)
    if not isinstance(art, dict) or art.get("kind") != KIND:
        raise SystemExit("Arquivo do modelo v3 inválido. Rode: python scripts_predict/modelo_v3.py")
    art["pasta_estimadores"] = Path(path).parent / PASTA_ESTIMADORES
    return art


def carregar_segmento(art, tipo):
    seg = art["segments"][canonical_tipo(tipo)]
    for e in seg["ensemble"]:
        if "estimator" not in e:
            e["estimator"] = load_model(art["pasta_estimadores"] / e["arquivo"])
    return seg


def linhas_imovel(seg, anos, *, area_total, area_privada, bairro, tipo_imovel, quartos, banheiros, vagas):
    tipo = canonical_tipo(tipo_imovel)
    linhas = pd.DataFrame(
        [
            {
                "area_total": float(area_total or 0),
                "area_privada": float(area_privada or 0),
                "quartos": int(quartos or 0),
                "banheiros": int(banheiros or 0),
                "vagas": int(vagas or 0),
                "bairro": normalize_text(bairro),
                "tipo_imovel": tipo,
                "ano": float(ano),
            }
            for ano in anos
        ]
    )
    linhas = apply_rare_bairros(linhas, set(seg["rare_bairros"]))
    return apply_numeric_clip_bounds(linhas, {k: tuple(v) for k, v in seg["clip_bounds"].items()})


def prever_precos_v3(art, anos, api=None, **dados):
    """Preços estimados do imóvel em cada ano (ano + MEIO_DO_ANO = jun daquele ano).

    Se o modelo escolhido para o tipo é o TabPFN pela API, `api` é ele já ajustado
    (`modelo_api(seg["api"]["treino"])`). Sem `api`, ou se a API falhar, usa a reserva que
    roda no site. Retorna (preços, nome do modelo usado).
    """
    seg = carregar_segmento(art, dados["tipo_imovel"])
    linhas = linhas_imovel(seg, anos, **dados)
    if seg.get("api") and api is not None:
        try:
            return [float(v) for v in api.predict(linhas)], seg["api"]["model"]
        except Exception as exc:  # noqa: BLE001 - qualquer falha da API cai na reserva
            print(f"[aviso] TabPFN pela API falhou, usando a reserva: {exc}", flush=True)
    return [float(v) for v in prever(seg["ensemble"], linhas)], seg["reserva"]["model"]


def prever_preco_v3(art, *, ano, **dados):
    """Preço estimado do imóvel no ano informado, com o modelo que roda no site."""
    return prever_precos_v3(art, [ano], **dados)[0][0]


# ------------------------------------------------------------------ saídas
def exportar_dataset(df, db_path, csv_path):
    """Dataset v3 completo (reais + simulados) em SQLite e, para o site, em CSV."""
    base = df[df["tipo_imovel"].isin(SEGMENT_TYPES)]
    completo = pd.concat([base, simular(base)], ignore_index=True)
    # Preço anunciado original (só nos reais) ao lado do preço no mês de referência.
    indice = nivel()
    reais = completo["simulado"] == 0
    completo["preco_anunciado"] = np.where(
        reais,
        completo["preco"] / np.array([fator(m, r, indice) for m, r in zip(completo["mes_anuncio"], completo["mes"])]),
        np.nan,
    )
    completo.loc[~reais, "mes_anuncio"] = None
    colunas = ["imovel", "simulado", "ano", "mes", "tipo_imovel", "bairro", "area_total",
               "area_privada", "quartos", "banheiros", "vagas", "preco", "mes_anuncio", "preco_anunciado"]
    completo = completo[colunas].sort_values(["imovel", "ano"])
    completo["preco"] = completo["preco"].round(0)
    completo["preco_anunciado"] = completo["preco_anunciado"].round(0)
    with sqlite3.connect(db_path) as conn:
        completo.to_sql("imoveis_v3", conn, if_exists="replace", index=False)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    completo.to_csv(csv_path, index=False, encoding="utf-8")
    return int((completo["simulado"] == 0).sum()), int((completo["simulado"] == 1).sum())


def relatorio(art, v1):
    idx = art["indice"]
    linhas = [
        "# Modelo de preço v3: anúncios reais + histórico simulado (IPCA+INCC)",
        "",
        "Gerado por `scripts_predict/modelo_v3.py`. Para cada imóvel real foram simulados os",
        f"anos {ANOS_SIMULADOS[0]} a {ANOS_SIMULADOS[-1]}, deflacionando o preço pelo índice",
        f"IPCA+INCC realizado (média de 10 anos: {idx['media_anual_pct']:.2f}% ao ano). Erros",
        "medidos só nos preços reais do teste, que o modelo nunca viu.",
        "",
        "| Tipo | Linhas reais (treino) | Linhas simuladas (treino) | MAE v3 | MAE v1 | R² v3 | R² v1 | MdAPE v3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tipo, s in art["segments"].items():
        m = s["metrics"]
        ref = v1.get(tipo, {})
        linhas.append(
            f"| {tipo} | {s['linhas_reais_treino']} | {s['linhas_simuladas_treino']} | "
            f"R$ {m['mae']:,.0f} | R$ {ref.get('mae', float('nan')):,.0f} | {m['r2']:.3f} | "
            f"{ref.get('r2', float('nan')):.3f} | {m['erro_mediano_pct']:.1f}% |".replace(",", ".")
        )
    linhas += ["", "## Coerência com o índice (não é medida de acerto)", "",
               "Quanto o modelo diz que os imóveis de teste valiam em jun/2021, como fração do valor",
               "de hoje (mediana), e quanto o índice diria:", "",
               "| Tipo | Modelo | Índice IPCA+INCC |", "|---|---:|---:|"]
    for tipo, s in art["segments"].items():
        c = s["coerencia_indice"]
        linhas.append(f"| {tipo} | {c['razao_modelo_2021_sobre_hoje']:.3f} | {c['razao_indice_2021_sobre_hoje']:.3f} |")
    linhas += ["", "## Escolha do modelo (validação cruzada por imóvel, erro só nas linhas reais)", "",
               "Critério: menor erro percentual mediano (MdAPE). Empate (mesmo MdAPE com 1 casa",
               "decimal) é decidido pelo menor MAE. O ensemble (média ponderada dos 3 melhores que",
               "rodam no site, peso 1/MdAPE) concorre como mais um candidato. O TabPFN local entra",
               "no ranking, mas não pode ser publicado (PyTorch, minutos por previsão em CPU); o",
               "TabPFN 3.5 pela API da Prior Labs pode. Se a API não responder, o site usa a reserva:",
               "o melhor entre os que rodam no próprio site.", ""]
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
        if s.get("api"):
            linhas += ["", f"Publicado (★): {s['api']['model']}, com reserva {s['reserva']['model']}."]
        linhas += ["", "Roda no site (★ ou reserva): " + ", ".join(
            f"{x['model_name']} ({x['target']}, peso {x['weight']:.2f})" for x in s["selected"]), ""]
        if s["tabpfn_tentativas"]:
            linhas += ["Tentativas do TabPFN (`tabpfn_v3.py`):", "",
                       "| Variante | Linhas de treino | Resultado | Tempo |", "|---|---|---|---:|"]
            for t in s["tabpfn_tentativas"]:
                status = {"ok": "terminou", "cortado": "cortado por tempo", "erro": "erro", "pulado": "pulado"}[t["status"]]
                if t["detalhe"]:
                    status += f": {t['detalhe']}"
                linhas.append(f"| {t['modelo']} | {t['linhas'] or '—'} | {status} | {t['minutos']:.0f} min |")
            linhas.append("")
    return "\n".join(linhas) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Modelo de preço v3 (histórico simulado IPCA+INCC).")
    parser.add_argument("--normalized-db", default=str(DEFAULT_NORMALIZED_DB))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--v1-model", default="modelos/preco_imovel_modelo_hibrido_rapido.pkl")
    args = parser.parse_args()

    df = carregar_reais(args.normalized_db)
    segmentos = {}
    for tipo in SEGMENT_TYPES:
        print(f"Treinando {tipo}...", flush=True)
        segmentos[tipo] = segmento(df, tipo)
    art = {
        "kind": KIND,
        "indice": resumo(),
        "anos_simulados": ANOS_SIMULADOS,
        "segments": segmentos,
        "trained_at": datetime.now().isoformat(),
    }
    separar_estimadores(art, Path(args.model_path).parent / PASTA_ESTIMADORES)
    save_model(art, Path(args.model_path))
    reais, simulados = exportar_dataset(df, DEFAULT_DATASET_DB, DEFAULT_DATASET_CSV)

    from imoveis_ml_hibrido import load_artifact

    v1 = {t: s["metrics"] for t, s in load_artifact(Path(args.v1_model))["segments"].items()}
    Path(DEFAULT_REPORT).write_text(relatorio(art, v1), encoding="utf-8")
    print(f"Modelo v3 salvo em {args.model_path}; dataset: {reais} reais + {simulados} simulados.")
    for tipo, s in segmentos.items():
        m = s["metrics"]
        print(f"[{tipo}] v3 MAE {m['mae']:,.0f} R2 {m['r2']:.3f} | v1 MAE {v1[tipo]['mae']:,.0f} R2 {v1[tipo]['r2']:.3f}")
        print(f"        coerência 2021/hoje: modelo {s['coerencia_indice']['razao_modelo_2021_sobre_hoje']:.3f} "
              f"índice {s['coerencia_indice']['razao_indice_2021_sobre_hoje']:.3f}")
    print(json.dumps({"report": str(DEFAULT_REPORT)}))


if __name__ == "__main__":
    import modelo_v3

    modelo_v3.main()
