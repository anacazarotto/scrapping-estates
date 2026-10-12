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
- a validação cruzada agrupa por imóvel (GroupKFold) e mede o erro só nas linhas reais.

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
# Random Forest e MLP/SVM ficam de fora: com ~15 mil linhas, o Random Forest gera um
# arquivo de centenas de MB (inviável no site) e MLP/SVM foram os piores no v1.
CANDIDATOS = ["CatBoost", "XGBoost", "Gradient Boosting", "Ridge"]
KIND = "preco_v3_historico_simulado"


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
    mes = pd.to_datetime(df["data_insercao"]).dt.to_period("M")
    df["mes"] = mes.astype(str)
    df["ano"] = mes.dt.year + (mes.dt.month - 0.5) / 12
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
    pipe = Pipeline([("pre", preprocessador()), ("modelo", make_model_factories(seed=seed)[nome]())])
    if usa_log:
        return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=safe_expm1, check_inverse=False)
    return pipe


def treinar_ensemble(treino, ranking, seed=42):
    top = ranking[:3]
    pesos = np.array([1 / r["mae"] for r in top])
    pesos = pesos / pesos.sum()
    entradas = []
    for peso, r in zip(pesos, top):
        modelo = montar(r["model"], r["target"] == "log(preco)", seed).fit(treino[FEATURES_V3], treino["preco"])
        entradas.append({"model_name": r["model"], "target": r["target"], "weight": float(peso), "estimator": modelo})
    return entradas


def prever(entradas, X):
    return sum(e["weight"] * np.asarray(e["estimator"].predict(X[FEATURES_V3]), dtype=float) for e in entradas)


def avaliar(y, pred):
    m = calculate_metrics(y, pred)
    m.pop("mape", None)
    m["erro_mediano_pct"] = float(np.median(np.abs(pred - y) / y) * 100)
    return m


def segmento(df, tipo, seed=42):
    split = split_and_prepare_segment(df, tipo, seed=seed)
    treino_real = split["train"]
    treino = pd.concat([treino_real, simular(treino_real)], ignore_index=True)
    grupos = treino["imovel"].to_numpy()
    real = (treino["simulado"] == 0).to_numpy()

    ranking = []
    cv = GroupKFold(n_splits=5)
    for nome in CANDIDATOS:
        for usa_log in (False, True):
            pred = cross_val_predict(montar(nome, usa_log, seed), treino[FEATURES_V3], treino["preco"], cv=cv, groups=grupos)
            y = treino["preco"].to_numpy()
            ranking.append(
                {
                    "model": nome,
                    "target": "log(preco)" if usa_log else "preco",
                    "mae": float(np.mean(np.abs(pred[real] - y[real]))),  # só linhas reais
                }
            )
            print(f"  [{tipo}] {nome} ({ranking[-1]['target']}): MAE CV real {ranking[-1]['mae']:,.0f}", flush=True)
    ranking.sort(key=lambda r: r["mae"])

    entradas = treinar_ensemble(treino, ranking, seed)
    teste_tip, teste_comp = split["test_typical"], split["test_full"]
    metricas = avaliar(teste_tip["preco"].to_numpy(), prever(entradas, teste_tip))
    metricas_comp = avaliar(teste_comp["preco"].to_numpy(), prever(entradas, teste_comp))

    # Coerência com o índice (NÃO é medida de acerto): quanto o modelo diz que os imóveis
    # de teste valiam em jun/2021, comparado ao que o índice diria.
    hoje = prever(entradas, teste_tip)
    em2021 = prever(entradas, teste_tip.assign(ano=2021 + MEIO_DO_ANO))
    indice = nivel()
    esperado = np.array([1 / fator("2021-06", m, indice) for m in teste_tip["mes"]])
    coerencia = {
        "razao_modelo_2021_sobre_hoje": float(np.median(em2021 / hoje)),
        "razao_indice_2021_sobre_hoje": float(np.median(esperado)),
    }

    # Modelo final: todos os imóveis do tipo (limpeza com a base inteira) + simulados.
    final_real, rare, clip = prepare_segment_frame(df, tipo)
    final = pd.concat([final_real, simular(final_real)], ignore_index=True)
    entradas_finais = treinar_ensemble(final, ranking, seed)
    return {
        "linhas_reais_treino": int(len(treino_real)),
        "linhas_simuladas_treino": int((treino["simulado"] == 1).sum()),
        "teste_tipico": int(len(teste_tip)),
        "ranking": ranking,
        "metrics": metricas,
        "metrics_full": metricas_comp,
        "coerencia_indice": coerencia,
        "rare_bairros": sorted(rare),
        "clip_bounds": {k: list(v) for k, v in clip.items()},
        "ensemble": entradas_finais,
        "selected": [{k: e[k] for k in ("model_name", "target", "weight")} for e in entradas_finais],
    }


# ---------------------------------------------------------------- previsão
def load_v3(path=DEFAULT_MODEL_PATH):
    art = load_model(path)
    if not isinstance(art, dict) or art.get("kind") != KIND:
        raise SystemExit("Arquivo do modelo v3 inválido. Rode: python scripts_predict/modelo_v3.py")
    return art


def prever_preco_v3(art, *, ano, area_total, area_privada, bairro, tipo_imovel, quartos, banheiros, vagas):
    """Preço estimado do imóvel no ano informado (ano + MEIO_DO_ANO = jun daquele ano)."""
    tipo = canonical_tipo(tipo_imovel)
    seg = art["segments"][tipo]
    linha = pd.DataFrame(
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
        ]
    )
    linha = apply_rare_bairros(linha, set(seg["rare_bairros"]))
    linha = apply_numeric_clip_bounds(linha, {k: tuple(v) for k, v in seg["clip_bounds"].items()})
    return float(prever(seg["ensemble"], linha)[0])


# ------------------------------------------------------------------ saídas
def exportar_dataset(df, db_path, csv_path):
    """Dataset v3 completo (reais + simulados) em SQLite e, para o site, em CSV."""
    base = df[df["tipo_imovel"].isin(SEGMENT_TYPES)]
    completo = pd.concat([base, simular(base)], ignore_index=True)
    colunas = ["imovel", "simulado", "ano", "mes", "tipo_imovel", "bairro", "area_total",
               "area_privada", "quartos", "banheiros", "vagas", "preco"]
    completo = completo[colunas].sort_values(["imovel", "ano"])
    completo["preco"] = completo["preco"].round(0)
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
        "| Tipo | Linhas reais (treino) | Linhas simuladas (treino) | MAE v3 | MAE v1 | R² v3 | R² v1 | Erro mediano v3 |",
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
    linhas += ["", "## Ensemble escolhido (validação cruzada por imóvel, erro só nas linhas reais)", ""]
    for tipo, s in art["segments"].items():
        linhas.append(f"- **{tipo}:** " + ", ".join(f"{e['model_name']} ({e['target']}, peso {e['weight']:.2f})" for e in s["selected"]))
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
