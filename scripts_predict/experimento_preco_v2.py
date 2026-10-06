"""Experimento v2 do modelo de preço: ideias para aproximar mais o valor, contra o v1.

Pedido do orientador: buscar uma forma de o valor estimado ficar mais próximo, usando
o IPCA ou outra referência no treino, sem perder o modelo atual (v1). Cada variante usa
exatamente o protocolo do v1 (`imoveis_ml_hibrido.py`): mesma separação treino/teste
(semente 42), mesma limpeza aprendida só no treino, ranking por validação cruzada de 5
partes, ensemble dos 3 melhores com peso 1/MAE e avaliação no teste típico.

Variantes:
- v1:        o modelo atual (referência);
- bairro:    + mediana do preço/m² do bairro, aprendida só no treino de cada dobra;
- m2:        prevê o preço/m² e multiplica pela área;
- bairro+m2: as duas anteriores juntas;
- ipca:      preços corrigidos pelo IPCA até ago/2026 (mês da última coleta) a partir
             do mês em que o imóvel apareceu; a previsão volta para a data do anúncio
             antes de medir o erro, para comparar com o v1 na mesma escala.

Candidatos de cada variante: CatBoost, XGBoost, Random Forest, Gradient Boosting e
Ridge, com alvo direto e em log (os que ficaram no topo do v1).

    DISABLE_TABPFN=1 python scripts_predict/experimento_preco_v2.py
"""

import argparse
import json
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imoveis_ml_hibrido import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    SEGMENT_TYPES,
    calculate_metrics,
    canonical_tipo,
    format_currency,
    make_model_factories,
    normalize_text,
    safe_expm1,
    split_and_prepare_segment,
)
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

DEFAULT_NORMALIZED_DB = Path("imoveis_normalizados.db")
DEFAULT_IPCA = Path("dados/ipca.json")
DEFAULT_REPORT = Path("docs/reports/experimento_preco_v2.md")
DEFAULT_JSON = Path("docs/reports/experimento_preco_v2.json")
CANDIDATOS = ["CatBoost", "XGBoost", "Random Forest", "Gradient Boosting", "Ridge"]
VARIANTES = ["v1", "bairro", "m2", "bairro+m2", "ipca"]
MES_REFERENCIA = (2026, 8)


# ------------------------------------------------------------------ dados
def indice_ipca(path):
    """Índice acumulado do IPCA por (ano, mês)."""
    nivel, idx = 1.0, {}
    for m in json.loads(Path(path).read_text(encoding="utf-8")):
        _, mes, ano = m["data"].split("/")
        nivel *= 1 + m["valor"] / 100
        idx[(int(ano), int(mes))] = nivel
    return idx


def carregar(normalized_db, ipca_path):
    """Mesmos filtros de `load_training_frame`, mais o fator do IPCA de cada imóvel."""
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
    idx = indice_ipca(ipca_path)
    ref = idx.get(MES_REFERENCIA, max(idx.values()))
    meses = pd.to_datetime(df["data_insercao"]).dt.to_period("M")
    df["fator_ipca"] = [ref / idx.get((p.year, p.month), ref) for p in meses]
    return df.drop(columns="data_insercao")


# ------------------------------------------------------------- variantes
def area_ref(X):
    X = pd.DataFrame(X)
    area = X["area_privada"].where(X["area_privada"] > 0, X["area_total"])
    return area.where(area > 0, 1.0).to_numpy(dtype=float)


class BairroPrecoM2(BaseEstimator, TransformerMixin):
    """Acrescenta a mediana do preço/m² do bairro, aprendida no fit (só com o treino)."""

    def __init__(self, alvo="preco", min_imoveis=5):
        self.alvo = alvo  # "preco", "log_preco", "pm2" ou "log_pm2": o que chega em y
        self.min_imoveis = min_imoveis

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        if self.alvo.startswith("log"):
            y = np.expm1(y)
        pm2 = y if self.alvo.endswith("pm2") else y / area_ref(X)
        grupos = pd.DataFrame({"b": pd.DataFrame(X)["bairro"].to_numpy(), "v": pm2})
        stats = grupos.groupby("b")["v"].agg(["median", "size"])
        self.mapa_ = stats.loc[stats["size"] >= self.min_imoveis, "median"].to_dict()
        self.geral_ = float(np.median(pm2))
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X["bairro_pm2"] = X["bairro"].map(self.mapa_).fillna(self.geral_).astype(float)
        return X


class PorM2(BaseEstimator, RegressorMixin):
    """Treina no preço/m² e devolve o preço (preço/m² previsto x área)."""

    def __init__(self, regressor=None):
        self.regressor = regressor

    def fit(self, X, y):
        self.regressor_ = clone(self.regressor).fit(X, np.asarray(y, dtype=float) / area_ref(X))
        return self

    def predict(self, X):
        return self.regressor_.predict(X) * area_ref(X)


def preprocessador(extras):
    numericas = Pipeline([("i", SimpleImputer(strategy="constant", fill_value=0)), ("s", StandardScaler())])
    categoricas = Pipeline(
        [
            ("i", SimpleImputer(strategy="constant", fill_value="")),
            ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", numericas, NUMERIC_COLUMNS + extras), ("cat", categoricas, CATEGORICAL_COLUMNS)]
    )


def montar(nome, variante, usa_log, seed=42):
    modelo = make_model_factories(seed=seed)[nome]()
    por_m2 = variante in ("m2", "bairro+m2")
    com_bairro = variante in ("bairro", "bairro+m2")
    passos = []
    if com_bairro:
        alvo = ("log_" if usa_log else "") + ("pm2" if por_m2 else "preco")
        passos.append(("bairro_pm2", BairroPrecoM2(alvo=alvo)))
    passos += [("pre", preprocessador(["bairro_pm2"] if com_bairro else [])), ("modelo", modelo)]
    pipe = Pipeline(passos)
    inverso = np.expm1 if por_m2 else safe_expm1
    if usa_log:
        pipe = TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=inverso, check_inverse=False)
    return PorM2(pipe) if por_m2 else pipe


# ------------------------------------------------------------ avaliação
def avaliar_variante(train, test, variante, seed=42):
    corrige = variante == "ipca"
    y_tr = train["preco"].to_numpy(dtype=float) * (train["fator_ipca"].to_numpy() if corrige else 1.0)
    base = "v1" if corrige else variante
    X_tr, X_te = train[FEATURE_COLUMNS], test[FEATURE_COLUMNS]
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    ranking = []
    for nome in CANDIDATOS:
        for usa_log in (False, True):
            pred = cross_val_predict(montar(nome, base, usa_log, seed), X_tr, y_tr, cv=cv)
            ranking.append((float(np.mean(np.abs(pred - y_tr))), nome, usa_log))
    ranking.sort()
    top = ranking[:3]
    pesos = np.array([1 / mae for mae, _, _ in top])
    pesos /= pesos.sum()
    pred = sum(
        p * montar(nome, base, usa_log, seed).fit(X_tr, y_tr).predict(X_te)
        for p, (_, nome, usa_log) in zip(pesos, top)
    )
    if corrige:  # volta da data de referência para a data do anúncio
        pred = pred / test["fator_ipca"].to_numpy()
    y_te = test["preco"].to_numpy(dtype=float)
    m = calculate_metrics(y_te, pred)
    return {
        "mae": m["mae"],
        "rmse": m["rmse"],
        "r2": m["r2"],
        "erro_mediano_pct": float(np.median(np.abs(pred - y_te) / y_te) * 100),
        "mae_cv_melhor": top[0][0],
        "ensemble": [f"{nome}{' (log)' if usa_log else ''}" for _, nome, usa_log in top],
    }


def relatorio(resultados, fator_min, fator_max):
    linhas = [
        "# Experimento v2 do modelo de preço",
        "",
        "Gerado por `scripts_predict/experimento_preco_v2.py`. Mesmo protocolo do v1: teste",
        "separado antes da limpeza (semente 42), regras só do treino, ranking por validação",
        "cruzada de 5 partes, ensemble dos 3 melhores com peso 1/MAE, teste típico.",
        "",
        f"Correção pelo IPCA: fator entre {fator_min:.4f} e {fator_max:.4f} (preços de mar a",
        "ago/2026 levados a ago/2026).",
        "",
    ]
    for tipo, linhas_tipo in resultados.items():
        v1 = linhas_tipo["v1"]["mae"]
        linhas += [
            f"## {tipo}",
            "",
            "| Variante | MAE no teste | Diferença para o v1 | R² | Erro mediano | Ensemble |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for variante, r in linhas_tipo.items():
            dif = (r["mae"] / v1 - 1) * 100
            linhas.append(
                f"| {variante} | {format_currency(r['mae'])} | {dif:+.1f}% | {r['r2']:.3f} | "
                f"{r['erro_mediano_pct']:.1f}% | {', '.join(r['ensemble'])} |"
            )
        linhas.append("")
    return "\n".join(linhas)


def main():
    parser = argparse.ArgumentParser(description="Experimento v2 do modelo de preço.")
    parser.add_argument("--normalized-db", default=str(DEFAULT_NORMALIZED_DB))
    parser.add_argument("--ipca", default=str(DEFAULT_IPCA))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT))
    parser.add_argument("--json-path", default=str(DEFAULT_JSON))
    args = parser.parse_args()

    df = carregar(args.normalized_db, args.ipca)
    resultados = {}
    for tipo in SEGMENT_TYPES:
        split = split_and_prepare_segment(df, tipo, seed=42)
        resultados[tipo] = {}
        for variante in VARIANTES:
            r = avaliar_variante(split["train"], split["test_typical"], variante)
            resultados[tipo][variante] = r
            print(f"[{tipo}] {variante}: MAE {r['mae']:,.0f} R2 {r['r2']:.3f}", flush=True)
    Path(args.json_path).write_text(json.dumps(resultados, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(args.report_path).write_text(
        relatorio(resultados, df["fator_ipca"].min(), df["fator_ipca"].max()), encoding="utf-8"
    )
    print(f"Relatório salvo em {args.report_path}")


if __name__ == "__main__":
    main()
