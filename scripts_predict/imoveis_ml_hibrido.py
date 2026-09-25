import argparse
import sqlite3
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from model_io import load_model, save_model
from tabpfn_model import TABPFN_NAME, make_tabpfn, tabpfn_enabled
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - optional dependency
    XGBRegressor = None

try:
    from sklearn_compat import CatBoostRegressor
except ImportError:  # pragma: no cover - optional dependency
    CatBoostRegressor = None

DEFAULT_NORMALIZED_DB = Path("imoveis_normalizados.db")
DEFAULT_MODEL_PATH = Path("modelos/preco_imovel_modelo_hibrido.pkl")
DEFAULT_REPORT_PATH = Path("docs/modelo_hibrido.md")
SEGMENT_TYPES = ("Casa", "Apartamento")
RARE_BAIRRO_MIN_COUNT = 10
CLIP_QUANTILES = (0.02, 0.98)
TOP_MODELS_PER_SEGMENT = 3
CV_FOLDS = 5
MIN_SEGMENT_ROWS = 30

FEATURE_COLUMNS = [
    "area_total",
    "area_privada",
    "quartos",
    "banheiros",
    "vagas",
    "bairro",
    "tipo_imovel",
]
NUMERIC_COLUMNS = ["area_total", "area_privada", "quartos", "banheiros", "vagas"]
CATEGORICAL_COLUMNS = ["bairro", "tipo_imovel"]


MAX_LOG_PRICE = 21.0  # e^21 ≈ R$ 1,3 bilhão: teto para previsões em log


def safe_expm1(values):
    """Inverso do log1p com teto.

    Modelos lineares em log(preço) extrapolam para imóveis com área muito fora do
    padrão (ex.: terrenos/chácaras enormes) e o expm1 explode para valores
    astronômicos, destruindo MAE/RMSE. O teto mantém a previsão num intervalo real.
    """
    return np.expm1(np.clip(values, 0.0, MAX_LOG_PRICE))


def normalize_text(value):
    if value is None:
        return ""
    text = unicodedata.normalize("NFD", str(value)).encode("ascii", "ignore").decode()
    return " ".join(text.lower().strip().split())


def canonical_tipo(value):
    text = normalize_text(value)
    if not text:
        return ""
    if "lanc" in text:
        return "Lançamento"
    if "comercial" in text or text == "sala" or "sala/" in text:
        return "Comercial"
    if "studio" in text or "loft" in text or "flat" in text:
        return "Studio"
    if "cobert" in text or "penthouse" in text:
        return "Cobertura"
    if "apart" in text:
        return "Apartamento"
    if "sobrado" in text or "geminad" in text or "casa" in text:
        return "Casa"
    if "terreno" in text or "lote" in text:
        return "Terreno"
    if "chacara" in text or "sitio" in text or "area rural" in text or text == "area":
        return "Chácara"
    if "predio" in text:
        return "Prédio"
    return "Outros"


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    denom = np.where(y_true == 0, np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def format_currency(value):
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def load_training_frame(normalized_db):
    conn = sqlite3.connect(normalized_db)
    df = pd.read_sql_query(
        """
        SELECT
            preco,
            bairro,
            tipo_imovel,
            area_total,
            area_privada,
            quartos,
            banheiros,
            vagas
        FROM imoveis_normalizados
        WHERE preco IS NOT NULL AND preco > 0
        """,
        conn,
    )
    conn.close()

    if df.empty:
        raise RuntimeError("Nenhum registro válido encontrado para treinamento.")

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df = df[(df["area_total"] > 0) | (df["area_privada"] > 0)]
    df["bairro"] = df["bairro"].fillna("").map(normalize_text)
    df["tipo_imovel"] = df["tipo_imovel"].fillna("").map(canonical_tipo)
    df["preco"] = pd.to_numeric(df["preco"], errors="coerce")
    df = df[df["preco"].notna() & (df["preco"] > 0)]
    return df


def compute_rare_bairros(df, min_count=RARE_BAIRRO_MIN_COUNT):
    if df.empty:
        return set()
    counts = df["bairro"].fillna("").astype(str).value_counts()
    return set(counts[counts < min_count].index)


def apply_rare_bairros(df, rare_bairros, rare_label="outros"):
    work = df.copy()
    if not rare_bairros:
        return work
    work["bairro"] = work["bairro"].where(~work["bairro"].isin(rare_bairros), rare_label)
    return work


def compute_numeric_clip_bounds(df, columns=NUMERIC_COLUMNS, quantiles=CLIP_QUANTILES):
    bounds = {}
    if df.empty:
        return bounds
    lower_q, upper_q = quantiles
    for column in columns:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        series = series[series > 0]
        if len(series) < 8:
            continue
        lower = float(series.quantile(lower_q))
        upper = float(series.quantile(upper_q))
        if lower <= upper:
            bounds[column] = (lower, upper)
    return bounds


def apply_numeric_clip_bounds(df, bounds):
    work = df.copy()
    if not bounds:
        return work
    for column, (lower, upper) in bounds.items():
        work[column] = pd.to_numeric(work[column], errors="coerce").clip(lower=lower, upper=upper)
    return work


def remove_outliers_iqr(df, columns=("preco", "preco_m2"), factor=1.5):
    if df.empty:
        return df.copy()
    work = df.copy()
    keep = pd.Series(True, index=work.index)
    for column in columns:
        if column not in work.columns:
            continue
        series = pd.to_numeric(work[column], errors="coerce").dropna()
        series = series[series > 0]
        if len(series) < 8:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            continue
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        col_values = pd.to_numeric(work[column], errors="coerce")
        keep &= col_values.between(lower, upper) | col_values.isna()
    return work[keep].copy()


def make_preprocessor():
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="")),
            ("onehot", encoder),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLUMNS),
            ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
        ]
    )


def make_model_factories(seed=42):
    factories = {
        "Regressão Linear Múltipla": lambda: LinearRegression(),
        "Ridge": lambda: Ridge(alpha=5.0),
        "Lasso": lambda: Lasso(alpha=0.0005, max_iter=20000, random_state=seed),
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=400,
            random_state=seed,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
        "Gradient Boosting": lambda: GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=3,
            random_state=seed,
        ),
        "Redes Neurais Artificiais (MLP)": lambda: MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=600,
            early_stopping=True,
            random_state=seed,
        ),
        "Máquinas de Vetores de Suporte (SVM)": lambda: SVR(
            kernel="rbf",
            C=20.0,
            epsilon=0.1,
            gamma="scale",
        ),
    }
    if XGBRegressor is not None:
        factories["XGBoost"] = lambda: XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
        )
    if CatBoostRegressor is not None:
        factories["CatBoost"] = lambda: CatBoostRegressor(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            random_seed=seed,
            verbose=False,
        )
    if tabpfn_enabled():
        factories[TABPFN_NAME] = lambda: make_tabpfn(NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, seed=seed)
    return factories


def build_regressor(name, *, seed=42, use_log=True):
    factories = make_model_factories(seed=seed)
    if name not in factories:
        raise ValueError(f"Modelo não suportado: {name}")
    if name == TABPFN_NAME:
        # TabPFN faz a própria codificação (sem one-hot/padronização).
        pipeline = Pipeline([("model", factories[name]())])
    else:
        pipeline = Pipeline(
            [
                ("preprocess", make_preprocessor()),
                ("model", factories[name]()),
            ]
        )
    if not use_log:
        return pipeline
    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=safe_expm1,
        check_inverse=False,
    )


def add_price_per_m2(df):
    work = df.copy()
    area_ref = work["area_privada"].where(work["area_privada"] > 0, work["area_total"])
    area_ref = area_ref.where(area_ref > 0)
    work["preco_m2"] = work["preco"] / area_ref
    return work


def compute_iqr_bounds(df, columns=("preco", "preco_m2"), factor=1.5):
    """Aprende os limites do IQR (mesma regra de remove_outliers_iqr), sem aplicá-los."""
    bounds = {}
    if df.empty:
        return bounds
    for column in columns:
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        series = series[series > 0]
        if len(series) < 8:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            continue
        bounds[column] = (float(q1 - factor * iqr), float(q3 + factor * iqr))
    return bounds


def within_iqr_bounds(df, bounds):
    """Máscara booleana: True para linhas dentro dos limites aprendidos."""
    keep = pd.Series(True, index=df.index)
    for column, (lower, upper) in bounds.items():
        values = pd.to_numeric(df[column], errors="coerce")
        keep &= values.between(lower, upper) | values.isna()
    return keep


def prepare_segment_frame(df, property_type):
    """Preparo com a base INTEIRA do segmento.

    Usado apenas para treinar o modelo FINAL (o que vai para produção). Não deve ser
    usado para medir desempenho, porque aprende limites com todas as linhas.
    """
    subset = df[df["tipo_imovel"].map(canonical_tipo) == property_type].copy()
    subset = subset[subset["preco"] > 0].copy()
    if subset.empty:
        return subset, set(), {}

    rare_bairros = compute_rare_bairros(subset, min_count=RARE_BAIRRO_MIN_COUNT)
    clip_bounds = compute_numeric_clip_bounds(subset, columns=NUMERIC_COLUMNS, quantiles=CLIP_QUANTILES)
    subset = apply_rare_bairros(subset, rare_bairros)
    subset = apply_numeric_clip_bounds(subset, clip_bounds)
    subset = add_price_per_m2(subset)
    subset = remove_outliers_iqr(subset, columns=("preco", "preco_m2"), factor=1.5)
    subset = subset[subset["preco"] > 0].copy()
    return subset, rare_bairros, clip_bounds


def split_and_prepare_segment(df, property_type, seed=42, test_size=0.2):
    """Divide em treino/teste ANTES de qualquer pré-processamento.

    Bairros raros, limites de clipagem e limites de outlier são aprendidos SOMENTE no
    treino e depois aplicados ao teste. Outliers só são removidos do treino; o teste é
    devolvido em duas versões:
      - test_full:    todas as linhas do teste (cenário realista, com anúncios estranhos);
      - test_typical: só as linhas dentro da faixa típica de preço (comparável aos
                      relatórios antigos, que mediam apenas imóveis "sem outliers").
    """
    raw = df[df["tipo_imovel"].map(canonical_tipo) == property_type].copy()
    raw = raw[raw["preco"] > 0].copy()
    if len(raw) < MIN_SEGMENT_ROWS:
        return None

    train_raw, test_raw = train_test_split(raw, test_size=test_size, random_state=seed)

    rare_bairros = compute_rare_bairros(train_raw, min_count=RARE_BAIRRO_MIN_COUNT)
    clip_bounds = compute_numeric_clip_bounds(train_raw, columns=NUMERIC_COLUMNS, quantiles=CLIP_QUANTILES)

    def transform(part):
        part = apply_rare_bairros(part, rare_bairros)
        part = apply_numeric_clip_bounds(part, clip_bounds)
        return add_price_per_m2(part)

    train = transform(train_raw)
    test_full = transform(test_raw)

    iqr_bounds = compute_iqr_bounds(train, columns=("preco", "preco_m2"), factor=1.5)
    train = train[within_iqr_bounds(train, iqr_bounds)].copy()
    train = train[train["preco"] > 0].copy()
    test_typical = test_full[within_iqr_bounds(test_full, iqr_bounds)].copy()

    return {
        "train": train,
        "test_full": test_full,
        "test_typical": test_typical,
        "iqr_bounds": iqr_bounds,
    }


def empty_metrics():
    return {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "mape": 0.0}


def evaluate_candidates_cv(X_train, y_train, seed=42, cv_folds=CV_FOLDS):
    """Ranking dos candidatos por validação cruzada DENTRO do treino.

    O conjunto de teste não participa da escolha dos modelos nem dos pesos.
    """
    folds = int(max(2, min(cv_folds, len(X_train) // 10)))
    cv = KFold(n_splits=folds, shuffle=True, random_state=seed)
    rows = []
    factories = make_model_factories(seed=seed)
    for use_log in (False, True):
        target_label = "log(preco)" if use_log else "preco"
        for name in factories:
            model = build_regressor(name, seed=seed, use_log=use_log)
            preds = cross_val_predict(model, X_train, y_train, cv=cv)
            rows.append({"model": name, "target": target_label, **calculate_metrics(y_train, preds)})
    rows.sort(key=lambda item: (item["mae"], item["rmse"]))
    return rows, folds


def fit_weighted_ensemble(X, y, selected_rows, seed=42):
    entries = []
    mae_values = [max(float(row["mae"]), 1e-9) for row in selected_rows]
    inv = [1.0 / value for value in mae_values]
    total = sum(inv) or 1.0
    weights = [value / total for value in inv]
    for row, weight in zip(selected_rows, weights):
        use_log = row["target"] == "log(preco)"
        model = build_regressor(row["model"], seed=seed, use_log=use_log)
        model.fit(X, y)
        entries.append(
            {
                "model_name": row["model"],
                "target": row["target"],
                "weight": float(weight),
                "mae_validacao": float(row["mae"]),
                "estimator": model,
            }
        )
    return entries


def predict_weighted_ensemble(entries, X):
    weighted = None
    for entry in entries:
        pred = np.asarray(entry["estimator"].predict(X), dtype=float)
        current = pred * float(entry["weight"])
        weighted = current if weighted is None else (weighted + current)
    return weighted


def score_on(entries, test_df):
    """Retorna (métricas, y_verdadeiro, y_previsto) do ensemble em um conjunto de teste."""
    if test_df.empty:
        return empty_metrics(), np.array([]), np.array([])
    X_test = test_df[FEATURE_COLUMNS].copy()
    y_test = test_df["preco"].astype(float).to_numpy()
    pred = predict_weighted_ensemble(entries, X_test)
    return calculate_metrics(y_test, pred), y_test, np.asarray(pred, dtype=float)


def train_hybrid(df, seed=42, test_size=0.2, cv_folds=CV_FOLDS):
    """Treina o híbrido com avaliação sem vazamento.

    Fluxo por segmento (Casa / Apartamento):
      1. split treino/teste antes de qualquer pré-processamento;
      2. pré-processamento aprendido só no treino;
      3. ranking dos candidatos por validação cruzada no treino;
      4. ensemble (top-N ponderado por 1/MAE de CV) treinado SÓ no treino;
      5. métricas medidas no teste (nunca visto);
      6. modelo FINAL (o que vai para produção) refeito com todos os dados.
    """
    segments = {}
    typ_true, typ_pred = [], []
    full_true, full_pred = [], []

    for property_type in SEGMENT_TYPES:
        split = split_and_prepare_segment(df, property_type, seed=seed, test_size=test_size)
        if split is None:
            continue
        train, test_full, test_typical = split["train"], split["test_full"], split["test_typical"]
        if len(train) < MIN_SEGMENT_ROWS // 2:
            continue

        X_train = train[FEATURE_COLUMNS].copy()
        y_train = train["preco"].astype(float).to_numpy()

        rows, folds_used = evaluate_candidates_cv(X_train, y_train, seed=seed, cv_folds=cv_folds)
        selected_rows = rows[:TOP_MODELS_PER_SEGMENT]

        # (4) ensemble treinado apenas com o treino -> usado só para medir desempenho
        eval_entries = fit_weighted_ensemble(X_train, y_train, selected_rows, seed=seed)

        # (5) métricas no teste, que o ensemble de avaliação nunca viu
        metrics_typical, yt, pt = score_on(eval_entries, test_typical)
        metrics_full, yf, pf = score_on(eval_entries, test_full)
        typ_true.extend(yt.tolist())
        typ_pred.extend(pt.tolist())
        full_true.extend(yf.tolist())
        full_pred.extend(pf.tolist())

        individual_test = []
        for entry in eval_entries:
            if test_typical.empty:
                individual_test.append(empty_metrics())
                continue
            preds = entry["estimator"].predict(test_typical[FEATURE_COLUMNS].copy())
            individual_test.append(calculate_metrics(test_typical["preco"].astype(float).to_numpy(), preds))

        # (6) modelo final: pré-processamento e treino com a base inteira do segmento
        final_subset, rare_bairros, clip_bounds = prepare_segment_frame(df, property_type)
        X_final = final_subset[FEATURE_COLUMNS].copy()
        y_final = final_subset["preco"].astype(float).to_numpy()
        final_entries = fit_weighted_ensemble(X_final, y_final, selected_rows, seed=seed)

        segments[property_type] = {
            "rows": int(len(final_subset)),
            "train_rows": int(len(train)),
            "test_rows": int(len(test_typical)),
            "test_rows_full": int(len(test_full)),
            "cv_folds": int(folds_used),
            "rare_bairros": sorted(rare_bairros),
            "clip_bounds": {k: list(v) for k, v in clip_bounds.items()},
            "leaderboard": rows,
            "selected": [
                {
                    "model_name": item["model_name"],
                    "target": item["target"],
                    "weight": item["weight"],
                    "mae_validacao": item["mae_validacao"],
                    "mae_teste": individual_test[i]["mae"],
                    "r2_teste": individual_test[i]["r2"],
                }
                for i, item in enumerate(final_entries)
            ],
            "ensemble": final_entries,
            "metrics": metrics_typical,
            "metrics_full": metrics_full,
        }

    if not segments:
        raise RuntimeError("Dados insuficientes para treinar o modelo híbrido.")

    overall = calculate_metrics(typ_true, typ_pred) if typ_true else empty_metrics()
    overall_full = calculate_metrics(full_true, full_pred) if full_true else empty_metrics()
    return {
        "kind": "hybrid_segmented_weighted_ensemble_v2",
        "strategy": {
            "segments": list(SEGMENT_TYPES),
            "rare_bairro_min_count": RARE_BAIRRO_MIN_COUNT,
            "clip_quantiles": list(CLIP_QUANTILES),
            "outlier_method": "IQR por segmento (limites aprendidos só no treino)",
            "target_modes_tested": ["preco", "log(preco)"],
            "top_models_per_segment": TOP_MODELS_PER_SEGMENT,
            "ensemble": "média ponderada por 1/MAE de validação cruzada (no treino)",
            "cv_folds": int(cv_folds),
            "evaluation": "split antes do pré-processamento; ensemble de avaliação treinado só no treino",
        },
        "segments": segments,
        "metrics": overall,
        "metrics_full": overall_full,
        "seed": seed,
        "test_size": test_size,
        "trained_at": datetime.now().isoformat(),
    }


def build_report_markdown(artifact):
    m = artifact["metrics"]
    mf = artifact.get("metrics_full") or m
    lines = [
        "# Relatório do algoritmo híbrido",
        "",
        "Este algoritmo combina os aprendizados do baseline:",
        "- segmentação por **Casa** e **Apartamento**;",
        "- agrupamento de bairros raros;",
        "- clipagem de variáveis numéricas;",
        "- remoção de outliers por IQR (somente no treino);",
        "- teste de alvo em `preco` e `log(preco)`.",
        "",
        "Depois disso, ele junta os melhores modelos em um ensemble ponderado por MAE.",
        "",
        "## Como as métricas foram medidas",
        "",
        "- O conjunto de teste é separado **antes** de qualquer pré-processamento e nunca é usado",
        "  para escolher modelos, pesos, bairros raros, limites de clipagem ou limites de outlier.",
        "- Os candidatos são ranqueados por **validação cruzada dentro do treino**.",
        "- O ensemble usado para medir desempenho é treinado **somente com o treino**.",
        "  O modelo final (salvo para previsão) é retreinado depois com todos os dados.",
        "- **Teste típico**: imóveis do teste dentro da faixa normal de preço (sem outliers).",
        "- **Teste completo**: todos os imóveis do teste, inclusive anúncios com preço atípico.",
        "",
        "## Resultado geral (ensemble híbrido)",
        "",
        "| Conjunto de teste | MAE | RMSE | R² | MAPE |",
        "|---|---:|---:|---:|---:|",
        f"| Típico (sem outliers) | {format_currency(m['mae'])} | {format_currency(m['rmse'])} | {m['r2']:.4f} | {m['mape']:.2f}% |",
        f"| Completo (com outliers) | {format_currency(mf['mae'])} | {format_currency(mf['rmse'])} | {mf['r2']:.4f} | {mf['mape']:.2f}% |",
        "",
    ]
    for segment_name, segment in artifact["segments"].items():
        sm = segment["metrics"]
        sf = segment.get("metrics_full") or sm
        lines.extend(
            [
                f"## Segmento: {segment_name}",
                "",
                f"- Registros no modelo final: {segment['rows']}",
                f"- Treino (após limpeza): {segment.get('train_rows', '-')} | "
                f"Teste típico: {segment.get('test_rows', '-')} | "
                f"Teste completo: {segment.get('test_rows_full', '-')}",
                f"- Validação cruzada: {segment.get('cv_folds', '-')} folds",
                "",
                "| Conjunto de teste | MAE | RMSE | R² | MAPE |",
                "|---|---:|---:|---:|---:|",
                f"| Típico (sem outliers) | {format_currency(sm['mae'])} | {format_currency(sm['rmse'])} | {sm['r2']:.4f} | {sm['mape']:.2f}% |",
                f"| Completo (com outliers) | {format_currency(sf['mae'])} | {format_currency(sf['rmse'])} | {sf['r2']:.4f} | {sf['mape']:.2f}% |",
                "",
                "### Modelos escolhidos no ensemble",
                "",
                "| Modelo | Alvo | Peso | MAE validação cruzada | MAE no teste típico | R² no teste típico |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in segment["selected"]:
            mae_teste = row.get("mae_teste")
            r2_teste = row.get("r2_teste")
            lines.append(
                f"| {row['model_name']} | {row['target']} | {row['weight']:.3f} | "
                f"{format_currency(row['mae_validacao'])} | "
                f"{format_currency(mae_teste) if mae_teste is not None else '-'} | "
                f"{f'{r2_teste:.4f}' if r2_teste is not None else '-'} |"
            )
        lines.extend(
            [
                "",
                "### Top 10 do ranking (validação cruzada no treino)",
                "",
                "| Modelo | Alvo | MAE | RMSE | R² | MAPE |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in segment["leaderboard"][:10]:
            lines.append(
                f"| {row['model']} | {row['target']} | {format_currency(row['mae'])} | "
                f"{format_currency(row['rmse'])} | {row['r2']:.4f} | {row['mape']:.2f}% |"
            )
        lines.append("")
    return "\n".join(lines)


def save_artifact(artifact, model_path):
    save_model(artifact, model_path)


def load_artifact(model_path):
    # Modelos treinados rodando este arquivo como script guardam `__main__.safe_expm1`.
    # Quem carrega de outro módulo (projeção, interface) precisa encontrá-la lá.
    main_module = sys.modules.get("__main__")
    if main_module is not None and not hasattr(main_module, "safe_expm1"):
        main_module.safe_expm1 = safe_expm1
    artifact = load_model(model_path)
    if not isinstance(artifact, dict) or artifact.get("kind") not in ("hybrid_segmented_weighted_ensemble_v1", "hybrid_segmented_weighted_ensemble_v2"):
        raise SystemExit("Arquivo de modelo híbrido inválido.")
    return artifact


def predict_price(artifact, area_total, area_privada, bairro, tipo_imovel, quartos, banheiros, vagas):
    tipo = canonical_tipo(tipo_imovel)
    if tipo not in artifact["segments"]:
        raise SystemExit("Tipo fora do modelo híbrido. Use Casa ou Apartamento.")

    segment = artifact["segments"][tipo]
    row = pd.DataFrame(
        [
            {
                "area_total": float(area_total or 0),
                "area_privada": float(area_privada or 0),
                "quartos": int(quartos or 0),
                "banheiros": int(banheiros or 0),
                "vagas": int(vagas or 0),
                "bairro": normalize_text(bairro),
                "tipo_imovel": tipo,
            }
        ]
    )
    row = apply_rare_bairros(row, set(segment["rare_bairros"]))
    clip_bounds = {k: tuple(v) for k, v in segment["clip_bounds"].items()}
    row = apply_numeric_clip_bounds(row, clip_bounds)
    prediction = float(predict_weighted_ensemble(segment["ensemble"], row)[0])
    return prediction


def cmd_benchmark(args):
    df = load_training_frame(Path(args.normalized_db or DEFAULT_NORMALIZED_DB))
    artifact = train_hybrid(df, seed=args.seed, test_size=args.test_size, cv_folds=args.cv_folds)
    report_path = Path(args.report_path or DEFAULT_REPORT_PATH)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report_markdown(artifact), encoding="utf-8")

    print(f"Relatorio salvo em {report_path}")
    mt = artifact["metrics"]
    mf = artifact["metrics_full"]
    print(
        f"Ensemble híbrido (teste típico)   | MAE={mt['mae']:.2f} RMSE={mt['rmse']:.2f} R2={mt['r2']:.4f}"
    )
    print(
        f"Ensemble híbrido (teste completo) | MAE={mf['mae']:.2f} RMSE={mf['rmse']:.2f} R2={mf['r2']:.4f}"
    )
    for segment_name, segment in artifact["segments"].items():
        best = segment["leaderboard"][0]
        print(
            f"[{segment_name}] melhor individual (CV): {best['model']} ({best['target']}) "
            f"MAE={best['mae']:.2f} | ensemble no teste MAE={segment['metrics']['mae']:.2f}"
        )


def cmd_train(args):
    df = load_training_frame(Path(args.normalized_db or DEFAULT_NORMALIZED_DB))
    artifact = train_hybrid(df, seed=args.seed, test_size=args.test_size, cv_folds=args.cv_folds)
    save_artifact(artifact, Path(args.model_path or DEFAULT_MODEL_PATH))
    print(f"Modelo híbrido salvo em {args.model_path or DEFAULT_MODEL_PATH}")
    print(
        f"[teste típico] MAE={artifact['metrics']['mae']:.2f} RMSE={artifact['metrics']['rmse']:.2f} "
        f"R2={artifact['metrics']['r2']:.4f} | "
        f"[teste completo] MAE={artifact['metrics_full']['mae']:.2f} R2={artifact['metrics_full']['r2']:.4f}"
    )


def cmd_predict(args):
    area_total = args.area_total if args.area_total is not None else args.area_m2
    area_privada = args.area_privada if args.area_privada is not None else args.area_m2
    if area_total is None and area_privada is None:
        raise SystemExit("Informe --area-total e/ou --area-privada (ou --area-m2).")

    artifact = load_artifact(Path(args.model_path or DEFAULT_MODEL_PATH))
    prediction = predict_price(
        artifact,
        area_total=area_total,
        area_privada=area_privada,
        bairro=args.bairro,
        tipo_imovel=args.tipo_imovel,
        quartos=args.quartos,
        banheiros=args.banheiros,
        vagas=args.vagas,
    )
    print(
        f"Preço estimado (híbrido): R$ {prediction:,.2f}"
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )
    print(
        f"Modelo treinado com MAE={artifact['metrics']['mae']:.2f}, "
        f"RMSE={artifact['metrics']['rmse']:.2f}, R2={artifact['metrics']['r2']:.4f}"
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Modelo híbrido de previsão de preço de imóveis."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_benchmark = subparsers.add_parser(
        "benchmark-hibrido", help="Executa benchmark e gera relatório híbrido."
    )
    p_benchmark.add_argument("--normalized-db", default=None, help="Banco SQLite normalizado.")
    p_benchmark.add_argument(
        "--report-path",
        default=str(DEFAULT_REPORT_PATH),
        help="Arquivo markdown do relatório.",
    )
    p_benchmark.add_argument("--seed", type=int, default=42, help="Seed do split.")
    p_benchmark.add_argument("--test-size", type=float, default=0.2, help="Proporção de teste.")
    p_benchmark.add_argument("--cv-folds", type=int, default=CV_FOLDS, help="Folds da validação cruzada.")
    p_benchmark.set_defaults(func=cmd_benchmark)

    p_train = subparsers.add_parser("train-hibrido", help="Treina o modelo híbrido.")
    p_train.add_argument("--normalized-db", default=None, help="Banco SQLite normalizado.")
    p_train.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Arquivo do modelo.")
    p_train.add_argument("--seed", type=int, default=42, help="Seed do split.")
    p_train.add_argument("--test-size", type=float, default=0.2, help="Proporção de teste.")
    p_train.add_argument("--cv-folds", type=int, default=CV_FOLDS, help="Folds da validação cruzada.")
    p_train.set_defaults(func=cmd_train)

    p_predict = subparsers.add_parser(
        "predict-hibrido", help="Executa previsão com o modelo híbrido."
    )
    p_predict.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Arquivo do modelo.")
    p_predict.add_argument("--area-total", type=float, default=None, help="Área total em m².")
    p_predict.add_argument("--area-privada", type=float, default=None, help="Área privada em m².")
    p_predict.add_argument("--area-m2", type=float, default=None, help="Alias para área total/privada.")
    p_predict.add_argument("--bairro", required=True, help="Bairro.")
    p_predict.add_argument("--tipo-imovel", required=True, help="Tipo de imóvel.")
    p_predict.add_argument("--quartos", type=int, default=0, help="Quantidade de quartos.")
    p_predict.add_argument("--banheiros", type=int, default=0, help="Quantidade de banheiros.")
    p_predict.add_argument("--vagas", type=int, default=0, help="Quantidade de vagas.")
    p_predict.set_defaults(func=cmd_predict)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    # Executa pelo módulo importado para que os modelos salvos referenciem
    # `imoveis_ml_hibrido.safe_expm1` (e não `__main__`), podendo ser abertos em outros scripts.
    import imoveis_ml_hibrido

    imoveis_ml_hibrido.main()
