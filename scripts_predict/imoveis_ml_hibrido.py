import argparse
import json
import sqlite3
import unicodedata
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - optional dependency
    XGBRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover - optional dependency
    CatBoostRegressor = None

DEFAULT_NORMALIZED_DB = Path("imoveis_normalizados.db")
DEFAULT_MODEL_PATH = Path("modelos/preco_imovel_modelo_hibrido.pkl")
DEFAULT_REPORT_PATH = Path("docs/modelo_hibrido.md")
SEGMENT_TYPES = ("Casa", "Apartamento")
RARE_BAIRRO_MIN_COUNT = 10
CLIP_QUANTILES = (0.02, 0.98)
TOP_MODELS_PER_SEGMENT = 3

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
    return factories


def build_regressor(name, *, seed=42, use_log=True):
    factories = make_model_factories(seed=seed)
    if name not in factories:
        raise ValueError(f"Modelo não suportado: {name}")
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
        inverse_func=np.expm1,
        check_inverse=False,
    )


def prepare_segment_frame(df, property_type):
    subset = df[df["tipo_imovel"].map(canonical_tipo) == property_type].copy()
    subset = subset[subset["preco"] > 0].copy()
    if subset.empty:
        return subset, set(), {}

    rare_bairros = compute_rare_bairros(subset, min_count=RARE_BAIRRO_MIN_COUNT)
    clip_bounds = compute_numeric_clip_bounds(subset, columns=NUMERIC_COLUMNS, quantiles=CLIP_QUANTILES)
    subset = apply_rare_bairros(subset, rare_bairros)
    subset = apply_numeric_clip_bounds(subset, clip_bounds)
    area_ref = subset["area_privada"].where(subset["area_privada"] > 0, subset["area_total"])
    area_ref = area_ref.where(area_ref > 0)
    subset["preco_m2"] = subset["preco"] / area_ref
    subset = remove_outliers_iqr(subset, columns=("preco", "preco_m2"), factor=1.5)
    subset = subset[subset["preco"] > 0].copy()
    return subset, rare_bairros, clip_bounds


def evaluate_candidates(subset, seed=42, test_size=0.2):
    X = subset[FEATURE_COLUMNS].copy()
    y = subset["preco"].astype(float).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    rows = []
    factories = make_model_factories(seed=seed)
    for use_log in (False, True):
        target_label = "log(preco)" if use_log else "preco"
        for name in factories:
            model = build_regressor(name, seed=seed, use_log=use_log)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics = calculate_metrics(y_test, preds)
            rows.append(
                {
                    "model": name,
                    "target": target_label,
                    **metrics,
                }
            )
    rows.sort(key=lambda item: (item["mae"], item["rmse"]))
    return rows, X, y, y_test, X_test


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


def train_hybrid(df, seed=42, test_size=0.2):
    segments = {}
    all_true = []
    all_pred = []

    for property_type in SEGMENT_TYPES:
        subset, rare_bairros, clip_bounds = prepare_segment_frame(df, property_type)
        if len(subset) < 20:
            continue

        rows, X, y, y_test, X_test = evaluate_candidates(subset, seed=seed, test_size=test_size)
        selected_rows = rows[:TOP_MODELS_PER_SEGMENT]
        ensemble_entries = fit_weighted_ensemble(X, y, selected_rows, seed=seed)
        segment_pred = predict_weighted_ensemble(ensemble_entries, X_test)
        segment_metrics = calculate_metrics(y_test, segment_pred)

        all_true.extend(y_test.tolist())
        all_pred.extend(segment_pred.tolist())
        segments[property_type] = {
            "rows": int(len(subset)),
            "rare_bairros": sorted(rare_bairros),
            "clip_bounds": {k: list(v) for k, v in clip_bounds.items()},
            "leaderboard": rows,
            "selected": [
                {
                    "model_name": item["model_name"],
                    "target": item["target"],
                    "weight": item["weight"],
                    "mae_validacao": item["mae_validacao"],
                }
                for item in ensemble_entries
            ],
            "ensemble": ensemble_entries,
            "metrics": segment_metrics,
        }

    if not segments:
        raise RuntimeError("Dados insuficientes para treinar o modelo híbrido.")

    overall = calculate_metrics(all_true, all_pred) if all_true else {
        "mae": 0.0,
        "rmse": 0.0,
        "r2": 0.0,
        "mape": 0.0,
    }
    return {
        "kind": "hybrid_segmented_weighted_ensemble_v1",
        "strategy": {
            "segments": list(SEGMENT_TYPES),
            "rare_bairro_min_count": RARE_BAIRRO_MIN_COUNT,
            "clip_quantiles": list(CLIP_QUANTILES),
            "outlier_method": "IQR por segmento",
            "target_modes_tested": ["preco", "log(preco)"],
            "top_models_per_segment": TOP_MODELS_PER_SEGMENT,
            "ensemble": "média ponderada por 1/MAE de validação",
        },
        "segments": segments,
        "metrics": overall,
        "seed": seed,
        "test_size": test_size,
        "trained_at": datetime.now().isoformat(),
    }


def build_report_markdown(artifact):
    lines = [
        "# Relatório do algoritmo híbrido",
        "",
        "Este algoritmo combina os aprendizados do baseline:",
        "- segmentação por **Casa** e **Apartamento**;",
        "- agrupamento de bairros raros;",
        "- clipagem de variáveis numéricas;",
        "- remoção de outliers por IQR;",
        "- teste de alvo em `preco` e `log(preco)`.",
        "",
        "Depois disso, ele junta os melhores modelos em um ensemble ponderado por MAE.",
        "",
        "## Resultado geral (ensemble híbrido)",
        "",
        f"- MAE: {format_currency(artifact['metrics']['mae'])}",
        f"- RMSE: {format_currency(artifact['metrics']['rmse'])}",
        f"- R²: {artifact['metrics']['r2']:.4f}",
        f"- MAPE: {artifact['metrics']['mape']:.2f}%",
        "",
    ]
    for segment_name, segment in artifact["segments"].items():
        lines.extend(
            [
                f"## Segmento: {segment_name}",
                "",
                f"- Registros usados: {segment['rows']}",
                f"- MAE do ensemble: {format_currency(segment['metrics']['mae'])}",
                f"- RMSE do ensemble: {format_currency(segment['metrics']['rmse'])}",
                f"- R² do ensemble: {segment['metrics']['r2']:.4f}",
                f"- MAPE do ensemble: {segment['metrics']['mape']:.2f}%",
                "",
                "### Modelos escolhidos no ensemble",
                "",
                "| Modelo | Alvo | Peso | MAE validação |",
                "|---|---|---:|---:|",
            ]
        )
        for row in segment["selected"]:
            lines.append(
                f"| {row['model_name']} | {row['target']} | {row['weight']:.3f} | {format_currency(row['mae_validacao'])} |"
            )
        lines.extend(
            [
                "",
                "### Top 10 do benchmark do segmento",
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
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)


def load_artifact(model_path):
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict) or artifact.get("kind") != "hybrid_segmented_weighted_ensemble_v1":
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
    artifact = train_hybrid(df, seed=args.seed, test_size=args.test_size)
    report_path = Path(args.report_path or DEFAULT_REPORT_PATH)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report_markdown(artifact), encoding="utf-8")

    print(f"Relatorio salvo em {report_path}")
    print(
        f"Ensemble híbrido | MAE={artifact['metrics']['mae']:.2f} "
        f"RMSE={artifact['metrics']['rmse']:.2f} R2={artifact['metrics']['r2']:.4f} "
        f"MAPE={artifact['metrics']['mape']:.2f}%"
    )
    for segment_name, segment in artifact["segments"].items():
        best = segment["leaderboard"][0]
        print(
            f"[{segment_name}] melhor individual: {best['model']} ({best['target']}) "
            f"MAE={best['mae']:.2f} | ensemble MAE={segment['metrics']['mae']:.2f}"
        )


def cmd_train(args):
    df = load_training_frame(Path(args.normalized_db or DEFAULT_NORMALIZED_DB))
    artifact = train_hybrid(df, seed=args.seed, test_size=args.test_size)
    save_artifact(artifact, Path(args.model_path or DEFAULT_MODEL_PATH))
    print(f"Modelo híbrido salvo em {args.model_path or DEFAULT_MODEL_PATH}")
    print(
        f"MAE={artifact['metrics']['mae']:.2f} RMSE={artifact['metrics']['rmse']:.2f} "
        f"R2={artifact['metrics']['r2']:.4f} MAPE={artifact['metrics']['mape']:.2f}%"
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
    p_benchmark.set_defaults(func=cmd_benchmark)

    p_train = subparsers.add_parser("train-hibrido", help="Treina o modelo híbrido.")
    p_train.add_argument("--normalized-db", default=None, help="Banco SQLite normalizado.")
    p_train.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Arquivo do modelo.")
    p_train.add_argument("--seed", type=int, default=42, help="Seed do split.")
    p_train.add_argument("--test-size", type=float, default=0.2, help="Proporção de teste.")
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
    main()
