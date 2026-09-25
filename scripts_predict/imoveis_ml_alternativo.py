import argparse
import sqlite3
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from model_io import load_model, save_model
from tabpfn_model import TABPFN_NAME, make_tabpfn, tabpfn_enabled
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

DEFAULT_NORMALIZED_DB = Path("imoveis_normalizados.db")
DEFAULT_MODEL_PATH = Path("modelos/preco_imovel_modelo_alternativo.pkl")
DEFAULT_REPORT_PATH = Path("docs/modelo_alternativo.md")

CV_FOLDS = 5
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


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    denom = np.where(y_true == 0, np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def make_model_factories(seed=42):
    factories = {
        "Regressão Linear Múltipla": lambda: LinearRegression(),
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
            solver="adam",
            alpha=1e-4,
            batch_size="auto",
            learning_rate_init=1e-3,
            max_iter=500,
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
    if tabpfn_enabled():
        factories[TABPFN_NAME] = lambda: make_tabpfn(NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, seed=seed)
    return factories


def build_regressor(name, seed=42):
    factories = make_model_factories(seed=seed)
    if name not in factories:
        raise ValueError(f"Modelo não suportado: {name}")
    model = factories[name]()
    if name == TABPFN_NAME:
        # TabPFN faz a própria codificação (sem one-hot/padronização).
        return Pipeline([("model", model)])
    return Pipeline([("preprocess", make_preprocessor()), ("model", model)])


def format_currency(value):
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def benchmark_models(df, seed=42, test_size=0.2):
    X = df[FEATURE_COLUMNS].copy()
    y = df["preco"].astype(float).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    results = []
    fitted = {}
    folds = int(max(2, min(CV_FOLDS, len(X_train) // 10)))
    cv = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for name in make_model_factories(seed=seed):
        # MAE de validação cruzada NO TREINO: é o critério de escolha do modelo.
        cv_preds = cross_val_predict(build_regressor(name, seed=seed), X_train, y_train, cv=cv)
        cv_mae = float(np.mean(np.abs(y_train - cv_preds)))

        model = build_regressor(name, seed=seed)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = calculate_metrics(y_test, preds)
        row = {"model": name, "cv_mae": cv_mae, **metrics}
        results.append(row)
        fitted[name] = model

    # Ordena pela validação cruzada (não pelo teste), para o teste continuar sendo
    # uma estimativa honesta do desempenho do modelo escolhido.
    results.sort(key=lambda item: (item["cv_mae"], item["mae"]))
    split_info = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "seed": seed,
        "test_size": test_size,
        "cv_folds": folds,
    }
    return results, fitted, split_info


def build_benchmark_markdown(results, split_info):
    lines = [
        "# Relatório alternativo de modelos",
        "",
        "Modelos testados:",
        *[f"- {row['model']}" for row in results],
        "",
        "O ranking usa o MAE de **validação cruzada no treino**; as demais colunas são",
        "medidas no conjunto de teste, que não participa da escolha do modelo.",
        "",
        f"- treino: {split_info['train_rows']}",
        f"- teste: {split_info['test_rows']}",
        f"- seed: {split_info['seed']}",
        "",
        "## Métricas",
        "",
        "- **MAE**: erro médio absoluto (menor é melhor).",
        "- **RMSE**: penaliza mais erros grandes (menor é melhor).",
        "- **R²**: capacidade explicativa (maior é melhor).",
        "- **MAPE**: erro percentual médio (menor é melhor).",
        "",
        "## Resultado",
        "",
        "| Modelo | MAE (CV treino) | MAE | RMSE | R² | MAPE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['model']} | {format_currency(row.get('cv_mae', 0.0))} | "
            f"{format_currency(row['mae'])} | {format_currency(row['rmse'])} | "
            f"{row['r2']:.4f} | {row['mape']:.2f}% |"
        )

    best = results[0]["model"] if results else "N/A"
    lines.extend(
        [
            "",
            "## Melhor candidato",
            "",
            f"- Melhor por MAE de validação cruzada: **{best}**",
            "- O comando `train-alt` salva esse melhor modelo como padrão alternativo.",
            "",
        ]
    )
    return "\n".join(lines)


def save_artifact(artifact, model_path):
    save_model(artifact, model_path)


def load_artifact(model_path):
    return load_model(model_path)


def fit_best_model(df, seed=42, test_size=0.2):
    results, _, split_info = benchmark_models(df, seed=seed, test_size=test_size)
    if not results:
        raise RuntimeError("Nenhum modelo treinado.")

    best_name = results[0]["model"]
    final_model = build_regressor(best_name, seed=seed)
    X = df[FEATURE_COLUMNS].copy()
    y = df["preco"].astype(float).to_numpy()
    final_model.fit(X, y)

    artifact = {
        "kind": "algoritmo_alternativo_multimodelo",
        "selected_model": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "model": final_model,
        "benchmark": results,
        "metrics": {
            **results[0],
            "train_rows": split_info["train_rows"],
            "test_rows": split_info["test_rows"],
            "rows": int(len(df)),
        },
        "trained_at": datetime.now().isoformat(),
        "seed": seed,
        "test_size": test_size,
    }
    return artifact


def predict_price(
    model_path, area_total, area_privada, bairro, tipo_imovel, quartos, banheiros, vagas
):
    artifact = load_artifact(model_path)
    if not isinstance(artifact, dict) or artifact.get("kind") != "algoritmo_alternativo_multimodelo":
        raise SystemExit("Arquivo de modelo alternativo inválido.")

    row = pd.DataFrame(
        [
            {
                "area_total": float(area_total or 0),
                "area_privada": float(area_privada or 0),
                "quartos": int(quartos or 0),
                "banheiros": int(banheiros or 0),
                "vagas": int(vagas or 0),
                "bairro": normalize_text(bairro),
                "tipo_imovel": canonical_tipo(tipo_imovel),
            }
        ]
    )
    prediction = float(artifact["model"].predict(row)[0])
    return prediction, artifact.get("metrics", {}), artifact.get("selected_model", "")


def cmd_benchmark(args):
    normalized_db = Path(args.normalized_db or DEFAULT_NORMALIZED_DB)
    report_path = Path(args.report_path or DEFAULT_REPORT_PATH)
    df = load_training_frame(normalized_db)
    results, _, split_info = benchmark_models(df, seed=args.seed, test_size=args.test_size)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_benchmark_markdown(results, split_info), encoding="utf-8")

    print(f"Relatorio salvo em {report_path}")
    print("| Modelo | MAE | RMSE | R² | MAPE |")
    print("|---|---:|---:|---:|---:|")
    for row in results:
        print(
            f"| {row['model']} | {format_currency(row['mae'])} | {format_currency(row['rmse'])} | "
            f"{row['r2']:.4f} | {row['mape']:.2f}% |"
        )


def cmd_train(args):
    normalized_db = Path(args.normalized_db or DEFAULT_NORMALIZED_DB)
    model_path = Path(args.model_path or DEFAULT_MODEL_PATH)
    df = load_training_frame(normalized_db)
    artifact = fit_best_model(df, seed=args.seed, test_size=args.test_size)
    save_artifact(artifact, model_path)

    print(f"Modelo alternativo salvo em {model_path}")
    print(f"Modelo escolhido: {artifact['selected_model']}")
    metrics = artifact["metrics"]
    print(
        f"MAE={metrics['mae']:.2f} RMSE={metrics['rmse']:.2f} R2={metrics['r2']:.3f} "
        f"treino={metrics['train_rows']} teste={metrics['test_rows']} base={metrics['rows']}"
    )


def cmd_predict(args):
    area_total = args.area_total if args.area_total is not None else args.area_m2
    area_privada = args.area_privada if args.area_privada is not None else args.area_m2
    if area_total is None and area_privada is None:
        raise SystemExit("Informe --area-total e/ou --area-privada (ou --area-m2).")

    prediction, metrics, selected_model = predict_price(
        args.model_path,
        area_total=area_total,
        area_privada=area_privada,
        bairro=args.bairro,
        tipo_imovel=args.tipo_imovel,
        quartos=args.quartos,
        banheiros=args.banheiros,
        vagas=args.vagas,
    )
    print(
        f"Preço estimado (alternativo): R$ {prediction:,.2f}"
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )
    if metrics:
        print(
            f"Modelo: {selected_model} | MAE={metrics.get('mae', 0):.2f} "
            f"RMSE={metrics.get('rmse', 0):.2f} R2={metrics.get('r2', 0):.3f}"
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Pipeline alternativo de previsão de preço de imóveis."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_benchmark = subparsers.add_parser(
        "benchmark-alt", help="Compara modelos do algoritmo alternativo."
    )
    p_benchmark.add_argument("--normalized-db", default=None, help="Banco SQLite normalizado.")
    p_benchmark.add_argument(
        "--report-path",
        default=str(DEFAULT_REPORT_PATH),
        help="Arquivo markdown do relatório alternativo.",
    )
    p_benchmark.add_argument("--seed", type=int, default=42, help="Seed do split.")
    p_benchmark.add_argument("--test-size", type=float, default=0.2, help="Proporção de teste.")
    p_benchmark.set_defaults(func=cmd_benchmark)

    p_train = subparsers.add_parser(
        "train-alt", help="Treina e salva o melhor modelo alternativo."
    )
    p_train.add_argument("--normalized-db", default=None, help="Banco SQLite normalizado.")
    p_train.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Arquivo do modelo.")
    p_train.add_argument("--seed", type=int, default=42, help="Seed do split.")
    p_train.add_argument("--test-size", type=float, default=0.2, help="Proporção de teste.")
    p_train.set_defaults(func=cmd_train)

    p_predict = subparsers.add_parser("predict-alt", help="Executa previsão com o modelo alternativo.")
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
