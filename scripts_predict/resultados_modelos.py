"""Resultados de TODOS os candidatos do modelo de preço, para o modo avançado do site.

Usa exatamente o protocolo do modelo híbrido (`imoveis_ml_hibrido.py`): a mesma
separação treino/teste (semente 42), a mesma limpeza aprendida só no treino e a mesma
validação cruzada de 5 partes. Para cada candidato (algoritmo x alvo) grava:

- métricas na validação cruzada (treino) e no teste típico e completo;
- as previsões no teste típico (preço real x previsto), para os gráficos.

Também exporta a base usada no treino (casas e apartamentos, sem código do anúncio,
endereço, foto ou imobiliária) para a aba "Dataset".

    DISABLE_TABPFN=1 python scripts_predict/resultados_modelos.py
    TABPFN_VERSIONS=v2,v3.5 python scripts_predict/resultados_modelos.py   # com TabPFN
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from imoveis_ml_hibrido import (
    CV_FOLDS,
    FEATURE_COLUMNS,
    SEGMENT_TYPES,
    TOP_MODELS_PER_SEGMENT,
    build_regressor,
    calculate_metrics,
    evaluate_candidates_cv,
    fit_weighted_ensemble,
    load_training_frame,
    predict_weighted_ensemble,
    split_and_prepare_segment,
)
from tabpfn_model import enabled_tabpfn_variants

DEFAULT_NORMALIZED_DB = Path("imoveis_normalizados.db")
DEFAULT_RESULTS_PATH = Path("dados/resultados_modelos.json")
DEFAULT_DATASET_PATH = Path("dados/dataset_imoveis.csv")


def erro_percentual(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    pct = np.abs(np.asarray(y_pred, dtype=float) - y_true) / y_true * 100
    return {
        "erro_mediano_pct": float(np.median(pct)),
        "ate_10_pct": float(np.mean(pct <= 10) * 100),
        "ate_20_pct": float(np.mean(pct <= 20) * 100),
    }


def avaliar(y_true, y_pred):
    m = calculate_metrics(y_true, y_pred)
    m.pop("mape", None)  # inútil com os erros de cadastro (ver EDA)
    return {**m, **erro_percentual(y_true, y_pred)}


def resultados_segmento(df, tipo, seed=42):
    split = split_and_prepare_segment(df, tipo, seed=seed)
    train, test_typ, test_full = split["train"], split["test_typical"], split["test_full"]
    X_tr, y_tr = train[FEATURE_COLUMNS], train["preco"].to_numpy(dtype=float)
    X_typ, y_typ = test_typ[FEATURE_COLUMNS], test_typ["preco"].to_numpy(dtype=float)
    X_full, y_full = test_full[FEATURE_COLUMNS], test_full["preco"].to_numpy(dtype=float)

    ranking, folds = evaluate_candidates_cv(X_tr, y_tr, seed=seed, cv_folds=CV_FOLDS)
    contexto = {
        "bairro": test_typ["bairro"].tolist(),
        "area": test_typ["area_privada"].where(test_typ["area_privada"] > 0, test_typ["area_total"]).round(1).tolist(),
    }

    modelos = []
    for pos, row in enumerate(ranking, start=1):
        print(f"[{tipo}] {pos}/{len(ranking)} {row['model']} ({row['target']})", flush=True)
        model = build_regressor(row["model"], seed=seed, use_log=row["target"] == "log(preco)")
        try:
            model.fit(X_tr, y_tr)
            pred_typ = np.asarray(model.predict(X_typ), dtype=float)
            pred_full = np.asarray(model.predict(X_full), dtype=float)
        except (RuntimeError, ValueError) as exc:
            print(f"[aviso] {row['model']} ignorado: {exc}")
            continue
        cv = {k: float(row[k]) for k in ("mae", "rmse", "r2")}
        modelos.append(
            {
                "modelo": row["model"],
                "alvo": row["target"],
                "posicao_cv": pos,
                "cv": cv,
                "teste_tipico": avaliar(y_typ, pred_typ),
                "teste_completo": avaliar(y_full, pred_full),
                "previsto": [round(float(v)) for v in pred_typ],
            }
        )

    selected = ranking[:TOP_MODELS_PER_SEGMENT]
    entries = fit_weighted_ensemble(X_tr, y_tr, selected, seed=seed)
    ens_typ = np.asarray(predict_weighted_ensemble(entries, X_typ), dtype=float)
    ens_full = np.asarray(predict_weighted_ensemble(entries, X_full), dtype=float)
    modelos.insert(
        0,
        {
            "modelo": "Ensemble",
            "alvo": " + ".join(f"{e['model_name']} ({e['target']}, peso {e['weight']:.2f})" for e in entries),
            "posicao_cv": 0,
            "cv": None,
            "teste_tipico": avaliar(y_typ, ens_typ),
            "teste_completo": avaliar(y_full, ens_full),
            "previsto": [round(float(v)) for v in ens_typ],
        },
    )
    return {
        "linhas_treino": int(len(train)),
        "linhas_teste_tipico": int(len(test_typ)),
        "linhas_teste_completo": int(len(test_full)),
        "folds": int(folds),
        "real": [round(float(v)) for v in y_typ],
        **contexto,
        "modelos": modelos,
    }


def exportar_dataset(df, path):
    base = df[df["tipo_imovel"].isin(SEGMENT_TYPES)].copy()
    area = base["area_privada"].where(base["area_privada"] > 0, base["area_total"])
    base["preco_m2"] = (base["preco"] / area.where(area > 0)).round(2)
    colunas = ["tipo_imovel", "bairro", "area_total", "area_privada", "quartos",
               "banheiros", "vagas", "preco", "preco_m2"]
    base = base[colunas].sort_values(["tipo_imovel", "bairro", "preco"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(path, index=False, encoding="utf-8")
    return len(base)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--normalized-db", default=str(DEFAULT_NORMALIZED_DB))
    parser.add_argument("--results-path", default=str(DEFAULT_RESULTS_PATH))
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_training_frame(Path(args.normalized_db))
    enabled_tabpfn_variants(verbose=True)  # avisa quais versões do TabPFN entram
    resultados = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "protocolo": "split 80/20 (semente 42) antes da limpeza; regras aprendidas só no "
        "treino; ranking por validação cruzada de 5 partes; teste usado uma vez",
        "segmentos": {tipo: resultados_segmento(df, tipo, seed=args.seed) for tipo in SEGMENT_TYPES},
    }
    out = Path(args.results_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(resultados, ensure_ascii=False), encoding="utf-8")
    n = exportar_dataset(df, Path(args.dataset_path))
    print(f"Resultados salvos em {out}; dataset com {n} imóveis em {args.dataset_path}")


if __name__ == "__main__":
    main()
