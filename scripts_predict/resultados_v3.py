"""Resultados de TODOS os candidatos treinados no protocolo da versão 3 (para o site).

Mesmo protocolo de `modelo_v3.py`:
- separação treino/teste por imóvel (semente 42, a mesma do v1) ANTES de simular;
- treino = imóveis reais do treino + histórico simulado (IPCA+INCC, 2016 a 2025);
- validação cruzada agrupada por imóvel (GroupKFold), erro medido só nas linhas reais;
- teste só com preços reais, que nenhum modelo viu.

Grava `dados/resultados_modelos_v3.json` no mesmo formato de `resultados_modelos.py`,
lido pelo modo avançado do site. O "Ensemble" é o mesmo do modelo publicado: os 3
melhores entre os candidatos que cabem no site (CatBoost, XGBoost, Gradient Boosting e
Ridge). O TabPFN fica de fora: aceita no máximo 10 mil linhas de treino e, com o
histórico simulado, levaria horas em CPU.

    DISABLE_TABPFN=1 python scripts_predict/resultados_v3.py
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from imoveis_ml_hibrido import SEGMENT_TYPES, make_model_factories, split_and_prepare_segment
from modelo_v3 import (
    CANDIDATOS,
    DEFAULT_NORMALIZED_DB,
    FEATURES_V3,
    avaliar,
    carregar_reais,
    montar,
    prever,
    simular,
    treinar_ensemble,
)
from sklearn.model_selection import GroupKFold, cross_val_predict
from tabpfn_model import TABPFN_VARIANTS

warnings.filterwarnings("ignore")

DEFAULT_OUTPUT = Path("dados/resultados_modelos_v3.json")


def avaliar_completo(y, pred):
    m = avaliar(y, pred)
    pct = np.abs(pred - y) / y * 100
    m["ate_10_pct"] = float(np.mean(pct <= 10) * 100)
    m["ate_20_pct"] = float(np.mean(pct <= 20) * 100)
    return m


def segmento(df, tipo, seed=42):
    split = split_and_prepare_segment(df, tipo, seed=seed)
    treino_real = split["train"]
    treino = pd.concat([treino_real, simular(treino_real)], ignore_index=True)
    grupos = treino["imovel"].to_numpy()
    real = (treino["simulado"] == 0).to_numpy()
    y = treino["preco"].to_numpy()
    teste_tip, teste_comp = split["test_typical"], split["test_full"]

    nomes = [n for n in make_model_factories(seed=seed) if n not in TABPFN_VARIANTS]
    linhas = []
    for nome in nomes:
        for usa_log in (False, True):
            alvo = "log(preco)" if usa_log else "preco"
            print(f"  [{tipo}] {nome} ({alvo})", flush=True)
            try:
                pred_cv = cross_val_predict(
                    montar(nome, usa_log, seed), treino[FEATURES_V3], y, cv=GroupKFold(n_splits=5), groups=grupos
                )
                modelo = montar(nome, usa_log, seed).fit(treino[FEATURES_V3], y)
                pred_tip = np.asarray(modelo.predict(teste_tip[FEATURES_V3]), dtype=float)
                pred_comp = np.asarray(modelo.predict(teste_comp[FEATURES_V3]), dtype=float)
            except (RuntimeError, ValueError) as exc:
                print(f"[aviso] {nome} ({alvo}) ignorado: {exc}")
                continue
            erro = pred_cv[real] - y[real]
            linhas.append(
                {
                    "modelo": nome,
                    "alvo": alvo,
                    "no_site": nome in CANDIDATOS,
                    "cv": {
                        "mae": float(np.mean(np.abs(erro))),
                        "rmse": float(np.sqrt(np.mean(erro**2))),
                        "r2": float(1 - np.sum(erro**2) / np.sum((y[real] - y[real].mean()) ** 2)),
                    },
                    "teste_tipico": avaliar_completo(teste_tip["preco"].to_numpy(), pred_tip),
                    "teste_completo": avaliar_completo(teste_comp["preco"].to_numpy(), pred_comp),
                    "previsto": [round(float(v)) for v in pred_tip],
                }
            )
    linhas.sort(key=lambda m: m["cv"]["mae"])
    for pos, m in enumerate(linhas, start=1):
        m["posicao_cv"] = pos

    # Ensemble publicado: 3 melhores (na validação cruzada) entre os que cabem no site.
    ranking_site = [
        {"model": m["modelo"], "target": m["alvo"], "mae": m["cv"]["mae"]} for m in linhas if m["no_site"]
    ]
    entradas = treinar_ensemble(treino, ranking_site, seed)
    ens_tip = prever(entradas, teste_tip)
    ens_comp = prever(entradas, teste_comp)
    linhas.insert(
        0,
        {
            "modelo": "Ensemble",
            "alvo": " + ".join(f"{e['model_name']} ({e['target']}, peso {e['weight']:.2f})" for e in entradas),
            "no_site": True,
            "posicao_cv": 0,
            "cv": None,
            "teste_tipico": avaliar_completo(teste_tip["preco"].to_numpy(), ens_tip),
            "teste_completo": avaliar_completo(teste_comp["preco"].to_numpy(), ens_comp),
            "previsto": [round(float(v)) for v in ens_tip],
        },
    )
    area = teste_tip["area_privada"].where(teste_tip["area_privada"] > 0, teste_tip["area_total"])
    return {
        "linhas_treino": int(len(treino_real)),
        "linhas_simuladas_treino": int((treino["simulado"] == 1).sum()),
        "linhas_teste_tipico": int(len(teste_tip)),
        "linhas_teste_completo": int(len(teste_comp)),
        "folds": 5,
        "real": [round(float(v)) for v in teste_tip["preco"]],
        "bairro": teste_tip["bairro"].tolist(),
        "area": area.round(1).tolist(),
        "modelos": linhas,
    }


def main():
    df = carregar_reais(DEFAULT_NORMALIZED_DB)
    resultados = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "versao": "v3",
        "protocolo": (
            "separação treino/teste por imóvel (semente 42) antes de simular; treino com os "
            "anúncios reais + histórico simulado pelo índice IPCA+INCC (2016 a 2025); validação "
            "cruzada de 5 partes agrupada por imóvel; erro medido só nos preços reais"
        ),
        "segmentos": {tipo: segmento(df, tipo) for tipo in SEGMENT_TYPES},
    }
    DEFAULT_OUTPUT.write_text(json.dumps(resultados, ensure_ascii=False), encoding="utf-8")
    for tipo, s in resultados["segmentos"].items():
        print(f"== {tipo}")
        for m in s["modelos"]:
            t = m["teste_tipico"]
            print(f"   {m['posicao_cv']:>2} {m['modelo'][:34]:<34} {m['alvo'][:10]:<10} MAE teste {t['mae']:>10,.0f}  R2 {t['r2']:.3f}")
    print(f"Resultados salvos em {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    import resultados_v3

    resultados_v3.main()
