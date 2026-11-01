"""Resultados de TODOS os candidatos treinados no protocolo da versão 3 (para o site).

Mesmo protocolo de `modelo_v3.py`:
- separação treino/teste por imóvel (semente 42, a mesma do v1) ANTES de simular;
- treino = imóveis reais do treino + histórico simulado (IPCA+INCC, 2016 a 2025);
- validação cruzada agrupada por imóvel (GroupKFold), erro medido só nas linhas reais;
- teste só com preços reais, que nenhum modelo viu.

Grava `dados/resultados_modelos_v3.json` no mesmo formato de `resultados_modelos.py`,
lido pelo modo avançado do site. Ranking e escolha como em `modelo_v3.py`: menor MdAPE
na validação cruzada (empate decidido pelo MAE), com o "Ensemble" (os 3 melhores que
cabem no site: CatBoost, XGBoost, Gradient Boosting e Ridge) como mais um candidato. O
TabPFN entra com as previsões de `tabpfn_v3.py` (rodado antes, com corte de tempo).

    DISABLE_TABPFN=1 python scripts_predict/resultados_v3.py
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from imoveis_ml_hibrido import SEGMENT_TYPES, make_model_factories
from modelo_v3 import (
    CANDIDATOS,
    DEFAULT_NORMALIZED_DB,
    ENSEMBLE,
    FEATURES_V3,
    avaliar,
    carregar_reais,
    chave,
    dados_segmento,
    linha_ensemble,
    mdape,
    montar,
    resultados_tabpfn,
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
    split, treino, grupos, real = dados_segmento(df, tipo, seed)
    treino_real = split["train"]
    y = treino["preco"].to_numpy()
    yr = y[real]
    teste_tip, teste_comp = split["test_typical"], split["test_full"]
    y_tip, y_comp = teste_tip["preco"].to_numpy(), teste_comp["preco"].to_numpy()

    def linha(nome, alvo, pred_cv, pred_tip, pred_comp, no_site, **extra):
        erro = pred_cv - yr
        return {
            "modelo": nome,
            "alvo": alvo,
            "no_site": no_site,
            "cv": {
                "mdape": mdape(yr, pred_cv),
                "mae": float(np.mean(np.abs(erro))),
                "rmse": float(np.sqrt(np.mean(erro**2))),
                "r2": float(1 - np.sum(erro**2) / np.sum((yr - yr.mean()) ** 2)),
            },
            "teste_tipico": avaliar_completo(y_tip, pred_tip),
            "teste_completo": avaliar_completo(y_comp, pred_comp),
            "previsto": [round(float(v)) for v in pred_tip],
            **extra,
        }

    nomes = [n for n in make_model_factories(seed=seed) if n not in TABPFN_VARIANTS]
    linhas, previsoes_cv, previsoes_teste = [], {}, {}
    for nome in nomes:
        for usa_log in (False, True):
            alvo = "log(preco)" if usa_log else "preco"
            print(f"  [{tipo}] {nome} ({alvo})", flush=True)
            try:
                pred_cv = cross_val_predict(
                    montar(nome, usa_log, seed), treino[FEATURES_V3], y, cv=GroupKFold(n_splits=5), groups=grupos
                )[real]
                modelo = montar(nome, usa_log, seed).fit(treino[FEATURES_V3], y)
                pred_tip = np.asarray(modelo.predict(teste_tip[FEATURES_V3]), dtype=float)
                pred_comp = np.asarray(modelo.predict(teste_comp[FEATURES_V3]), dtype=float)
            except (RuntimeError, ValueError) as exc:
                print(f"[aviso] {nome} ({alvo}) ignorado: {exc}")
                continue
            previsoes_cv[(nome, alvo)] = pred_cv
            previsoes_teste[(nome, alvo)] = (pred_tip, pred_comp)
            linhas.append(linha(nome, alvo, pred_cv, pred_tip, pred_comp, nome in CANDIDATOS))

    # TabPFN: previsões das tentativas que terminaram dentro do corte (tabpfn_v3.py).
    tentativas, tabpfn = resultados_tabpfn(tipo, treino.loc[real, "imovel"])
    for nome, dados in tabpfn.items():
        linhas.append(
            linha(nome, "preco", np.asarray(dados["cv_previsto"]), np.asarray(dados["teste_tipico"]),
                  np.asarray(dados["teste_completo"]), dados["api"], linhas_treino=dados["linhas_treino"],
                  minutos=dados["minutos"], api=dados["api"])
        )

    # Ensemble: os 3 melhores (por MdAPE) entre os que cabem no site, como em modelo_v3.py.
    # No teste, a média ponderada dos mesmos modelos treinados acima.
    ranking = [{"model": m["modelo"], "target": m["alvo"], "no_site": m["no_site"], "api": m.get("api", False), **m["cv"]}
               for m in linhas]
    ens, pred_cv = linha_ensemble(ranking, previsoes_cv, yr)
    pred_tip = sum(e["weight"] * previsoes_teste[(e["model"], e["target"])][0] for e in ens["membros"])
    pred_comp = sum(e["weight"] * previsoes_teste[(e["model"], e["target"])][1] for e in ens["membros"])
    composicao = " + ".join(f"{e['model']} ({e['target']}, peso {e['weight']:.2f})" for e in ens["membros"])
    linhas.append(linha(ENSEMBLE, composicao, pred_cv, pred_tip, pred_comp, True))

    def chave_linha(m):
        return chave({"mdape": m["cv"]["mdape"], "mae": m["cv"]["mae"]})

    linhas.sort(key=chave_linha)
    escolhido = min((m for m in linhas if m["no_site"]), key=chave_linha)
    for pos, m in enumerate(linhas, start=1):
        m["posicao_cv"] = pos
        m["escolhido"] = m is escolhido
    area = teste_tip["area_privada"].where(teste_tip["area_privada"] > 0, teste_tip["area_total"])
    return {
        "linhas_treino": int(len(treino_real)),
        "linhas_simuladas_treino": int((treino["simulado"] == 1).sum()),
        "linhas_teste_tipico": int(len(teste_tip)),
        "linhas_teste_completo": int(len(teste_comp)),
        "folds": 5,
        "tabpfn_tentativas": tentativas,
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
            print(f"   {m['posicao_cv']:>2}{'*' if m['escolhido'] else ' '} {m['modelo'][:34]:<34} {m['alvo'][:10]:<10} "
                  f"MdAPE CV {m['cv']['mdape']:5.1f}%  MdAPE teste {t['erro_mediano_pct']:5.1f}%  R2 {t['r2']:.3f}")
    print(f"Resultados salvos em {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    import resultados_v3

    resultados_v3.main()
