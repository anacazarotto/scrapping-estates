"""Referência simples para comparar com os modelos v3 (seção de modelos do artigo).

Regra de corretor: preço por m² mediano do bairro no treino x área do imóvel de teste
(bairro sem dados usa a mediana do tipo). Usa a mesma separação treino/teste do v3 e
mede o erro nos mesmos imóveis de teste típicos.

Uso: python scripts_predict/referencia_simples_v3.py
"""

import numpy as np
from imoveis_ml_hibrido import SEGMENT_TYPES, split_and_prepare_segment
from modelo_v3 import DEFAULT_NORMALIZED_DB, avaliar, carregar_reais


def area(df):
    return np.where(df["area_privada"] > 0, df["area_privada"], df["area_total"])


def main():
    df = carregar_reais(DEFAULT_NORMALIZED_DB)
    for tipo in SEGMENT_TYPES:
        split = split_and_prepare_segment(df, tipo, seed=42)
        treino, teste = split["train"], split["test_typical"]
        m2 = treino["preco"] / area(treino)
        pred = teste["bairro"].map(m2.groupby(treino["bairro"]).median()).fillna(m2.median()).to_numpy() * area(teste)
        y = teste["preco"].to_numpy()
        m = avaliar(y, pred)
        dentro = 100 * np.mean(np.abs(pred - y) / y < 0.20)
        print(f"{tipo:<12} n={len(teste)}  MdAPE {m['erro_mediano_pct']:.1f}%  MAE {m['mae']:,.0f}  "
              f"R² {m['r2']:.3f}  a menos de 20%: {dentro:.1f}%")


if __name__ == "__main__":
    main()
