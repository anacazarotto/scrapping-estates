"""Índice IPCA+INCC: referência de valorização da versão 3.

Lógica para o TCC: o preço de um imóvel tem duas partes. O terreno e o bem como um todo
tendem a acompanhar a inflação geral (IPCA); a construção tende a acompanhar o custo de
construir (INCC, da FGV, publicado pelo Banco Central). O índice combinado é a média
simples das duas variações mensais (peso 50% cada), uma escolha declarada, não ajustada
para acertar nenhum caso.

Séries reais do Banco Central (SGS 433 e 192), na cópia local `dados/series_macro.json`
(atualizada por `valorizacao_macro.py`).
"""

import json
import math
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERIES = PROJECT_ROOT / "dados" / "series_macro.json"
ANOS_MEDIA = 10
FONTE = "BCB — média mensal do IPCA (SGS 433) e do INCC (SGS 192)"


def mensal(path=DEFAULT_SERIES):
    """Variação mensal (%) do índice combinado, indexada por mês."""
    dados = json.loads(Path(path).read_text(encoding="utf-8"))

    def serie(nome):
        itens = dados[nome]
        idx = pd.to_datetime([i["data"] for i in itens], dayfirst=True).to_period("M")
        return pd.Series([i["valor"] for i in itens], index=idx)

    return ((serie("ipca") + serie("incc")) / 2).dropna()


def nivel(path=DEFAULT_SERIES):
    """Índice acumulado (base 1 no primeiro mês)."""
    m = mensal(path)
    return (1 + m / 100).cumprod()


def fator(de, ate, indice=None):
    """Quanto o índice subiu entre dois meses (ex.: fator('2021-06', '2026-08') = 1,33).

    Meses fora da série usam o mês disponível mais próximo.
    """
    indice = nivel() if indice is None else indice

    def valor(mes):
        mes = pd.Period(mes, freq="M")
        mes = min(max(mes, indice.index.min()), indice.index.max())
        return float(indice.loc[mes])

    return valor(ate) / valor(de)


def resumo(path=DEFAULT_SERIES, anos=ANOS_MEDIA):
    """Média anual de `anos` anos, acumulado de 12 meses e faixa (p10 a p90) de 12 meses."""
    m = mensal(path)
    ultimos = m.iloc[-12 * anos :]
    media = 100 * (math.prod(1 + v / 100 for v in ultimos) ** (1 / anos) - 1)
    acum12 = 100 * (math.prod(1 + v / 100 for v in m.iloc[-12:]) - 1)
    janelas = (
        (1 + m / 100).rolling(12).apply(lambda x: x.prod(), raw=True).dropna().iloc[-12 * anos :] - 1
    ) * 100
    return {
        "media_anual_pct": media,
        "acumulado_12m_pct": acum12,
        "faixa_12m_pct": (float(janelas.quantile(0.10)), float(janelas.quantile(0.90))),
        "anos_media": anos,
        "inicio": str(ultimos.index.min()),
        "fim": str(ultimos.index.max()),
        "fonte": FONTE,
    }


if __name__ == "__main__":
    r = resumo()
    print(
        f"IPCA+INCC: média {r['media_anual_pct']:.2f}%/ano ({r['inicio']} a {r['fim']}), "
        f"12 meses {r['acumulado_12m_pct']:.2f}%, faixa {r['faixa_12m_pct']}"
    )
