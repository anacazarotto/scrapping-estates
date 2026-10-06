"""IVG-R (Banco Central, SGS série 21340): índice de preços de IMÓVEIS para a projeção.

O IVG-R (Índice de Valores de Garantia de Imóveis Residenciais Financiados) acompanha
o valor de avaliação dos imóveis dados em garantia nos financiamentos bancários. É um
índice de imóveis, ao contrário do IPCA (inflação geral), por isso é a referência
padrão da valorização v2. Limitação: é nacional, sem recorte para Chapecó.

A série vem em nível (base 100 em mar/2001). Aqui ela é convertida em variação mensal
(%), no mesmo formato do IPCA, para reaproveitar os cálculos de `ipca.py`. Cópia local
em `dados/ivgr.json`, usada quando não houver internet.
"""

import json
from datetime import date
from pathlib import Path

import requests
from ipca import ANOS_MEDIA, faixa_anual, summarize

SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.21340/dados"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = PROJECT_ROOT / "dados" / "ivgr.json"
FONTE = "BCB — IVG-R (valores de garantia de imóveis residenciais financiados), série SGS 21340"


def fetch_monthly(anos=ANOS_MEDIA, timeout=30):
    """Variação mensal (%) do IVG-R nos últimos `anos` anos (+1 de folga)."""
    hoje = date.today()
    params = {
        "formato": "json",
        "dataInicial": date(hoje.year - anos - 1, 1, 1).strftime("%d/%m/%Y"),
        "dataFinal": hoje.strftime("%d/%m/%Y"),
    }
    resp = requests.get(SGS_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    niveis = [(item["data"], float(item["valor"])) for item in resp.json()]
    return [
        {"data": data, "valor": 100 * (nivel / anterior - 1)}
        for (_, anterior), (data, nivel) in zip(niveis, niveis[1:])
    ]


def load_ivgr(cache_path=DEFAULT_CACHE, atualizar=True):
    """Resumo do IVG-R (média anual de 10 anos, 12 meses e faixa) do BCB ou do cache."""
    cache_path = Path(cache_path)
    if atualizar:
        try:
            monthly = fetch_monthly()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(monthly, ensure_ascii=False, indent=0), encoding="utf-8")
        except (requests.RequestException, ValueError, KeyError) as exc:
            print(f"[aviso] IVG-R não baixado do BCB ({exc}); usando {cache_path}.")
    if not cache_path.exists():
        return None
    resumo = summarize(json.loads(cache_path.read_text(encoding="utf-8")))
    resumo["fonte"] = FONTE
    resumo["faixa_12m_pct"] = faixa_anual(cache_path)
    return resumo


if __name__ == "__main__":
    r = load_ivgr()
    print(
        f"IVG-R: média {r['media_anual_pct']:.2f}%/ano ({r['inicio']} a {r['fim']}), "
        f"12 meses {r['acumulado_12m_pct']:.2f}%, faixa {r['faixa_12m_pct']}"
    )
