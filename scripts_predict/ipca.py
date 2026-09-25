"""IPCA oficial (Banco Central, SGS série 433) como taxa de referência da projeção.

O IPCA é usado como cenário de comparação: "o imóvel acompanhando a inflação". Não há
índice de preços de imóveis (ex.: FipeZap) para Chapecó, por isso a referência é a
inflação oficial.

- IPCA média anual (padrão): média geométrica dos últimos 10 anos de IPCA mensal.
  Mais estável para projetar vários anos à frente.
- IPCA 12 meses: acumulado dos últimos 12 meses (retrato do momento).

A série é baixada da API pública do BCB e guardada em `dados/ipca.json`, que é usado
quando não houver internet.
"""

import json
import math
from datetime import date
from pathlib import Path

import requests

SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = PROJECT_ROOT / "dados" / "ipca.json"
ANOS_MEDIA = 10
FONTE = "IBGE/BCB — IPCA mensal, série SGS 433"


def fetch_monthly(anos=ANOS_MEDIA, timeout=30):
    """Últimos `anos` anos (+1 de folga) de IPCA mensal em %, do BCB."""
    hoje = date.today()
    inicio = date(hoje.year - anos - 1, 1, 1)
    params = {
        "formato": "json",
        "dataInicial": inicio.strftime("%d/%m/%Y"),
        "dataFinal": hoje.strftime("%d/%m/%Y"),
    }
    resp = requests.get(SGS_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    return [
        {"data": item["data"], "valor": float(item["valor"])} for item in resp.json()
    ]


def summarize(monthly, anos=ANOS_MEDIA):
    """Taxas anuais em % a partir da série mensal."""
    if len(monthly) < 12 * anos:
        raise ValueError(f"Série do IPCA com menos de {anos} anos.")
    ultimos = [m["valor"] for m in monthly[-12 * anos :]]
    fator = math.prod(1 + v / 100 for v in ultimos)
    fator_12m = math.prod(1 + m["valor"] / 100 for m in monthly[-12:])
    return {
        "media_anual_pct": 100 * (fator ** (1 / anos) - 1),
        "acumulado_12m_pct": 100 * (fator_12m - 1),
        "anos_media": anos,
        "inicio": monthly[-12 * anos]["data"],
        "fim": monthly[-1]["data"],
        "fonte": FONTE,
    }


def load_ipca(cache_path=DEFAULT_CACHE, atualizar=True):
    """Resumo do IPCA: baixa do BCB (e atualiza o cache) ou usa o cache local."""
    cache_path = Path(cache_path)
    if atualizar:
        try:
            monthly = fetch_monthly()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(monthly, ensure_ascii=False, indent=0), encoding="utf-8"
            )
            return summarize(monthly)
        except (requests.RequestException, ValueError, KeyError) as exc:
            print(f"[aviso] IPCA não baixado do BCB ({exc}); usando {cache_path}.")
    if not cache_path.exists():
        return None
    return summarize(json.loads(cache_path.read_text(encoding="utf-8")))
