import json
import re
import sqlite3
import unicodedata
from datetime import datetime
from time import sleep

import requests

from db import get_db_path

conn = sqlite3.connect(get_db_path())
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS imoveis (
    id TEXT PRIMARY KEY,
    preco REAL,
    preco_m2 REAL,
    bairro TEXT,
    cidade TEXT,
    tipo_imovel TEXT,
    area_total REAL,
    area_privada REAL,
    quartos INTEGER,
    banheiros INTEGER,
    vagas INTEGER,
    date_registration TEXT,
    data_insercao TEXT
)
""")

URL_LISTA = "https://padra.simob.com.br/v2/integracaoApi/imovel/filtro/categoria/caracteristicas"
URL_DETALHE = (
    "https://padra.simob.com.br/v2/integracaoApi/detalhes/imovel/{codigo}"
    "?calcularValorAbono=false&considerarPrevisaoSaida=false"
    "&somenteTelefonePublicarSite=true&validadeOpcaoVenda=false"
)

HEADERS = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
    "authorization": "Bearer 324545ec5ba67f9e6fac78137631cc2b",  # Token genérico encontrado nas requisições do site
    "from": "site",
    "origin": "https://www.padra.com.br",
    "referer": "https://www.padra.com.br/",
}

MAX_RESULTS = 20
first_result = 0

# IDs das características no retorno de DETALHE (campo idCaracteristica)
ID_QUARTOS = 89  # Dormitório(s)
ID_QUARTOS_FALLBACK = 103  # Sendo suite(s) — fallback quando 89 não estiver presente
ID_BANHEIROS = 84  # Banheiro(s)
ID_VAGAS = 104  # Vaga de garagem
ID_AREA_PRIVADA = 124  # Área privativa - m²
ID_AREA_TERRENO = 107  # Área do terreno - m²
ID_AREA_TOTAL = 123  # Área total - m²
ID_AREA_TOTAL_EDIF = 109  # Área total edif - m²
ID_AREA_UNIDADE = 106  # Área da unidade - m²


def get_carac_valor(caracteristicas, id_carac):
    """Busca pelo campo idCaracteristica (retorno da API de detalhe)."""
    for c in caracteristicas:
        if c.get("idCaracteristica") == id_carac:
            try:
                val = c.get("valor", 0)
                return float(val) if val not in (None, "", "0") else None
            except (ValueError, TypeError):
                return None
    return None


# Tipos a ignorar
TIPOS_IGNORADOS = {"Barracão", "Barraco", "Sala Comercial", "Sala"}


def normalize_str(s):
    if s is None:
        return ""
    return (
        unicodedata.normalize("NFD", str(s))
        .encode("ascii", "ignore")
        .decode()
        .lower()
        .strip()
    )


def parse_quartos_text(val):
    """Converte valores variados de 'Dormitórios' vindos da API em inteiro.

    A API às vezes retorna números (ex: 2) ou strings (ex: 'Suíte +1', '2 Suítes +1').
    Regras:
      - Se for numérico, retorna int.
      - Se contiver 'su' (suíte variações), soma os números encontrados; 'Suíte +1' -> 2.
      - Se apenas números presentes, retorna o primeiro.
    """
    if val is None:
        return None

    # já é número
    if isinstance(val, (int, float)):
        try:
            return int(val)
        except Exception:
            return None

    txt = normalize_str(val)
    nums = [int(n) for n in re.findall(r"\d+", txt)]
    has_suite = "su" in txt  # pega suite, suite com acento vira 'su'

    if has_suite:
        if len(nums) >= 2:
            return sum(nums)
        if len(nums) == 1:
            # 'suite +1' -> 1 + 1
            if "+" in txt and not txt.startswith(str(nums[0])):
                return 1 + nums[0]
            return nums[0]
        return 1

    if nums:
        return nums[0]

    return None


def main():
    global first_result
    while True:
        payload_inner = {
            "idsCategorias": [],
            "finalidade": 2,
            "ceps": ["CHAPECÓ"],
            "idsBairros": [],
            "rangeValue": {"max": "", "min": ""},
            "caracteristicas": [
                {
                    "id": 89,
                    "idTipoCaracteristica": 3,
                    "qtd": 0,
                    "considerarValorExato": False,
                },
                {
                    "id": 103,
                    "idTipoCaracteristica": 3,
                    "qtd": 0,
                    "considerarValorExato": False,
                },
                {
                    "id": 140,
                    "idTipoCaracteristica": 3,
                    "qtd": 0,
                    "considerarValorExato": False,
                },
            ],
            "selectedOptions": {
                "categorias": [],
                "caracteristicas": [
                    {
                        "id": 89,
                        "descricao": "Dormitório(s)",
                        "tipoCaracteristica": 3,
                        "ordem": 1,
                        "grupo": {"id": 1, "descricao": "Físicas"},
                        "publicar": 1,
                        "buscaImovel": 1,
                        "finalidade": 3,
                        "value": 0,
                    },
                    {
                        "id": 103,
                        "descricao": "Sendo suite(s)",
                        "tipoCaracteristica": 3,
                        "ordem": 2,
                        "grupo": {"id": 1, "descricao": "Físicas"},
                        "publicar": 1,
                        "buscaImovel": 0,
                        "finalidade": 3,
                        "value": 0,
                    },
                    {
                        "id": 140,
                        "descricao": "Sendo demi-suíte(s)",
                        "tipoCaracteristica": 3,
                        "ordem": 3,
                        "grupo": {"id": 1, "descricao": "Físicas"},
                        "publicar": 1,
                        "buscaImovel": 1,
                        "finalidade": 3,
                        "value": 0,
                    },
                ],
                "bairros": [],
                "cidades": [
                    {
                        "cidade": "CHAPECÓ",
                        "uf": "SC",
                        "count": 373,
                        "idsImoveis": "3040,21,3174,94,3651,2828,1887,1886,1676,463,3239,638,3111,803,3113,859,3879,3863,3466,3372,1332,1733,1507,3620,3619,3617,1644,1668,3854,3717,2612,1716,1793,1792,1791,2969,2230,2220,1867,3636,2337,1907,3868,3701,3647,1972,1977,2023,3495,3494,3490,3373,3358,2264,2242,3838,2267,2329,3210,2785,2352,2376,2396,2443,2481,2483,2626,2812,2970,3033,2994,3094,3165,3189,3207,3234,3244,3257,3272,3282,3287,3360,4038,3377,3670,3695,3698,3705,3722,3723,3728,3792,1402,979,1629,3341,3340,3427,117,2373,962,2006,3779,3689,3453,3375,3175,3195,3330,3587,3600,3790,3802,242,3671,3260,1575,1574,1573,1572,2084,3884,3883,3882,3862,3843,3455,2300,2557,2638,2745,2968,3221,1790,497,4028,953,4030,1732,1928,3823,2146,2183,3560,3442,3371,3043,2314,2334,3215,2519,2734,3767,3308,3470,3980,3761,3894,2960,769,3759,1563,2285,3562,2681,3533,3610,3913,582,3632,2283,2648,3262,1949,589,3727,2224,2259,3158,4014,2893,594,3567,1773,1362,1779,1894,3501,2364,2950,2554,2649,2774,3860,2875,2902,3028,3053,3318,3777,3510,3580,3664,2995,3613,3450,3115,3199,3338,3504,905,1487,1875,3557,920,1711,3425,2340,3903,2696,3108,2920,2918,3091,3184,3316,3789,3322,3412,3449,3730,1175,2118,3445,2407,3201,3801,3807,1542,1541,1539,1538,2883,3109,2997,2669,2839,2694,3259,2733,2515,3799,3601,3774,3773,3410,3350,3349,3348,3347,3346,2813,1643,1641,1927,3755,3548,2463,2513,2955,2527,2860,2870,3899,1694,3550,3435,2260,2465,3681,3766,3848,3576,3374,1892,2077,3577,2668,3744,3312,3421,3446,3624,3699,3700,3857,4032,1962,3758,3625,2007,3685,3682,2824,3828,3409,3488,3279,3324,3851,3961,2020,3406,2101,2827,2899,3217,2132,2539,3826,3263,3359,3447,3500,2149,3046,2204,2410,2990,2993,2992,3518,3661,4027,3416,3478,3477,3441,3436,2602,3529,3564,4009,3444,3463,3538,3521,2533,2593,2673,2865,2884,3757,2287,2317,2471,2728,2371,2444,2482,2547,2549,2576,2751,3211,2971,3229,3247,3678,3688,3690,3778,3904",
                        "idsCategorias": "41,33,40,47,35,39,37,44,54",
                    }
                ],
                "finalidade": "Comprar",
                "range": {"maxRange": "", "minRange": ""},
            },
            "offset": {"maxResults": MAX_RESULTS, "firstResult": first_result},
            "acuracidade": 100,
            "countResults": False,
            "considerarPrevisaoSaida": False,
            "calcularValorAbono": False,
            "validade_opcao_venda": False,
            "orderBy": [
                {
                    "sort": "valor",
                    "descricao": "Valor",
                    "order": "desc",
                    "active": False,
                    "type": "number",
                },
                {"sort": "metrica", "order": "desc"},
            ],
            "trazerCaracteristicas": 3,
        }

        payload = {"data": json.dumps(payload_inner, ensure_ascii=False)}
        r = requests.post(URL_LISTA, data=payload, headers=HEADERS)
        data = r.json()
        results = data.get("result", [])

        if not results:
            break

        print(f"Página firstResult={first_result}: {len(results)} imóveis encontrados")

        for item in results:
            codigo = item.get("codigo")
            if not codigo:
                continue

            # Consulta a API de detalhe pelo código
            r_det = requests.get(URL_DETALHE.format(codigo=codigo), headers=HEADERS)
            det_data = r_det.json()
            det_list = det_data.get("result", [])

            if not det_list:
                print(f"  Detalhe não encontrado para código {codigo}")
                sleep(1)
                continue

            p = det_list[0]
            caracteristicas = p.get("caracteristicas", [])

            # Preço vem de configVenda.valor
            config_venda = p.get("configVenda") or {}
            preco_str = config_venda.get("valor")
            try:
                preco = float(preco_str) if preco_str else None
            except (ValueError, TypeError):
                preco = None

            # Área privativa (id 124) com fallbacks
            area_privada = get_carac_valor(caracteristicas, ID_AREA_PRIVADA)
            area_total = (
                get_carac_valor(caracteristicas, ID_AREA_TOTAL_EDIF)
                or get_carac_valor(caracteristicas, ID_AREA_TERRENO)
                or get_carac_valor(caracteristicas, ID_AREA_TOTAL)
                or get_carac_valor(caracteristicas, ID_AREA_UNIDADE)
            )

            if area_total and not area_privada:
                area_privada = area_total
            elif area_privada and not area_total:
                area_total = area_privada

            area_ref = area_privada or area_total

            preco_m2 = None
            if preco and area_ref:
                preco_m2 = preco / area_ref

            # quartos/banheiros/vagas podem vir como número ou string (ex: 'Suíte +1')
            raw_quartos = None
            for c in caracteristicas:
                if c.get("idCaracteristica") == ID_QUARTOS:
                    raw_quartos = (
                        c.get("valor")
                        or c.get("valorFormatado")
                        or c.get("valorText")
                        or c.get("descricao")
                    )
                    break
            # fallback: usa id 103 (Sendo suite(s)) se id 89 não foi encontrado
            if not raw_quartos:
                for c in caracteristicas:
                    if c.get("idCaracteristica") == ID_QUARTOS_FALLBACK:
                        raw_quartos = (
                            c.get("valor")
                            or c.get("valorFormatado")
                            or c.get("valorText")
                            or c.get("descricao")
                        )
                        break
            quartos = parse_quartos_text(raw_quartos)

            raw_banheiros = None
            for c in caracteristicas:
                if c.get("idCaracteristica") == ID_BANHEIROS:
                    raw_banheiros = (
                        c.get("valor")
                        or c.get("valorFormatado")
                        or c.get("valorText")
                        or c.get("descricao")
                    )
                    break
            try:
                banheiros = (
                    int(raw_banheiros) if raw_banheiros not in (None, "") else None
                )
            except Exception:
                banheiros = None

            raw_vagas = None
            for c in caracteristicas:
                if c.get("idCaracteristica") == ID_VAGAS:
                    raw_vagas = (
                        c.get("valor")
                        or c.get("valorFormatado")
                        or c.get("valorText")
                        or c.get("descricao")
                    )
                    break
            try:
                vagas = int(raw_vagas) if raw_vagas not in (None, "") else None
            except Exception:
                vagas = None

            # Tipo vem de categoria.descricao
            categoria = p.get("categoria") or {}
            tipo_imovel = categoria.get("descricao") or item.get("descricaoCategoria")

            # Ignora Barracão e Sala Comercial
            if normalize_str(tipo_imovel) in [
                normalize_str(t) for t in TIPOS_IGNORADOS
            ]:
                print(f"  [{codigo}] Ignorado (tipo: {tipo_imovel})")
                sleep(1)
                continue

            cursor.execute(
                """
            INSERT OR REPLACE INTO imoveis
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    "P-" + str(p.get("id", codigo)),
                    preco,
                    preco_m2,
                    p.get("bairro"),
                    p.get("cidade"),
                    tipo_imovel,
                    area_total,
                    area_privada,
                    int(quartos) if quartos else None,
                    int(banheiros) if banheiros else None,
                    int(vagas) if vagas else None,
                    p.get("dataPublicacao") or item.get("updatedAt"),
                    datetime.now().isoformat(),
                ),
            )

            print(f"  [{codigo}] {tipo_imovel} - {p.get('bairro')} - R$ {preco}")
            sleep(1)

        conn.commit()

        if len(results) < MAX_RESULTS:
            break

        first_result += MAX_RESULTS
        sleep(5)

    conn.close()
    print("Scraping finalizado")


if __name__ == "__main__":
    print("PADRA")
    main()
