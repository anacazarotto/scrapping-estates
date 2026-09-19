import re
import sqlite3
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

URL = "https://www.simimoveischapeco.com/retornar-imoveis-disponiveis"

HEADERS = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
    "content-type": "application/x-www-form-urlencoded",
    "origin": "https://www.simimoveischapeco.com",
    "referer": "https://www.simimoveischapeco.com/",
}

PAGE_SIZE = 20
pagina = 1


def parse_valor(valor_str):
    """Converte 'R$ 4.800.000,00' para float."""
    if not valor_str:
        return None
    try:
        limpo = re.sub(r"[^\d,]", "", valor_str).replace(",", ".")
        return float(limpo) if limpo else None
    except (ValueError, TypeError):
        return None


def parse_area(area_str):
    """Converte '560,00' para float."""
    if not area_str:
        return None
    try:
        return float(str(area_str).replace(",", "."))
    except (ValueError, TypeError):
        return None


print("SIM IMOVEIS")
while True:
    form_data = (
        "finalidade=venda&codigounidade=&codigocondominio=0&codigoproprietario=0"
        "&codigocaptador=0&codigosimovei=0"
        "&tipos%5B0%5D%5Bcodigo%5D=2&tipos%5B0%5D%5Bnome%5D=+++++++++Apartamentos&tipos%5B0%5D%5Burl_amigavel%5D=apartamento"
        "&tipos%5B1%5D%5Bcodigo%5D=33&tipos%5B1%5D%5Bnome%5D=++++++++Apartamentos+studios+&tipos%5B1%5D%5Burl_amigavel%5D=apartamento-studio"
        "&tipos%5B2%5D%5Bcodigo%5D=1&tipos%5B2%5D%5Bnome%5D=+++++++Casas&tipos%5B2%5D%5Burl_amigavel%5D=casa"
        "&tipos%5B3%5D%5Bcodigo%5D=16&tipos%5B3%5D%5Bnome%5D=++++++Ch%C3%A1caras&tipos%5B3%5D%5Burl_amigavel%5D=chacara"
        "&tipos%5B4%5D%5Bcodigo%5D=18&tipos%5B4%5D%5Bnome%5D=++++++Coberturas&tipos%5B4%5D%5Burl_amigavel%5D=cobertura"
        "&tipos%5B5%5D%5Bcodigo%5D=3&tipos%5B5%5D%5Bnome%5D=++++Lotes&tipos%5B5%5D%5Burl_amigavel%5D=lote"
        "&tipos%5B6%5D%5Bcodigo%5D=26&tipos%5B6%5D%5Bnome%5D=+++Lotes+rurais&tipos%5B6%5D%5Burl_amigavel%5D=lote-rural"
        "&tipos%5B7%5D%5Bcodigo%5D=20&tipos%5B7%5D%5Bnome%5D=+++Lotes+Urbanos&tipos%5B7%5D%5Burl_amigavel%5D=lote-urbano"
        "&tipos%5B8%5D%5Bcodigo%5D=31&tipos%5B8%5D%5Bnome%5D=+Plantas+&tipos%5B8%5D%5Burl_amigavel%5D=planta"
        "&tipos%5B9%5D%5Bcodigo%5D=29&tipos%5B9%5D%5Bnome%5D=Terreno+&tipos%5B9%5D%5Burl_amigavel%5D=terreno"
        "&codigocidade=2&codigoregiao=0"
        "&bairros%5B0%5D%5Bcidade%5D=&bairros%5B0%5D%5Bcodigo%5D=&bairros%5B0%5D%5Bestado%5D="
        "&bairros%5B0%5D%5BestadoUrl%5D=&bairros%5B0%5D%5Bnome%5D=Todos"
        "&bairros%5B0%5D%5BnomeUrl%5D=todos-os-bairros&bairros%5B0%5D%5Bregiao%5D="
        "&endereco=&edificio=&numeroquartos=0-quartos&numerovagas=0-vagas"
        "&numerobanhos=0-banheiros&numerosuite=0-suites&numerovaranda=0&numeroelevador=0"
        "&valorde=0&valorate=0&areade=0&areaate=0&areaexternade=0&areaexternaate=0"
        "&extras=&destaque=0&opcaoimovel%5Bcodigo%5D=0&opcaoimovel%5Bnome%5D="
        "&opcaoimovel%5BnomeUrl%5D=todas-as-opcoes&codigoOpcaoimovel=0"
        f"&numeropagina={pagina}&numeroregistros={PAGE_SIZE}&ordenacao=valordesc"
        "&cidades%5Bcodigo%5D=2&cidades%5Bnome%5D=Chapec%C3%B3&cidades%5Bestado%5D=SC"
        "&cidades%5BnomeUrl%5D=chapeco&cidades%5BestadoUrl%5D=sc"
        "&condominio%5Bcodigo%5D=0&condominio%5Bnome%5D=&condominio%5BnomeUrl%5D=todos-os-condominios"
    )

    r = requests.post(URL, data=form_data, headers=HEADERS)
    data = r.json()

    results = data.get("lista", [])

    if not results:
        break

    print(f"Página {pagina}: {len(results)} imóveis encontrados")

    for p in results:
        preco = parse_valor(p.get("valor"))

        area_privada = parse_area(p.get("areainterna"))
        area_total = parse_area(p.get("areaprincipal")) or parse_area(p.get("arealote"))

        if area_total and not area_privada:
            area_privada = area_total
        elif area_privada and not area_total:
            area_total = area_privada

        area_ref = area_privada or area_total

        preco_m2 = None
        if preco and area_ref:
            preco_m2 = preco / area_ref

        try:
            quartos = int(p.get("numeroquartos") or 0) or None
        except (ValueError, TypeError):
            quartos = None

        try:
            banheiros = int(p.get("numerobanhos") or 0) or None
        except (ValueError, TypeError):
            banheiros = None

        try:
            vagas = int(p.get("numerovagas") or 0) or None
        except (ValueError, TypeError):
            vagas = None

        cursor.execute(
            """
        INSERT OR REPLACE INTO imoveis
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                "SI-" + str(p.get("codigo", "")),
                preco,
                preco_m2,
                p.get("bairro"),
                p.get("cidade"),
                p.get("tipo"),
                area_total,
                area_privada,
                quartos,
                banheiros,
                vagas,
                p.get("datahoracadastro"),
                datetime.now().isoformat(),
            ),
        )

        print(
            f"  [{p.get('codigo')}] {p.get('tipo')} - {p.get('bairro')} - {p.get('valor')}"
        )

    conn.commit()

    if len(results) < PAGE_SIZE:
        break

    pagina += 1
    sleep(5)

conn.close()
print("Scraping finalizado")
