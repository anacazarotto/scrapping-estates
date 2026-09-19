import re
import sqlite3
import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from db import get_db_path

BASE = "https://nostracasa.com.br"

LIST_URL = (
    BASE
    + "/comprar/apartamento+apartamento-cobertura+area-de-terra+casa+casa-geminada+empreendimento+predio+terreno/chapeco/?jsf=jet-engine:imoveis&pagenum={}"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/145.0.0.0 Safari/537.36",
    "Accept-Language": "pt-BR,pt;q=0.9",
}

session = requests.Session()
session.headers.update(HEADERS)

# ----------------------------
# BANCO
# ----------------------------

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

# ----------------------------
# PARSERS
# ----------------------------


def parse_preco(texto):

    m = re.search(r"R\$\s*([\d\.]+,\d+)", texto)

    if not m:
        return None

    valor = m.group(1).replace(".", "").replace(",", ".")
    return float(valor)


# ----------------------------
# SCRAPER DE DETALHE
# ----------------------------


def extrair_carac(soup):
    """Extrai características do bloco .carac-list na página de detalhe."""
    carac = {}
    for item in soup.select(".carac"):
        titulo = item.select_one(".carac__title")
        valor = item.select_one(".carac__value")
        if titulo and valor:
            carac[titulo.get_text(strip=True).lower()] = valor.get_text(strip=True)
    return carac


def scrape_imovel(url):

    r = session.get(url)
    soup = BeautifulSoup(r.text, "html.parser")

    # --- Preço ---
    preco = None
    for div in soup.select(".jet-listing-dynamic-field__content"):
        texto_div = div.get_text(" ", strip=True)
        p = parse_preco(texto_div)
        if p:
            preco = p
            break

    # --- Código / Referência ---
    codigo = None
    for div in soup.select(".jet-listing-dynamic-field__content"):
        texto_div = div.get_text(" ", strip=True)
        m = re.search(r"Refer[eê]ncia[:\s]+(\d+)", texto_div)
        if m:
            codigo = m.group(1)
            break

    # --- Bairro, Cidade, Tipo via classes do container elementor ---
    bairro = None
    cidade = None
    tipo = None
    container = soup.find(attrs={"data-elementor-type": "single-post"})
    if container:
        classes = container.get("class", [])
        for cls in classes:
            if cls.startswith("bairro-"):
                bairro = cls.replace("bairro-", "").replace("-", " ").title()
            elif cls.startswith("cidade-"):
                cidade = cls.replace("cidade-", "").replace("-", " ").title()
            elif cls.startswith("tipo-"):
                tipo = cls.replace("tipo-", "").replace("-", " ").title()

    # --- Características do imóvel ---
    carac = extrair_carac(soup)

    def carac_float(chaves):
        for chave in chaves:
            for k, v in carac.items():
                if chave in k:
                    try:
                        v_clean = v.strip()
                        # Se tem vírgula como decimal (ex: "93,67") ou ponto de milhar (ex: "1.234,56")
                        if "," in v_clean:
                            v_clean = v_clean.replace(".", "").replace(",", ".")
                        # Se já usa ponto como decimal (ex: "93.67") — não altera
                        return float(v_clean)
                    except ValueError:
                        pass
        return None

    def carac_int(chaves):
        for chave in chaves:
            for k, v in carac.items():
                if chave in k:
                    m = re.search(r"\d+", v)
                    if m:
                        return int(m.group())
        return 0

    area_total = carac_float(
        ["areatotal", "area construída", "área construída", "area total", "área total"]
    )
    area_privada = carac_float(
        [
            "areaprivativa",
            "área privativa",
            "area privativa",
            "área do terreno",
            "area do terreno",
        ]
    )
    quartos = carac_int(["dormitorio", "quarto"])
    banheiros = carac_int(["banheiro"])
    vagas = carac_int(["estacionamento", "vaga"])

    # Se só encontrou um valor de área, usa para os dois
    if area_total and not area_privada:
        area_privada = area_total
    elif area_privada and not area_total:
        area_total = area_privada

    preco_m2 = None
    area_ref = area_privada or area_total
    if preco and area_ref and area_ref > 0:
        preco_m2 = preco / area_ref

    return {
        "id": str(codigo) if codigo else None,
        "preco": preco,
        "preco_m2": preco_m2,
        "bairro": bairro,
        "cidade": cidade,
        "tipo_imovel": tipo,
        "area_total": area_total,
        "area_privada": area_privada,
        "quartos": quartos,
        "banheiros": banheiros,
        "vagas": vagas,
    }


# ----------------------------
# SCRAPER DE LISTA
# ----------------------------


def pegar_links(pagina):

    url = LIST_URL.format(pagina)

    print("Página", pagina)

    r = session.get(url)
    if r.status_code != 200:
        return []

    soup = BeautifulSoup(r.text, "html.parser")

    cards = soup.select(".cx-imovel")

    links = []

    for c in cards:

        link = c.get("data-permalink")

        if link:
            links.append(link)

    return links


# ----------------------------
# SALVAR BANCO
# ----------------------------


def salvar(d):

    if not d["id"]:
        return

    data_insercao = datetime.now().isoformat()

    cursor.execute(
        """
    INSERT OR REPLACE INTO imoveis VALUES (
        ?,?,?,?,?,?,?,?,?,?,?,?,?
    )
    """,
        (
            "N-" + str(d["id"]),
            d["preco"],
            d["preco_m2"],
            d["bairro"],
            d["cidade"],
            d["tipo_imovel"],
            d["area_total"],
            d["area_privada"],
            d["quartos"],
            d["banheiros"],
            d["vagas"],
            None,
            data_insercao,
        ),
    )

    conn.commit()


# ----------------------------
# LOOP PRINCIPAL
# ----------------------------

pagina = 1

while True:
    print("NOSTRACASA")

    links = pegar_links(pagina)

    if not links:
        print("acabou")
        break

    for link in links:

        try:

            dados = scrape_imovel(link)

            salvar(dados)

            print("salvo:", dados["id"])

            time.sleep(1)

        except Exception as e:

            print("erro:", e)

    pagina += 1
