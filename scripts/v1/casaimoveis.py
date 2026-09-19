import re
import sqlite3
import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from db import get_db_path

BASE_URL = "https://www.casaimoveis.net/imoveis/list/{}?finalidade=comprar&tipo%5B0%5D=1&tipo%5B1%5D=17&tipo%5B2%5D=21&tipo%5B3%5D=32&tipo%5B4%5D=22&tipo%5B5%5D=8&tipo%5B6%5D=28&tipo%5B7%5D=29&tipo%5B8%5D=30&valor_min=0%2C00&valor_max=0%2C00&cidade%5B0%5D=CHAPEC%C3%93"
BASE = "https://www.casaimoveis.net"

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

conn.commit()

# ----------------------------
# SCRAPER DE LISTA — coleta links
# ----------------------------


def pegar_links(pagina):

    url = BASE_URL.format(pagina)
    print(f"Página {pagina}: {url}")

    r = session.get(url, timeout=10)
    if r.status_code != 200:
        print("Erro HTTP:", r.status_code)
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    cards = soup.select(".imoveis_lista")

    if not cards:
        return []

    links = []
    for card in cards:
        a = card.select_one("a[href]")
        if a:
            href = a["href"]
            # href pode ser relativo ("imovel/32908/...") ou absoluto
            if href.startswith("http"):
                links.append(href)
            else:
                links.append(BASE + "/" + href.lstrip("/"))

    return links


# ----------------------------
# SCRAPER DE DETALHE
# ----------------------------


def scrape_imovel(url):

    r = session.get(url, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")

    # --- Código, Bairro, Cidade, Tipo via <h1> ---
    # Estrutura: <span>Casa 3 quartos à venda</span>\n Bairro Líder em Chapecó | Código: 15299
    codigo = None
    bairro = None
    cidade = None
    tipo = None

    h1 = soup.select_one(".titulos_pagina h1")
    if h1:
        span = h1.select_one("span")
        if span:
            tipo = span.get_text(strip=True).split()[0]  # "Casa", "Apartamento", etc.

        h1_texto = h1.get_text(" ", strip=True)

        m_cod = re.search(r"C[oó]digo[:\s]+(\d+)", h1_texto)
        if m_cod:
            codigo = m_cod.group(1)

        # "Bairro Líder em Chapecó"
        m_loc = re.search(
            r"Bairro\s+(.+?)\s+em\s+(.+?)(?:\s*\||\s*$)", h1_texto, re.IGNORECASE
        )
        if m_loc:
            bairro = m_loc.group(1).strip()
            cidade = m_loc.group(2).strip()

    # Fallback: extrai código da URL
    if not codigo:
        m_url = re.search(r"/imovel/(\d+)/", url)
        if m_url:
            codigo = m_url.group(1)

    # --- Preço ---
    preco = None
    valor_tag = soup.select_one("#valores_resp .valor span span")
    if valor_tag:
        texto = valor_tag.get_text(strip=True)
        texto = texto.replace("R$", "").replace(".", "").replace(",", ".")
        nums = re.findall(r"\d+\.?\d*", texto)
        if nums:
            preco = float(nums[0])

    # --- Quartos, Vagas, Banheiros ---
    # Busca primeiro em #destaques (apartamentos), depois em #caracteristicas (casas/prédios)
    quartos = banheiros = vagas = None

    destaques_lis = soup.select("#destaques ul.caracteristicas li")
    caract_lis = soup.select("#caracteristicas li")

    for li in destaques_lis + caract_lis:
        use = li.select_one("use")
        href_val = use.get("href", "") if use else ""
        texto = li.get_text(" ", strip=True).lower()
        num = re.search(r"(\d+)", texto)
        if not num:
            continue
        valor = int(num.group(1))
        # Usa o ícone como sinal primário; texto como fallback
        if "icone-dormitorio" in href_val or "quarto" in texto or "dormit" in texto:
            if quartos is None:
                quartos = valor
        elif "icone-vagas" in href_val or "vaga" in texto:
            if vagas is None:
                vagas = valor
        elif "icone-banheiro" in href_val or "banheiro" in texto:
            if banheiros is None:
                banheiros = valor

    # --- Área: li com #icone-area dentro de #destaques ou #caracteristicas ---
    area_total = area_privada = None

    # Busca em ambas as seções: #destaques (apartamentos) e #caracteristicas (casas/outros)
    area_lis = soup.select("#destaques ul.caracteristicas li") + soup.select(
        "#caracteristicas li"
    )

    for li in area_lis:
        use = li.select_one("use")
        if not use:
            continue
        href_val = use.get("href", "") or use.get("xlink:href", "")
        if "#icone-area" not in href_val or "servico" in href_val:
            continue
        strong = li.select_one("strong")
        label = li.get_text(" ", strip=True).lower()
        if strong:
            m_area = re.search(r"([\d\.,]+)\s*m", strong.get_text(" ", strip=True))
            if m_area:
                raw = m_area.group(1)
                # Formato PT-BR com vírgula decimal: "165,56" ou "1.200,00"
                if "," in raw:
                    area_val = float(raw.replace(".", "").replace(",", "."))
                else:
                    # Ponto como decimal: "58.00", "110.00" — não remover
                    area_val = float(raw)
                if "privativa" in label or "privado" in label or "edif." in label:
                    area_privada = area_val
                else:
                    area_total = area_val

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
        "id": codigo,
        "preco": preco,
        "preco_m2": preco_m2,
        "bairro": bairro,
        "cidade": cidade,
        "tipo_imovel": tipo,
        "area_total": area_total,
        "area_privada": area_privada,
        "quartos": quartos,
        "banheiros": banheiros,
        "vagas": vagas or 0,
    }


# ----------------------------
# SALVAR BANCO
# ----------------------------


def salvar(d):

    if not d["id"]:
        return

    cursor.execute(
        """
    INSERT OR REPLACE INTO imoveis VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """,
        (
            f"C-{d['id']}",
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
            datetime.now().isoformat(),
        ),
    )

    conn.commit()


# ----------------------------
# LOOP PRINCIPAL
# ----------------------------

pagina = 1

while True:
    print("CASA IMOVEIS")

    links = pegar_links(pagina)

    if not links:
        print("Sem mais imóveis. Encerrando.")
        break

    for link in links:

        try:

            dados = scrape_imovel(link)

            salvar(dados)

            print(
                "Salvo:",
                dados["id"],
                "|",
                dados["preco"],
                "|",
                dados["bairro"],
                dados["cidade"],
            )

            time.sleep(1)

        except Exception as e:
            print("Erro ao processar:", link, "->", e)

    pagina += 1
    time.sleep(2)

    if pagina == 60:
        break


conn.close()
