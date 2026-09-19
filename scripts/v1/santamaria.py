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

url = "https://ms-6e17af7c3fc7-10444.nyc.meilisearch.io/indexes/properties/search"

headers = {
    "Authorization": "Bearer c48309808058b7adc4de3241706108614ab9f18527399ff2d5a847affe1c87f3",
    "Content-Type": "application/json",
}

limit = 100
offset = 0

print("SANTA MARIA")
while True:

    body = {
        "filter": ["for_sale = true", "for_lease = false", "category = Residencial"],
        "sort": ["date_registration:desc"],
        "offset": offset,
        "limit": limit,
    }

    r = requests.post(url, headers=headers, json=body)
    data = r.json()

    hits = data["hits"]

    if not hits:
        break

    for p in hits:

        preco = p.get("price_sale")
        area_privada = p.get("private_area")

        preco_m2 = None
        area_total = p.get("total_area")

        # Se só encontrou um valor de área, usa para os dois
        if area_total and not area_privada:
            area_privada = area_total
        elif area_privada and not area_total:
            area_total = area_privada

        area_ref = area_privada or area_total

        if preco and area_ref:
            preco_m2 = preco / area_ref

        cursor.execute(
            """
        INSERT OR REPLACE INTO imoveis
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                "SM-" + str(p.get("id", "")),
                preco,
                preco_m2,
                p.get("address_district"),
                p.get("address_city"),
                p.get("type"),
                area_total if area_total else area_privada,
                area_privada,
                p.get("qtd_bedrooms"),
                p.get("qtd_bathrooms"),
                p.get("qtd_parking_lots"),
                p.get("date_registration"),
                datetime.now().isoformat(),
            ),
        )

    conn.commit()

    print(f"{len(hits)} imóveis processados")

    offset += limit
    sleep(5)

conn.close()

print("Scraping finalizado")
