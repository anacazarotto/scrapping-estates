import time

import requests

from base_scapper import BaseScraper
from db.config import require_env_value


class SantaMariaScraper(BaseScraper):
    PREFIX = "SM-"
    URL = "https://ms-32e09cad5e12-10555.sao.meilisearch.io/indexes/properties/search"
    HEADERS = {
        "Content-Type": "application/json",
    }
    LIMIT = 100

    def __init__(self):
        super().__init__()
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self.session.headers["Authorization"] = (
            f"Bearer {require_env_value('SANTAMARIA_API_TOKEN')}"
        )

    @staticmethod
    def build_body(offset, limit):
        return {
            "filter": [
                "for_sale = true",
                "for_lease = false",
                "category = Residencial",
            ],
            "sort": ["date_registration:desc"],
            "offset": offset,
            "limit": limit,
        }

    def pegar_lote(self, offset):
        body = self.build_body(offset=offset, limit=self.LIMIT)
        r = self.session.post(self.URL, json=body, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data.get("hits", [])

    @staticmethod
    def montar_dados(p):
        preco = p.get("price_sale")
        area_privada = p.get("private_area")
        area_total = p.get("total_area")

        # Se só encontrou uma área, replica para manter consistência no banco.
        if area_total and not area_privada:
            area_privada = area_total
        elif area_privada and not area_total:
            area_total = area_privada

        area_ref = area_privada or area_total
        preco_m2 = (preco / area_ref) if (preco and area_ref) else None
        images = p.get("images") or []
        image_url = p.get("cover_original_url") or p.get("cover_small_url")
        if not image_url and images:
            image_url = images[0].get("url_original") or images[0].get("url_small")

        return {
            "id": str(p.get("id", "")),
            "preco": preco,
            "preco_m2": preco_m2,
            "bairro": p.get("address_district"),
            "endereco": BaseScraper.format_endereco(
                p.get("address_street"),
                p.get("address_number"),
                p.get("address_complement"),
            ),
            "imagem_url": image_url,
            "cidade": p.get("address_city"),
            "tipo_imovel": p.get("type"),
            "area_total": area_total if area_total else area_privada,
            "area_privada": area_privada,
            "quartos": p.get("qtd_bedrooms"),
            "banheiros": p.get("qtd_bathrooms"),
            "vagas": p.get("qtd_parking_lots"),
            "date_registration": p.get("date_registration"),
        }

    def run(self):
        print("[SANTAMARIA_V2] Iniciando coleta")
        offset = 0
        total_salvos = 0

        while True:
            print(f"[SANTAMARIA_V2] Buscando lote offset={offset} limit={self.LIMIT}")
            try:
                hits = self.pegar_lote(offset)
            except Exception as e:
                print(f"[SANTAMARIA_V2] Erro na consulta offset={offset}: {e}")
                break

            if not hits:
                print(
                    f"[SANTAMARIA_V2] Nenhum imovel retornado em offset={offset}. Encerrando coleta."
                )
                break

            total_lote = len(hits)
            print(f"[SANTAMARIA_V2] offset={offset}: {total_lote} imoveis encontrados.")

            for idx, p in enumerate(hits, start=1):
                try:
                    dados = self.montar_dados(p)
                    if not dados.get("id"):
                        print(
                            f"[SANTAMARIA_V2] ({idx}/{total_lote}) Imovel sem id. Ignorado."
                        )
                        continue

                    self.upsert_imovel(dados)
                    total_salvos += 1
                    print(
                        f"[SANTAMARIA_V2] ({idx}/{total_lote}) Salvo ID={dados.get('id')} "
                        f"bairro={dados.get('bairro')} preco={dados.get('preco')}"
                    )
                except Exception as e:
                    print(
                        f"[SANTAMARIA_V2] ({idx}/{total_lote}) Erro ao processar item: {e}"
                    )

            if total_lote < self.LIMIT:
                print(
                    "[SANTAMARIA_V2] Ultimo lote identificado pela quantidade de resultados."
                )
                break

            offset += self.LIMIT
            time.sleep(5)

        print(f"[SANTAMARIA_V2] Coleta finalizada. Total salvos={total_salvos}")


if __name__ == "__main__":
    scraper = SantaMariaScraper()
    try:
        scraper.run()
    finally:
        scraper.close()
