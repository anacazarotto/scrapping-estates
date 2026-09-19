import re
import time
from typing import Any

import requests
from bs4 import BeautifulSoup

from base_scapper import BaseScraper


class NostraCasaScraper(BaseScraper):
    PREFIX = "N-"
    BASE = "https://nostracasa.com.br"
    LIST_URL = (
        BASE
        + "/comprar/apartamento+apartamento-cobertura+area-de-terra+casa+casa-geminada+empreendimento+predio+terreno/chapeco/?jsf=jet-engine:imoveis&pagenum={}"
    )
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/145.0.0.0 Safari/537.36",
        "Accept-Language": "pt-BR,pt;q=0.9",
    }

    def __init__(self):
        super().__init__()
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)

    @staticmethod
    def _make_soup(html):
        try:
            return BeautifulSoup(html, "lxml")
        except Exception:
            return BeautifulSoup(html, "html.parser")

    @staticmethod
    def parse_preco(texto):
        m = re.search(r"R\$\s*([\d.]+,\d+)", texto or "")
        if not m:
            return None
        return float(m.group(1).replace(".", "").replace(",", "."))

    @staticmethod
    def extrair_carac(soup):
        carac = {}
        for item in soup.select(".carac"):
            titulo = item.select_one(".carac__title")
            valor = item.select_one(".carac__value")
            if titulo and valor:
                carac[titulo.get_text(strip=True).lower()] = valor.get_text(strip=True)
        return carac

    def extract_image_url(self, soup):
        for selector, attr in (
            ('meta[property="og:image"]', "content"),
            (".swiper-slide img[src]", "src"),
            ("img[src]", "src"),
        ):
            node = soup.select_one(selector)
            if not node:
                continue
            value = node.get(attr)
            if value:
                return self.absolutize_url(self.BASE, value)
        return None

    def scrape_imovel(self, url):
        r = self.session.get(url, timeout=30)
        r.raise_for_status()
        soup = self._make_soup(r.content)

        preco = None
        for div in soup.select(".jet-listing-dynamic-field__content"):
            texto_div = " ".join(div.stripped_strings)
            p = self.parse_preco(texto_div)
            if p:
                preco = p
                break

        codigo = None
        for div in soup.select(".jet-listing-dynamic-field__content"):
            texto_div = " ".join(div.stripped_strings)
            m = re.search(r"Refer[eê]ncia[:\s]+(\d+)", texto_div)
            if m:
                codigo = m.group(1)
                break

        bairro = None
        cidade = None
        tipo = None
        container = soup.find(attrs={"data-elementor-type": "single-post"})
        if container:
            raw_classes: Any = container.attrs["class"] if "class" in container.attrs else []
            if isinstance(raw_classes, str):
                classes = [raw_classes]
            elif raw_classes:
                classes = [str(cls) for cls in raw_classes]
            else:
                classes = []
            for cls in classes:
                if cls.startswith("bairro-"):
                    bairro = cls.replace("bairro-", "").replace("-", " ").title()
                elif cls.startswith("cidade-"):
                    cidade = cls.replace("cidade-", "").replace("-", " ").title()
                elif cls.startswith("tipo-"):
                    tipo = cls.replace("tipo-", "").replace("-", " ").title()

        carac = self.extrair_carac(soup)

        def carac_float(chaves):
            for chave in chaves:
                for k, v in carac.items():
                    if chave in k:
                        try:
                            raw = v.strip()
                            if "," in raw:
                                raw = raw.replace(".", "").replace(",", ".")
                            return float(raw)
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
            [
                "areatotal",
                "area construída",
                "área construída",
                "area total",
                "área total",
            ]
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

        if area_total and not area_privada:
            area_privada = area_total
        elif area_privada and not area_total:
            area_total = area_privada

        area_ref = area_privada or area_total
        preco_m2 = (preco / area_ref) if (preco and area_ref and area_ref > 0) else None

        return {
            "id": str(codigo) if codigo else None,
            "preco": preco,
            "preco_m2": preco_m2,
            "bairro": bairro,
            "endereco": None,
            "imagem_url": self.extract_image_url(soup),
            "cidade": cidade,
            "tipo_imovel": tipo,
            "area_total": area_total,
            "area_privada": area_privada,
            "quartos": quartos,
            "banheiros": banheiros,
            "vagas": vagas,
            "date_registration": None,
        }

    def pegar_links(self, pagina):
        url = self.LIST_URL.format(pagina)
        print("Pagina", pagina)

        r = self.session.get(url, timeout=30)
        if r.status_code != 200:
            return []

        soup = self._make_soup(r.content)
        cards = soup.select(".cx-imovel")

        links = []
        for c in cards:
            link = c.get("data-permalink")
            if link:
                links.append(link)
        return links

    def run(self):
        pagina = 1
        while True:
            print(f"[NOSTRACASA_V2] Iniciando pagina {pagina}")
            links = self.pegar_links(pagina)

            if not links:
                print(
                    f"[NOSTRACASA_V2] Nenhum link encontrado na pagina {pagina}. Encerrando coleta."
                )
                break

            total_links = len(links)
            print(f"[NOSTRACASA_V2] Pagina {pagina}: {total_links} links encontrados.")

            for idx, link in enumerate(links, start=1):
                try:
                    print(f"[NOSTRACASA_V2] ({idx}/{total_links}) Coletando: {link}")
                    dados = self.scrape_imovel(link)
                    self.upsert_imovel(dados)
                    print(
                        f"[NOSTRACASA_V2] ({idx}/{total_links}) Salvo ID={dados.get('id')} URL={link}"
                    )
                    time.sleep(1)
                except Exception as e:
                    print(
                        f"[NOSTRACASA_V2] ({idx}/{total_links}) Erro ao processar URL={link}: {e}"
                    )

            print(f"[NOSTRACASA_V2] Pagina {pagina} finalizada.")
            pagina += 1


if __name__ == "__main__":
    scraper = NostraCasaScraper()
    try:
        scraper.run()
    finally:
        scraper.close()
