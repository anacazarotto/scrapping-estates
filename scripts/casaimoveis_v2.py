import re
import time

import requests
from bs4 import BeautifulSoup

from base_scapper import BaseScraper


class CasaImoveisScraper(BaseScraper):
    PREFIX = "C-"
    BASE = "https://www.casaimoveis.net"
    LIST_URL = (
        BASE
        + "/imoveis/list/{}?finalidade=comprar&tipo%5B0%5D=1&tipo%5B1%5D=17&tipo%5B2%5D=21&tipo%5B3%5D=32&tipo%5B4%5D=22&tipo%5B5%5D=8&tipo%5B6%5D=28&tipo%5B7%5D=29&tipo%5B8%5D=30&valor_min=0%2C00&valor_max=0%2C00&cidade%5B0%5D=CHAPEC%C3%93"
    )
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/145.0.0.0 Safari/537.36",
        "Accept-Language": "pt-BR,pt;q=0.9",
    }
    FINAL_PAGE = 60  # Limite de páginas para evitar loops infinitos

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
        nums = re.findall(
            r"\d+\.?\d*",
            (texto or "").replace("R$", "").replace(".", "").replace(",", "."),
        )
        if not nums:
            return None
        try:
            return float(nums[0])
        except ValueError:
            return None

    def extract_image_url(self, soup):
        for selector, attr in (
            ('meta[property="og:image"]', "content"),
            ("#fotos a[href]", "href"),
            ("#fotos img[src]", "src"),
        ):
            node = soup.select_one(selector)
            if not node:
                continue
            value = node.get(attr)
            if value:
                return self.absolutize_url(self.BASE, value)
        return None

    def pegar_links(self, pagina):
        url = self.LIST_URL.format(pagina)
        print(f"[CASAIMOVEIS_V2] Buscando pagina {pagina}: {url}")

        r = self.session.get(url, timeout=30)
        if r.status_code != 200:
            print(f"[CASAIMOVEIS_V2] HTTP {r.status_code} na pagina {pagina}")
            return []

        soup = self._make_soup(r.content)
        cards = soup.select(".imoveis_lista")
        if not cards:
            return []

        links = []
        for card in cards:
            a = card.select_one("a[href]")
            if not a:
                continue
            href = a.get("href")
            if not href:
                continue
            if href.startswith("http"):
                links.append(href)
            else:
                links.append(self.BASE + "/" + href.lstrip("/"))

        return links

    def scrape_imovel(self, url):
        r = self.session.get(url, timeout=30)
        r.raise_for_status()
        soup = self._make_soup(r.content)

        codigo = None
        bairro = None
        cidade = None
        tipo = None

        h1 = soup.select_one(".titulos_pagina h1")
        if h1:
            span = h1.select_one("span")
            if span:
                tipo = span.get_text(strip=True).split()[0]

            h1_texto = " ".join(h1.stripped_strings)

            m_cod = re.search(r"C[oó]digo[:\s]+(\d+)", h1_texto)
            if m_cod:
                codigo = m_cod.group(1)

            m_loc = re.search(
                r"Bairro\s+(.+?)\s+em\s+(.+?)(?:\s*\||\s*$)", h1_texto, re.IGNORECASE
            )
            if m_loc:
                bairro = m_loc.group(1).strip()
                cidade = m_loc.group(2).strip()

        if not codigo:
            m_url = re.search(r"/imovel/(\d+)/", url)
            if m_url:
                codigo = m_url.group(1)

        preco = None
        valor_tag = soup.select_one("#valores_resp .valor span span")
        if valor_tag:
            preco = self.parse_preco(valor_tag.get_text(strip=True))

        quartos = None
        banheiros = None
        vagas = None

        destaques_lis = soup.select("#destaques ul.caracteristicas li")
        caract_lis = soup.select("#caracteristicas li")

        for li in destaques_lis + caract_lis:
            use = li.select_one("use")
            href_val = use.get("href", "") if use else ""
            texto = " ".join(li.stripped_strings).lower()
            num = re.search(r"(\d+)", texto)
            if not num:
                continue
            valor = int(num.group(1))

            if "icone-dormitorio" in href_val or "quarto" in texto or "dormit" in texto:
                if quartos is None:
                    quartos = valor
            elif "icone-vagas" in href_val or "vaga" in texto:
                if vagas is None:
                    vagas = valor
            elif "icone-banheiro" in href_val or "banheiro" in texto:
                if banheiros is None:
                    banheiros = valor

        area_total = None
        area_privada = None

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
            label = " ".join(li.stripped_strings).lower()
            if not strong:
                continue

            m_area = re.search(r"([\d.,]+)\s*m", " ".join(strong.stripped_strings))
            if not m_area:
                continue

            raw = m_area.group(1)
            if "," in raw:
                area_val = float(raw.replace(".", "").replace(",", "."))
            else:
                area_val = float(raw)

            if "privativa" in label or "privado" in label or "edif." in label:
                area_privada = area_val
            else:
                area_total = area_val

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
            "vagas": vagas or 0,
            "date_registration": None,
        }

    def run(self):
        pagina = 1
        while True:
            print(f"[CASAIMOVEIS_V2] Iniciando pagina {pagina}")
            links = self.pegar_links(pagina)

            if not links:
                print(
                    f"[CASAIMOVEIS_V2] Nenhum link encontrado na pagina {pagina}. Encerrando coleta."
                )
                break

            total_links = len(links)
            print(f"[CASAIMOVEIS_V2] Pagina {pagina}: {total_links} links encontrados.")

            for idx, link in enumerate(links, start=1):
                try:
                    print(f"[CASAIMOVEIS_V2] ({idx}/{total_links}) Coletando: {link}")
                    dados = self.scrape_imovel(link)
                    self.upsert_imovel(dados)
                    print(
                        f"[CASAIMOVEIS_V2] ({idx}/{total_links}) Salvo ID={dados.get('id')} URL={link}"
                    )
                    time.sleep(1)
                except Exception as e:
                    print(
                        f"[CASAIMOVEIS_V2] ({idx}/{total_links}) Erro ao processar URL={link}: {e}"
                    )

            print(f"[CASAIMOVEIS_V2] Pagina {pagina} finalizada.")
            pagina += 1
            time.sleep(2)
            if pagina == self.FINAL_PAGE:
                print(f"[CASAIMOVEIS_V2] Página final {self.FINAL_PAGE} atingida. Encerrando.")
                break


if __name__ == "__main__":
    scraper = CasaImoveisScraper()
    try:
        scraper.run()
    finally:
        scraper.close()
