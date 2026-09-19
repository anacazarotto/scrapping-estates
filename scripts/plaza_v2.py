import re
import time
import unicodedata

import requests
from bs4 import BeautifulSoup

from base_scapper import BaseScraper


class PlazaScraper(BaseScraper):
    PREFIX = "PL-"
    BASE_URL = "https://plazachapeco.com.br"
    URL_LIST_TEMPLATE = "https://plazachapeco.com.br/comprar/imoveis/"
    URL_LIST_PAGE = "https://plazachapeco.com.br/comprar/imoveis/pag-{n}/"

    HEADERS = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "accept-language": "pt-BR,pt;q=0.9,en-US;q=0.8",
        "user-agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }

    TIPOS_IGNORADOS = {
        "barracao",
        "barraco",
        "sala comercial",
        "sala/conjunto",
        "sala-comercial",
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

    @staticmethod
    def parse_preco(texto):
        if not texto:
            return None
        texto = re.sub(r"R\$\s*", "", str(texto).strip())
        texto = texto.replace(".", "").replace(",", ".")
        try:
            return float(re.sub(r"[^\d.]", "", texto))
        except ValueError:
            return None

    @staticmethod
    def parse_float(texto):
        if not texto:
            return None
        texto = re.sub(r"[^\d,.]", "", str(texto)).replace(",", ".")
        partes = texto.split(".")
        if len(partes) > 2:
            texto = "".join(partes[:-1]) + "." + partes[-1]
        try:
            return float(texto)
        except ValueError:
            return None

    @staticmethod
    def parse_int(texto):
        if not texto:
            return None
        m = re.search(r"\d+", str(texto))
        return int(m.group()) if m else None

    @staticmethod
    def parse_quartos(texto):
        if not texto:
            return None
        t = unicodedata.normalize("NFD", str(texto)).encode("ascii", "ignore").decode()
        t = re.sub(r"\s+", " ", t).lower().strip()
        m = re.search(r"(\d+)\s*quarto", t)
        if m:
            return int(m.group(1))
        m = re.match(r"^(\d+)$", t)
        if m:
            return int(m.group(1))
        return None

    def extract_image_url(self, soup):
        for selector, attr in (
            ('meta[property="og:image"]', "content"),
            ('a[data-fancybox="galeria"][href]', "href"),
            ("a[href*='/wp-content/uploads/']", "href"),
            ("img[src]", "src"),
        ):
            node = soup.select_one(selector)
            if not node:
                continue
            value = node.get(attr)
            if value:
                return self.absolutize_url(self.BASE_URL, value)
        return None

    def fetch_page(self, url, tentativas=3):
        for i in range(tentativas):
            try:
                r = self.session.get(url, timeout=30)
                r.raise_for_status()
                return r.content
            except Exception as e:
                print(
                    f"[PLAZA_V2] Tentativa {i + 1}/{tentativas} falhou para {url}: {e}"
                )
                time.sleep(2)
        return None

    def coletar_urls_listagem(self):
        urls = []
        pagina = 1

        while True:
            url_pag = (
                self.URL_LIST_TEMPLATE
                if pagina == 1
                else self.URL_LIST_PAGE.format(n=pagina)
            )
            print(f"[PLAZA_V2] Buscando pagina {pagina}: {url_pag}")

            html = self.fetch_page(url_pag)
            if not html:
                print(
                    f"[PLAZA_V2] Falha ao buscar pagina {pagina}. Encerrando paginacao."
                )
                break

            soup = self._make_soup(html)
            cards = soup.select("a.link-imovel[href]")
            if not cards:
                print(f"[PLAZA_V2] Nenhum card encontrado na pagina {pagina}.")
                break

            for card in cards:
                href = card.get("href", "").strip()
                if not href:
                    continue
                if href.startswith("/"):
                    href = self.BASE_URL + href
                m = re.search(r"/imovel/(\d+)/", href)
                codigo = m.group(1) if m else None
                if codigo:
                    urls.append((codigo, href))

            proxima = soup.select_one('a.page-numbers[rel="next"]')
            if proxima:
                pagina += 1
                time.sleep(1)
            else:
                break

        vistos = set()
        unicos = []
        for codigo, href in urls:
            if codigo in vistos:
                continue
            vistos.add(codigo)
            unicos.append((codigo, href))

        return unicos

    def extrair_detalhe(self, html, codigo, url):
        soup = self._make_soup(html)

        dados = {
            "id": str(codigo),
            "preco": None,
            "preco_m2": None,
            "bairro": None,
            "endereco": None,
            "imagem_url": self.extract_image_url(soup),
            "cidade": "Chapecó",
            "tipo_imovel": None,
            "area_total": None,
            "area_privada": None,
            "quartos": None,
            "banheiros": None,
            "vagas": None,
            "date_registration": None,
        }

        li_tipo = soup.select_one("li.tipo div + div strong")
        if li_tipo:
            dados["tipo_imovel"] = li_tipo.get_text(strip=True)
        else:
            titulo = soup.title.get_text(strip=True) if soup.title else ""
            m = re.match(r"^([A-Za-zÀ-ÿ\s]+?)\s+de\s+", titulo)
            if m:
                dados["tipo_imovel"] = m.group(1).strip()

        vtotal = soup.select_one(".vtotalvenda")
        if vtotal:
            dados["preco"] = self.parse_preco(vtotal.get_text(strip=True))

        for li_area in soup.select("li.area"):
            div = li_area.select_one("div + div")
            if not div:
                continue
            label_raw = " ".join(div.stripped_strings)
            strong = li_area.select_one("strong")
            valor = self.parse_float(strong.get_text(strip=True)) if strong else None
            if valor is None:
                continue

            label_n = self.normalize_str(label_raw.split("|")[0])
            if "privativa" in label_n or "privada" in label_n:
                dados["area_privada"] = valor
            elif "lote" in label_n or "total" in label_n or "terreno" in label_n:
                dados["area_total"] = valor
            else:
                if dados["area_privada"] is None:
                    dados["area_privada"] = valor
                elif dados["area_total"] is None:
                    dados["area_total"] = valor

        if dados["area_total"] is None and dados["area_privada"] is not None:
            dados["area_total"] = dados["area_privada"]
        elif dados["area_privada"] is None and dados["area_total"] is not None:
            dados["area_privada"] = dados["area_total"]

        li_quarto = soup.select_one("li.quarto strong")
        if li_quarto:
            dados["quartos"] = self.parse_quartos(li_quarto.get_text(strip=True))

        li_banh = soup.select_one("li.banheiro strong")
        if li_banh:
            dados["banheiros"] = self.parse_int(li_banh.get_text(strip=True))

        li_vaga = soup.select_one("li.vaga strong")
        if li_vaga:
            dados["vagas"] = self.parse_int(li_vaga.get_text(strip=True))

        desc_hidden = soup.select_one("p.visually-hidden")
        if desc_hidden:
            m = re.search(
                r"bairro\s+(.+?)\s+em\s", desc_hidden.get_text(), re.IGNORECASE
            )
            if m:
                dados["bairro"] = m.group(1).strip()

        if not dados["bairro"]:
            m = re.search(r"/imovel/\d+/[^/]+-chapeco", url)
            if m:
                slug = m.group().split("/")[-1]
                slug = re.sub(r"-chapeco$", "", slug)
                slug = re.sub(r"^[a-z-]+-venda(-\d+-quartos)?-", "", slug)
                dados["bairro"] = slug.replace("-", " ").title()

        area_ref = dados["area_privada"] or dados["area_total"]
        if dados["preco"] and area_ref:
            dados["preco_m2"] = dados["preco"] / area_ref

        return dados

    def run(self):
        print("[PLAZA_V2] Iniciando scraping")
        lista = self.coletar_urls_listagem()

        if not lista:
            print("[PLAZA_V2] Nenhum imovel encontrado na listagem.")
            return

        total = len(lista)
        print(f"[PLAZA_V2] Total de imoveis encontrados: {total}")

        tipos_ignorados_norm = {self.normalize_str(t) for t in self.TIPOS_IGNORADOS}
        total_salvos = 0
        total_ignorados = 0

        for idx, (codigo, url) in enumerate(lista, start=1):
            try:
                print(f"[PLAZA_V2] ({idx}/{total}) Coletando codigo={codigo} URL={url}")
                html = self.fetch_page(url)
                if not html:
                    print(
                        f"[PLAZA_V2] ({idx}/{total}) Falha ao buscar detalhe codigo={codigo}"
                    )
                    time.sleep(1)
                    continue

                dados = self.extrair_detalhe(html, codigo, url)

                tipo_n = self.normalize_str(dados.get("tipo_imovel") or "")
                if any(ign in tipo_n for ign in tipos_ignorados_norm):
                    print(
                        f"[PLAZA_V2] ({idx}/{total}) Ignorado codigo={codigo} tipo={dados.get('tipo_imovel')}"
                    )
                    total_ignorados += 1
                    time.sleep(1)
                    continue

                self.upsert_imovel(dados)
                total_salvos += 1
                print(
                    f"[PLAZA_V2] ({idx}/{total}) Salvo ID={dados.get('id')} tipo={dados.get('tipo_imovel')} "
                    f"bairro={dados.get('bairro')} preco={dados.get('preco')}"
                )
                time.sleep(1)
            except Exception as e:
                print(
                    f"[PLAZA_V2] ({idx}/{total}) Erro ao processar codigo={codigo} URL={url}: {e}"
                )

        print(
            f"[PLAZA_V2] Coleta finalizada. Salvos={total_salvos} Ignorados={total_ignorados} Total={total}"
        )


if __name__ == "__main__":
    scraper = PlazaScraper()
    try:
        scraper.run()
    finally:
        scraper.close()
