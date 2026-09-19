import re
import time
import unicodedata

import requests
from bs4 import BeautifulSoup, SoupStrainer

from base_scapper import BaseScraper


class KatedralScraper(BaseScraper):
    PREFIX = "K-"
    BASE_URL = "https://katedralimoveis.com.br"
    URL_PESQUISA = (
        "https://katedralimoveis.com.br/pesquisa.php"
        "?tipo=0&bairro=0&quartos=0&banheiros=0&garagens=0"
        "&codigo=&precoMin=0&precoMax=1000000&pesquisaAvancada=1"
    )
    URL_IMOVEL = "https://katedralimoveis.com.br/imovel.php?id={id}"
    TIPOS_IGNORADOS = {"Barracão", "Barraco", "Sala Comercial"}
    HEADERS = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "accept-language": "pt-BR,pt;q=0.9,en-US;q=0.8",
        "user-agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }

    def __init__(self):
        super().__init__()
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)

    @staticmethod
    def _make_soup(html, *, parse_only=None):
        try:
            return BeautifulSoup(html, "lxml", parse_only=parse_only)
        except Exception:
            return BeautifulSoup(html, "html.parser", parse_only=parse_only)

    @staticmethod
    def parse_preco(texto):
        if not texto:
            return None

        texto = re.sub(r"R\$\s*", "", texto.strip())
        if "mil" in texto.lower():
            numero = re.sub(r"[^\d,]", "", texto).replace(",", ".")
            try:
                return float(numero) * 1000
            except ValueError:
                return None

        numero = re.sub(r"[^\d,]", "", texto).replace(",", ".")
        try:
            return float(numero)
        except ValueError:
            return None

    @staticmethod
    def parse_float(texto):
        if not texto:
            return None
        try:
            return float(str(texto).strip().replace(",", "."))
        except ValueError:
            return None

    @staticmethod
    def parse_int(texto):
        if not texto:
            return None
        try:
            return int(str(texto).strip())
        except ValueError:
            return None

    @staticmethod
    def parse_quartos(texto):
        if not texto:
            return None

        t = (
            unicodedata.normalize("NFD", texto)
            .encode("ascii", "ignore")
            .decode()
            .lower()
            .strip()
        )
        nums = [int(n) for n in re.findall(r"\d+", t)]
        has_suite = bool(re.search(r"su\w*te", t))

        if re.match(r"^\d+$", t):
            return int(t)

        if has_suite:
            if len(nums) >= 2:
                return sum(nums)
            if len(nums) == 1:
                if "+" in t and not t.strip().startswith(str(nums[0])):
                    return 1 + nums[0]
                return nums[0]
            return 1

        if nums:
            return nums[0]

        return None

    @staticmethod
    def normalize_str(s):
        if s is None:
            return ""
        return (
            unicodedata.normalize("NFD", s)
            .encode("ascii", "ignore")
            .decode()
            .lower()
            .strip()
        )

    def extrair_caracteristicas(self, soup):
        carac = {}
        rows = soup.select(
            "div.flex.flex-justify-space-between.flex-align-center.pv-2.border-bottom"
        )
        for row in rows:
            spans = row.find_all("span", class_="font-alt")
            if len(spans) < 2:
                continue

            label = spans[-2].get_text(strip=True)
            valor = spans[-1].get_text(strip=True)
            if label:
                carac[self.normalize_str(label)] = valor

        return carac

    def get_carac(self, carac, label):
        return carac.get(self.normalize_str(label))

    @staticmethod
    def fallback_summary_value(soup, possible_titles):
        for title in possible_titles:
            el = soup.find(
                lambda tag: tag.name == "div"
                and tag.get("title")
                and title.lower() in tag.get("title", "").lower()
            )
            if not el:
                continue

            spans = el.find_all("span")
            if spans:
                return spans[-1].get_text(strip=True)

        return None

    def extract_image_url(self, soup):
        for selector, attr in (
            ('meta[property="og:image"]', "content"),
            ("a[href*='/cdn/'][href]", "href"),
            ("img[src]", "src"),
        ):
            node = soup.select_one(selector)
            if not node:
                continue
            value = node.get(attr)
            if value:
                return self.absolutize_url(self.BASE_URL, value)
        return None

    def pegar_links(self):
        print(f"[KATEDRAL_V2] Buscando lista de imoveis: {self.URL_PESQUISA}")

        r = self.session.get(self.URL_PESQUISA, timeout=30)
        r.raise_for_status()

        only_links = SoupStrainer("a")
        soup_lista = self._make_soup(r.content, parse_only=only_links)
        links = soup_lista.find_all("a", href=re.compile(r"imovel\.php\?id=\d+"))

        ids = []
        vistos = set()
        for a in links:
            href = str(a.get("href", ""))
            m = re.search(r"id=(\d+)", href)
            if not m:
                continue
            imovel_id = m.group(1)
            if imovel_id in vistos:
                continue
            vistos.add(imovel_id)
            ids.append(imovel_id)

        print(len(ids), "imoveis encontrados na listagem")
        return ids

    def scrape_imovel(self, imovel_id):
        url_det = self.URL_IMOVEL.format(id=imovel_id)

        r_det = self.session.get(url_det, timeout=30)
        r_det.raise_for_status()

        soup = self._make_soup(r_det.content)
        carac = self.extrair_caracteristicas(soup)

        tipo_span = soup.select_one(
            "div.flex.flex-align-center span.material-icons[title]"
        )
        tipo = tipo_span.get("title", "").strip() if tipo_span else None

        bairro_h2 = soup.select_one("div.container div.col-12 h2")
        bairro = bairro_h2.get_text(strip=True) if bairro_h2 else None

        preco_h2 = soup.select_one("h2.text-primary")
        preco = self.parse_preco(preco_h2.get_text(strip=True)) if preco_h2 else None

        quartos_val = self.get_carac(carac, "Dormitórios")
        banheiros_val = self.get_carac(carac, "Banheiros")
        vagas_val = self.get_carac(carac, "Garagens")

        if not quartos_val:
            quartos_val = self.fallback_summary_value(soup, ["Quartos"])
        if not banheiros_val:
            banheiros_val = self.fallback_summary_value(soup, ["BWC", "Banheiros"])
        if not vagas_val:
            vagas_val = self.fallback_summary_value(
                soup, ["Vagas de Garagem", "Garagens", "Vagas"]
            )

        quartos = self.parse_quartos(quartos_val)
        banheiros = self.parse_int(banheiros_val)
        vagas = self.parse_int(vagas_val)

        area_privada = self.parse_float(self.get_carac(carac, "Área Privativa"))
        area_total = self.parse_float(self.get_carac(carac, "Área Total"))

        if area_total and not area_privada:
            area_privada = area_total
        elif area_privada and not area_total:
            area_total = area_privada

        area_ref = area_privada or area_total
        preco_m2 = (preco / area_ref) if (preco and area_ref) else None

        return {
            "id": str(imovel_id),
            "preco": preco,
            "preco_m2": preco_m2,
            "bairro": bairro,
            "endereco": None,
            "imagem_url": self.extract_image_url(soup),
            "cidade": "Chapeco",
            "tipo_imovel": tipo,
            "area_total": area_total,
            "area_privada": area_privada,
            "quartos": quartos,
            "banheiros": banheiros,
            "vagas": vagas,
            "date_registration": None,
        }

    def run(self):
        print("[KATEDRAL_V2] Iniciando coleta")
        ids = self.pegar_links()

        if not ids:
            print("[KATEDRAL_V2] Nenhum imovel encontrado na listagem")
            return

        total_ids = len(ids)
        print(f"[KATEDRAL_V2] {total_ids} imoveis encontrados na listagem")

        for idx, imovel_id in enumerate(ids, start=1):
            url_det = self.URL_IMOVEL.format(id=imovel_id)
            try:
                print(
                    f"[KATEDRAL_V2] ({idx}/{total_ids}) Coletando ID={imovel_id} URL={url_det}"
                )
                dados = self.scrape_imovel(imovel_id)

                if dados.get("tipo_imovel") in self.TIPOS_IGNORADOS:
                    print(
                        f"[KATEDRAL_V2] ({idx}/{total_ids}) Ignorado ID={imovel_id} tipo={dados.get('tipo_imovel')}"
                    )
                    time.sleep(1)
                    continue

                self.upsert_imovel(dados)
                print(
                    f"[KATEDRAL_V2] ({idx}/{total_ids}) Salvo ID={dados.get('id')} tipo={dados.get('tipo_imovel')} bairro={dados.get('bairro')}"
                )
                time.sleep(1)
            except Exception as e:
                print(
                    f"[KATEDRAL_V2] ({idx}/{total_ids}) Erro ao processar ID={imovel_id} URL={url_det}: {e}"
                )

        print("[KATEDRAL_V2] Coleta finalizada")


if __name__ == "__main__":
    scraper = KatedralScraper()
    try:
        scraper.run()
    finally:
        scraper.close()
