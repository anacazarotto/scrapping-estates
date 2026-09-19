import re
import time

import requests

from base_scapper import BaseScraper


class SimScraper(BaseScraper):
    PREFIX = "SI-"
    URL = "https://www.simimoveischapeco.com/retornar-imoveis-disponiveis"
    HEADERS = {
        "accept": "application/json, text/plain, */*",
        "accept-language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
        "content-type": "application/x-www-form-urlencoded",
        "origin": "https://www.simimoveischapeco.com",
        "referer": "https://www.simimoveischapeco.com/",
    }
    PAGE_SIZE = 20

    def __init__(self):
        super().__init__()
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)

    @staticmethod
    def parse_valor(valor_str):
        if not valor_str:
            return None
        try:
            limpo = re.sub(r"[^\d,]", "", valor_str).replace(",", ".")
            return float(limpo) if limpo else None
        except (ValueError, TypeError):
            return None

    @staticmethod
    def parse_area(area_str):
        if not area_str:
            return None
        try:
            return float(str(area_str).replace(",", "."))
        except (ValueError, TypeError):
            return None

    def build_form_data(self, pagina):
        return (
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
            f"&numeropagina={pagina}&numeroregistros={self.PAGE_SIZE}&ordenacao=valordesc"
            "&cidades%5Bcodigo%5D=2&cidades%5Bnome%5D=Chapec%C3%B3&cidades%5Bestado%5D=SC"
            "&cidades%5BnomeUrl%5D=chapeco&cidades%5BestadoUrl%5D=sc"
            "&condominio%5Bcodigo%5D=0&condominio%5Bnome%5D=&condominio%5BnomeUrl%5D=todos-os-condominios"
        )

    def pegar_links(self, pagina):
        form_data = self.build_form_data(pagina)
        r = self.session.post(self.URL, data=form_data, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data.get("lista", [])

    def scrape_imovel(self, p):
        preco = self.parse_valor(p.get("valor"))

        area_privada = self.parse_area(p.get("areainterna"))
        area_total = self.parse_area(p.get("areaprincipal")) or self.parse_area(
            p.get("arealote")
        )

        if area_total and not area_privada:
            area_privada = area_total
        elif area_privada and not area_total:
            area_total = area_privada

        area_ref = area_privada or area_total
        preco_m2 = (preco / area_ref) if (preco and area_ref) else None
        fotos = p.get("fotos") or []
        imagem_url = p.get("urlfotoprincipalp")
        if not imagem_url and fotos:
            imagem_url = fotos[0].get("urlp")

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

        return {
            "id": str(p.get("codigo", "")),
            "preco": preco,
            "preco_m2": preco_m2,
            "bairro": p.get("bairro"),
            "endereco": BaseScraper.format_endereco(
                p.get("endereco"),
                p.get("numero"),
                p.get("complemento"),
            ),
            "imagem_url": imagem_url,
            "cidade": p.get("cidade"),
            "tipo_imovel": p.get("tipo"),
            "area_total": area_total,
            "area_privada": area_privada,
            "quartos": quartos,
            "banheiros": banheiros,
            "vagas": vagas,
            "date_registration": p.get("datahoracadastro"),
        }

    def run(self):
        pagina = 1
        print("[SIM_V2] Iniciando coleta")

        while True:
            print(f"[SIM_V2] Buscando pagina {pagina}")
            try:
                results = self.pegar_links(pagina)
            except Exception as e:
                print(f"[SIM_V2] Erro ao buscar pagina {pagina}: {e}")
                break

            if not results:
                print(
                    f"[SIM_V2] Nenhum resultado na pagina {pagina}. Encerrando coleta."
                )
                break

            total_results = len(results)
            print(f"[SIM_V2] Pagina {pagina}: {total_results} imoveis encontrados.")

            for idx, p in enumerate(results, start=1):
                codigo = p.get("codigo")
                try:
                    print(
                        f"[SIM_V2] ({idx}/{total_results}) Processando codigo={codigo}"
                    )
                    dados = self.scrape_imovel(p)

                    if not dados.get("id"):
                        print(
                            f"[SIM_V2] ({idx}/{total_results}) Item sem codigo. Ignorado."
                        )
                        continue

                    self.upsert_imovel(dados)
                    print(
                        f"[SIM_V2] ({idx}/{total_results}) Salvo ID={dados.get('id')} "
                        f"tipo={dados.get('tipo_imovel')} bairro={dados.get('bairro')}"
                    )
                except Exception as e:
                    print(
                        f"[SIM_V2] ({idx}/{total_results}) Erro ao processar codigo={codigo}: {e}"
                    )

            if total_results < self.PAGE_SIZE:
                print(
                    "[SIM_V2] Ultima pagina identificada pela quantidade de resultados."
                )
                break

            pagina += 1
            time.sleep(5)

        print("[SIM_V2] Coleta finalizada")


if __name__ == "__main__":
    scraper = SimScraper()
    try:
        scraper.run()
    finally:
        scraper.close()
