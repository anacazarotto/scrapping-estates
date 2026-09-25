import json
import re
import time
import unicodedata

import requests

from base_scapper import BaseScraper
from db.config import require_env_value


class MarkizeScraper(BaseScraper):
    PREFIX = "M-"
    URL_LISTA = "https://markize.simob.com.br/v2/integracaoApi/imovel/filtro/categoria/caracteristicas"
    URL_DETALHE = (
        "https://markize.simob.com.br/v2/integracaoApi/detalhes/imovel/{codigo}"
        "?calcularValorAbono=false&considerarPrevisaoSaida=false"
        "&somenteTelefonePublicarSite=true&validadeOpcaoVenda=false"
    )
    HEADERS = {
        "accept": "application/json, text/plain, */*",
        "accept-language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
        "from": "site",
        "origin": "https://www.markize.com.br",
        "referer": "https://www.markize.com.br/",
    }

    MAX_RESULTS = 50

    ID_QUARTOS = 3
    ID_QUARTOS_FALLBACK = 5
    ID_BANHEIROS = 6
    ID_VAGAS = 4
    ID_VAGAS_2 = 15
    ID_AREA_PRIVADA = 9
    ID_AREA_TOTAL = 27

    TIPOS_IGNORADOS = {"Barracao", "Barraco", "Sala Comercial", "Sala"}

    def __init__(self):
        super().__init__()
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self.session.headers["authorization"] = (
            f"Bearer {require_env_value('MARKIZE_API_TOKEN')}"
        )

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

    def get_carac_valor(self, caracteristicas, id_carac):
        for c in caracteristicas:
            if c.get("idCaracteristica") != id_carac:
                continue
            try:
                val = c.get("valor", 0)
                return float(val) if val not in (None, "", "0") else None
            except (ValueError, TypeError):
                return None
        return None

    def parse_quartos_text(self, val):
        if val is None:
            return None

        if isinstance(val, (int, float)):
            try:
                return int(val)
            except Exception:
                return None

        txt = self.normalize_str(val)
        nums = [int(n) for n in re.findall(r"\d+", txt)]
        has_suite = "su" in txt

        if has_suite:
            if len(nums) >= 2:
                return sum(nums)
            if len(nums) == 1:
                if "+" in txt and not txt.startswith(str(nums[0])):
                    return 1 + nums[0]
                return nums[0]
            return 1

        if nums:
            return nums[0]

        return None

    def build_payload(self, first_result):
        payload_inner = {
            "idsCategorias": [],
            "finalidade": 2,
            "ceps": ["CHAPECO"],
            "idsBairros": [],
            "rangeValue": {"max": "", "min": ""},
            "caracteristicas": [
                {
                    "id": 89,
                    "idTipoCaracteristica": 3,
                    "qtd": 0,
                    "considerarValorExato": False,
                },
                {
                    "id": 103,
                    "idTipoCaracteristica": 3,
                    "qtd": 0,
                    "considerarValorExato": False,
                },
                {
                    "id": 140,
                    "idTipoCaracteristica": 3,
                    "qtd": 0,
                    "considerarValorExato": False,
                },
            ],
            "selectedOptions": {
                "categorias": [],
                "caracteristicas": [
                    {
                        "id": 89,
                        "descricao": "Dormitorio(s)",
                        "tipoCaracteristica": 3,
                        "ordem": 1,
                        "grupo": {"id": 1, "descricao": "Fisicas"},
                        "publicar": 1,
                        "buscaImovel": 1,
                        "finalidade": 3,
                        "value": 0,
                    },
                    {
                        "id": 103,
                        "descricao": "Sendo suite(s)",
                        "tipoCaracteristica": 3,
                        "ordem": 2,
                        "grupo": {"id": 1, "descricao": "Fisicas"},
                        "publicar": 1,
                        "buscaImovel": 0,
                        "finalidade": 3,
                        "value": 0,
                    },
                    {
                        "id": 140,
                        "descricao": "Sendo demi-suite(s)",
                        "tipoCaracteristica": 3,
                        "ordem": 3,
                        "grupo": {"id": 1, "descricao": "Fisicas"},
                        "publicar": 1,
                        "buscaImovel": 1,
                        "finalidade": 3,
                        "value": 0,
                    },
                ],
                "bairros": [],
                "cidades": [{"cidade": "CHAPECO", "uf": "SC"}],
                "finalidade": "Comprar",
                "range": {"maxRange": "", "minRange": ""},
            },
            "offset": {"maxResults": self.MAX_RESULTS, "firstResult": first_result},
            "acuracidade": 100,
            "countResults": False,
            "considerarPrevisaoSaida": False,
            "calcularValorAbono": False,
            "validade_opcao_venda": False,
            "orderBy": [
                {
                    "sort": "valor",
                    "descricao": "Valor",
                    "order": "desc",
                    "active": False,
                    "type": "number",
                },
                {"sort": "metrica", "order": "desc"},
            ],
            "trazerCaracteristicas": 3,
        }
        return {"data": json.dumps(payload_inner, ensure_ascii=False)}

    def pegar_links(self, first_result):
        payload = self.build_payload(first_result)
        r = self.session.post(self.URL_LISTA, data=payload, timeout=30)
        r.raise_for_status()

        data = r.json()
        return data.get("result", [])

    @staticmethod
    def _get_raw_valor_from_carac(caracteristicas, ids):
        for c in caracteristicas:
            if c.get("idCaracteristica") not in ids:
                continue
            return (
                c.get("valor")
                or c.get("valorFormatado")
                or c.get("valorText")
                or c.get("descricao")
            )
        return None

    def scrape_imovel(self, codigo, item_lista):
        r_det = self.session.get(self.URL_DETALHE.format(codigo=codigo), timeout=30)
        r_det.raise_for_status()

        det_data = r_det.json()
        det_list = det_data.get("result", [])
        if not det_list:
            return None

        p = det_list[0]
        caracteristicas = p.get("caracteristicas", [])

        config_venda = p.get("configVenda") or {}
        preco_str = config_venda.get("valor")
        try:
            preco = float(preco_str) if preco_str else None
        except (ValueError, TypeError):
            preco = None

        area_privada = self.get_carac_valor(caracteristicas, self.ID_AREA_PRIVADA)
        area_total = self.get_carac_valor(caracteristicas, self.ID_AREA_TOTAL)

        if area_total and not area_privada:
            area_privada = area_total
        elif area_privada and not area_total:
            area_total = area_privada

        area_ref = area_privada or area_total
        preco_m2 = (preco / area_ref) if (preco and area_ref) else None

        raw_quartos = self._get_raw_valor_from_carac(caracteristicas, {self.ID_QUARTOS})
        if not raw_quartos:
            raw_quartos = self._get_raw_valor_from_carac(
                caracteristicas, {self.ID_QUARTOS_FALLBACK}
            )
        quartos = self.parse_quartos_text(raw_quartos)

        raw_banheiros = self._get_raw_valor_from_carac(
            caracteristicas, {self.ID_BANHEIROS}
        )
        try:
            banheiros = int(raw_banheiros) if raw_banheiros not in (None, "") else None
        except Exception:
            banheiros = None

        raw_vagas = self._get_raw_valor_from_carac(
            caracteristicas, {self.ID_VAGAS, self.ID_VAGAS_2}
        )
        try:
            vagas = int(raw_vagas) if raw_vagas not in (None, "") else None
        except Exception:
            vagas = None

        categoria = p.get("categoria") or {}
        tipo_imovel = categoria.get("descricao") or item_lista.get("descricaoCategoria")
        imagens = p.get("imagens") or item_lista.get("imagens") or []
        imagem_url = None
        if imagens:
            imagem = imagens[0]
            imagem_url = (
                imagem.get("urlCDN")
                or imagem.get("url")
                or imagem.get("url_original")
                or imagem.get("url_small")
            )
            imagem_url = BaseScraper.absolutize_url(self.URL_DETALHE, imagem_url)

        return {
            "id": str(p.get("id", codigo)),
            "preco": preco,
            "preco_m2": preco_m2,
            "bairro": p.get("bairro"),
            "endereco": BaseScraper.format_endereco(
                p.get("endereco") or item_lista.get("endereco"),
                p.get("numero") or item_lista.get("numero"),
                p.get("complemento") or item_lista.get("complemento"),
            ),
            "imagem_url": imagem_url,
            "cidade": p.get("cidade"),
            "tipo_imovel": tipo_imovel,
            "area_total": area_total,
            "area_privada": area_privada,
            "quartos": int(quartos) if quartos else None,
            "banheiros": int(banheiros) if banheiros else None,
            "vagas": int(vagas) if vagas else None,
            "date_registration": p.get("dataPublicacao") or item_lista.get("updatedAt"),
        }

    def run(self):
        print("[MARKIZE_V2] Iniciando coleta")
        first_result = 0

        while True:
            print(f"[MARKIZE_V2] Buscando pagina com firstResult={first_result}")
            results = self.pegar_links(first_result)

            if not results:
                print(
                    f"[MARKIZE_V2] Nenhum resultado em firstResult={first_result}. Encerrando coleta."
                )
                break

            total_results = len(results)
            print(
                f"[MARKIZE_V2] firstResult={first_result}: {total_results} imoveis encontrados."
            )

            for idx, item in enumerate(results, start=1):
                codigo = item.get("codigo")
                if not codigo:
                    print(
                        f"[MARKIZE_V2] ({idx}/{total_results}) Item sem codigo. Ignorado."
                    )
                    continue

                try:
                    print(
                        f"[MARKIZE_V2] ({idx}/{total_results}) Coletando codigo={codigo}"
                    )
                    dados = self.scrape_imovel(codigo, item)

                    if not dados:
                        print(
                            f"[MARKIZE_V2] ({idx}/{total_results}) Detalhe nao encontrado para codigo={codigo}"
                        )
                        time.sleep(1)
                        continue

                    tipo_norm = self.normalize_str(dados.get("tipo_imovel"))
                    tipos_ignorados_norm = {
                        self.normalize_str(t) for t in self.TIPOS_IGNORADOS
                    }
                    if tipo_norm in tipos_ignorados_norm:
                        print(
                            f"[MARKIZE_V2] ({idx}/{total_results}) Ignorado codigo={codigo} tipo={dados.get('tipo_imovel')}"
                        )
                        time.sleep(1)
                        continue

                    self.upsert_imovel(dados)
                    print(
                        f"[MARKIZE_V2] ({idx}/{total_results}) Salvo ID={dados.get('id')} tipo={dados.get('tipo_imovel')} bairro={dados.get('bairro')}"
                    )
                    time.sleep(1)
                except Exception as e:
                    print(
                        f"[MARKIZE_V2] ({idx}/{total_results}) Erro ao processar codigo={codigo}: {e}"
                    )

            if total_results < self.MAX_RESULTS:
                print(
                    "[MARKIZE_V2] Ultima pagina identificada pela quantidade de resultados."
                )
                break

            first_result += self.MAX_RESULTS
            time.sleep(5)

        print("[MARKIZE_V2] Coleta finalizada")


if __name__ == "__main__":
    scraper = MarkizeScraper()
    try:
        scraper.run()
    finally:
        scraper.close()
