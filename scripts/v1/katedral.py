import re
import sqlite3
import unicodedata
from datetime import datetime
from time import sleep

import requests
from bs4 import BeautifulSoup

from db import get_db_path

BASE_URL = "https://katedralimoveis.com.br"
URL_PESQUISA = (
    "https://katedralimoveis.com.br/pesquisa.php"
    "?tipo=0&bairro=0&quartos=0&banheiros=0&garagens=0"
    "&codigo=&precoMin=0&precoMax=1000000&pesquisaAvancada=1"
)
URL_IMOVEL = "https://katedralimoveis.com.br/imovel.php?id={id}"

HEADERS = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "accept-language": "pt-BR,pt;q=0.9,en-US;q=0.8",
    "user-agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

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


def parse_preco(texto):
    """Converte 'R$340 mil' ou 'R$1.200.000' para float em reais."""
    if not texto:
        return None
    texto = texto.strip()
    texto = re.sub(r"R\$\s*", "", texto)
    if "mil" in texto.lower():
        numero = re.sub(r"[^\d,]", "", texto).replace(",", ".")
        try:
            return float(numero) * 1000
        except ValueError:
            return None
    else:
        numero = re.sub(r"[^\d,]", "", texto).replace(",", ".")
        try:
            return float(numero)
        except ValueError:
            return None


def parse_float(texto):
    """Converte '56.18' ou '56,18' para float."""
    if not texto:
        return None
    try:
        return float(str(texto).strip().replace(",", "."))
    except ValueError:
        return None


def parse_int(texto):
    """Converte '2' para int."""
    if not texto:
        return None
    try:
        return int(str(texto).strip())
    except ValueError:
        return None


# Tipos de imóvel que devem ser ignorados
TIPOS_IGNORADOS = {"Barracão", "Barraco", "Sala Comercial"}


def parse_quartos(texto):
    """
    Converte os padrões de quartos/suítes para número inteiro total.
    Agora é mais tolerante a variações de escrita (ex: 'Sute', 'Suíte +1', '2 Suítes +1').

    Estratégia:
      - Normaliza o texto (remove acentos e lower).
      - Detecta se o texto contém uma indicação de 'suíte' (com variação).
      - Extrai todos os inteiros do texto.
      - Regras heurísticas: se contém 'suíte' e houver números com '+', soma-os; se for 'Suíte +N' inferred suites=1 + N.
    """
    if not texto:
        return None

    # Normaliza: remove acentos e converte para minúsculas
    t = (
        unicodedata.normalize("NFD", texto)
        .encode("ascii", "ignore")
        .decode()
        .lower()
        .strip()
    )

    # números presentes no texto
    nums = [int(n) for n in re.findall(r"\d+", t)]

    # detecta variantes de "suíte" como 'suíte','suite','sute', 'suítes' etc
    has_suite = bool(re.search(r"su\w*te", t))

    # Caso: só um número puro -> retorna esse número
    if re.match(r"^\d+$", t):
        return int(t)

    # Se existir indicação de suítes
    if has_suite:
        # Ex: '2 suites +1' -> nums [2,1] -> soma
        if len(nums) >= 2:
            return sum(nums)
        # Ex: 'suite +1' -> nums [1] -> 1 (suite) + 1
        if len(nums) == 1:
            if "+" in t and not t.strip().startswith(str(nums[0])):
                # texto como 'suíte +1' (o primeiro número é o que vem após '+')
                return 1 + nums[0]
            else:
                # texto como '2 suítes' -> retorna 2
                return nums[0]
        # Sem números, mas palavra 'suíte' presente -> 1
        return 1

    # Se não for suíte, mas houver números, retorna o primeiro
    if nums:
        return nums[0]

    return None


def normalize_str(s):
    """Remove acentos e converte para minúsculas para comparação robusta."""
    return (
        unicodedata.normalize("NFD", s)
        .encode("ascii", "ignore")
        .decode()
        .lower()
        .strip()
    )


def extrair_caracteristicas(soup):
    """
    Extrai dicionário label→valor da seção de Características.
    Indexa por chave normalizada (sem acentos, minúscula) para ser
    resiliente a problemas de encoding do HTML.
    """
    carac = {}
    for row in soup.select(
        "div.flex.flex-justify-space-between.flex-align-center.pv-2.border-bottom"
    ):
        spans = row.find_all("span", class_="font-alt")
        if len(spans) >= 2:
            label = spans[-2].get_text(strip=True)
            valor = spans[-1].get_text(strip=True)
            if label:
                carac[normalize_str(label)] = valor
    return carac


def get_carac(carac, label):
    """Busca no dicionário de características ignorando acentos."""
    return carac.get(normalize_str(label))


def _fallback_summary_value(soup, possible_titles):
    """Procura no bloco de resumo (ícones) por um div com title correspondente e retorna o texto do último span.
    possible_titles: lista de strings (ex: ['Quartos','BWC'])
    """
    for t in possible_titles:
        # procura por atributo title exatamente ou contendo a string
        el = soup.find(
            lambda tag: tag.name == "div"
            and tag.get("title")
            and t.lower() in tag.get("title").lower()
        )
        if el:
            spans = el.find_all("span")
            if spans:
                return spans[-1].get_text(strip=True)
    return None


print("KATEDRAL")
# ── PASSO 1: busca página de pesquisa e coleta todos os IDs ──────────────────
print("Buscando lista de imóveis...")
r = requests.get(URL_PESQUISA, headers=HEADERS, timeout=30)
r.raise_for_status()
r.encoding = r.apparent_encoding

soup_lista = BeautifulSoup(r.text, "html.parser")

# Links do tipo imovel.php?id=XXXX
links = soup_lista.find_all("a", href=re.compile(r"imovel\.php\?id=\d+"))
ids = list(
    dict.fromkeys(  # remove duplicatas mantendo ordem
        re.search(r"id=(\d+)", a["href"]).group(1) for a in links
    )
)

print(f"{len(ids)} imóveis encontrados na listagem\n")

# ── PASSO 2: consulta detalhe de cada imóvel ─────────────────────────────────
total = 0

for imovel_id in ids:
    url_det = URL_IMOVEL.format(id=imovel_id)

    try:
        r_det = requests.get(url_det, headers=HEADERS, timeout=30)
        r_det.raise_for_status()
        r_det.encoding = r_det.apparent_encoding
    except Exception as e:
        print(f"  [K-{imovel_id}] Erro ao buscar detalhe: {e}")
        sleep(1)
        continue

    soup = BeautifulSoup(r_det.text, "html.parser")
    carac = extrair_caracteristicas(soup)

    # ── Tipo: span.material-icons[title] na área de detalhes ──────────────
    tipo_span = soup.select_one("div.flex.flex-align-center span.material-icons[title]")
    tipo = tipo_span["title"].strip() if tipo_span else None

    # Ignora Barracão e Sala Comercial
    if tipo in TIPOS_IGNORADOS:
        print(f"  [K-{imovel_id}] Ignorado (tipo: {tipo})")
        sleep(1)
        continue

    # ── Bairro: <h2> sem classe (primeiro h2 do conteúdo principal) ────────
    bairro_h2 = soup.select_one("div.container div.col-12 h2")
    bairro = bairro_h2.get_text(strip=True) if bairro_h2 else None

    # ── Preço: <h2 class="text-primary"> ─────────────────────────────────
    preco_h2 = soup.select_one("h2.text-primary")
    preco = parse_preco(preco_h2.get_text(strip=True)) if preco_h2 else None

    # ── Campos das Características ────────────────────────────────────────
    # tenta primeiro pela seção de características
    quartos_val = get_carac(carac, "Dormitórios")
    banheiros_val = get_carac(carac, "Banheiros")
    vagas_val = get_carac(carac, "Garagens")

    # fallbacks - alguns templates mostram as informações no cabeçalho/resumo
    if not quartos_val:
        quartos_val = _fallback_summary_value(soup, ["Quartos"])
    if not banheiros_val:
        banheiros_val = _fallback_summary_value(soup, ["BWC", "Banheiros"])
    if not vagas_val:
        vagas_val = _fallback_summary_value(
            soup, ["Vagas de Garagem", "Garagens", "Vagas"]
        )

    quartos = parse_quartos(quartos_val)
    banheiros = parse_int(banheiros_val)
    vagas = parse_int(vagas_val)
    area_privada = parse_float(get_carac(carac, "Área Privativa"))
    area_total = parse_float(get_carac(carac, "Área Total"))

    # Fallback entre área privativa e total
    if area_total and not area_privada:
        area_privada = area_total
    elif area_privada and not area_total:
        area_total = area_privada

    area_ref = area_privada or area_total

    preco_m2 = None
    if preco and area_ref:
        preco_m2 = preco / area_ref

    cursor.execute(
        """
    INSERT OR REPLACE INTO imoveis
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            "K-" + imovel_id,
            preco,
            preco_m2,
            bairro,
            "Chapecó",
            tipo,
            area_total,
            area_privada,
            quartos,
            banheiros,
            vagas,
            None,
            datetime.now().isoformat(),
        ),
    )

    total += 1
    print(f"  [K-{imovel_id}] {tipo} - {bairro} - R$ {preco}")
    sleep(1)

conn.commit()
conn.close()
print(f"\nScraping finalizado: {total} imóveis inseridos")
