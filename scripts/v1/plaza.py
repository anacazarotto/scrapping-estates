import re
import sqlite3
import unicodedata
from datetime import datetime
from time import sleep

import requests
from bs4 import BeautifulSoup

from db import get_db_path

# ── URLs ─────────────────────────────────────────────────────────────────────
BASE_URL = "https://plazachapeco.com.br"
URL_LIST_TEMPLATE = "https://plazachapeco.com.br/comprar/imoveis/"  # pág 1
URL_LIST_PAGE = "https://plazachapeco.com.br/comprar/imoveis/pag-{n}/"  # pág N≥2

HEADERS = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "accept-language": "pt-BR,pt;q=0.9,en-US;q=0.8",
    "user-agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

# ── Banco de dados ────────────────────────────────────────────────────────────
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

# ── Tipos ignorados ───────────────────────────────────────────────────────────
TIPOS_IGNORADOS = {
    "barracão",
    "barracao",
    "sala comercial",
    "sala/conjunto",
    "sala-comercial",
    "barracao",
}

# ── Helpers ───────────────────────────────────────────────────────────────────


def normalize_str(s):
    """Remove acentos e retorna minúsculas."""
    return (
        unicodedata.normalize("NFD", s)
        .encode("ascii", "ignore")
        .decode()
        .lower()
        .strip()
    )


def parse_preco(texto):
    """Converte 'R$ 2.950.000,00' ou 'R$382.978' para float."""
    if not texto:
        return None
    texto = texto.strip()
    texto = re.sub(r"R\$\s*", "", texto)  # remove "R$"
    texto = texto.replace(".", "").replace(",", ".")  # 2.950.000,00 → 2950000.00
    try:
        return float(re.sub(r"[^\d.]", "", texto))
    except ValueError:
        return None


def parse_float(texto):
    """Converte '320,00m²' ou '320.00' para float."""
    if not texto:
        return None
    texto = re.sub(r"[^\d,.]", "", str(texto)).replace(",", ".")
    # Remove pontos de milhar (ex: '1.021' → '1021')
    partes = texto.split(".")
    if len(partes) > 2:
        texto = "".join(partes[:-1]) + "." + partes[-1]
    try:
        return float(texto)
    except ValueError:
        return None


def parse_int(texto):
    if not texto:
        return None
    m = re.search(r"\d+", str(texto))
    return int(m.group()) if m else None


def parse_quartos(texto):
    """
    Extrai número total de quartos do campo de detalhes do Plaza.
    Exemplos:
      "3 quartos sendo 3 suítes"  → 3
      "2 quartos"                 → 2
      "2"                         → 2
    """
    if not texto:
        return None
    t = unicodedata.normalize("NFD", texto).encode("ascii", "ignore").decode()
    t = re.sub(r"\s+", " ", t).lower().strip()
    # "N quartos ..."
    m = re.search(r"(\d+)\s*quarto", t)
    if m:
        return int(m.group(1))
    # número puro
    m = re.match(r"^(\d+)$", t)
    if m:
        return int(m.group(1))
    return None


def fetch_page(url, tentativas=3):
    """GET com retry."""
    for i in range(tentativas):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            r.encoding = r.apparent_encoding or "utf-8"
            return r.text
        except Exception as e:
            print(f"  Tentativa {i+1}/{tentativas} falhou para {url}: {e}")
            sleep(2)
    return None


# ── PASSO 1: coleta URLs de todos os imóveis, percorrendo as páginas ──────────


def coletar_urls_listagem():
    """Percorre todas as páginas e retorna lista de (codigo, url_detalhe)."""
    urls = []
    pagina = 1

    while True:
        url_pag = URL_LIST_TEMPLATE if pagina == 1 else URL_LIST_PAGE.format(n=pagina)
        print(f"  Buscando página {pagina}: {url_pag}")
        html = fetch_page(url_pag)
        if not html:
            print("  Falha ao buscar página — encerrando paginação.")
            break

        soup = BeautifulSoup(html, "html.parser")

        # Cards de imóveis: <a class="link-imovel ...">
        cards = soup.select("a.link-imovel[href]")
        if not cards:
            print("  Nenhum card encontrado — fim das páginas.")
            break

        for card in cards:
            href = card["href"].strip()
            # Garante URL absoluta
            if href.startswith("/"):
                href = BASE_URL + href
            # Extrai código numérico da URL: /imovel/17818/...
            m = re.search(r"/imovel/(\d+)/", href)
            codigo = m.group(1) if m else None
            if codigo:
                urls.append((codigo, href))

        # Verifica se existe próxima página
        proxima = soup.select_one('a.page-numbers[rel="next"]')
        if proxima:
            pagina += 1
            sleep(1)
        else:
            break

    # Remove duplicatas mantendo ordem
    vistos = set()
    unicos = []
    for item in urls:
        if item[0] not in vistos:
            vistos.add(item[0])
            unicos.append(item)
    return unicos


# ── PASSO 2: extrai dados do detalhe de cada imóvel ──────────────────────────


def extrair_detalhe(html, codigo, url):
    """Parseia a página de detalhe e retorna dict com os campos."""
    soup = BeautifulSoup(html, "html.parser")

    dados = {
        "codigo": "PL-" + codigo,
        "preco": None,
        "bairro": None,
        "cidade": "Chapecó",
        "tipo_imovel": None,
        "area_total": None,
        "area_privada": None,
        "quartos": None,
        "banheiros": None,
        "vagas": None,
    }

    # ── Tipo ─────────────────────────────────────────────────────────────────
    li_tipo = soup.select_one("li.tipo div + div strong")
    if li_tipo:
        dados["tipo_imovel"] = li_tipo.get_text(strip=True)
    else:
        # Fallback: extrai do título da página
        titulo = soup.title.get_text(strip=True) if soup.title else ""
        m = re.match(r"^([A-Za-zÀ-ÿ\s]+?)\s+de\s+", titulo)
        if m:
            dados["tipo_imovel"] = m.group(1).strip()

    # ── Preço ─────────────────────────────────────────────────────────────────
    # <div class="vtotalvenda">R$ 2.950.000,00</div>
    vtotal = soup.select_one(".vtotalvenda")
    if vtotal:
        dados["preco"] = parse_preco(vtotal.get_text(strip=True))

    # ── Área privativa e do lote ──────────────────────────────────────────────
    # <li class="area"><div>Área privativa<br><strong>320,00m²</strong></div></li>
    for li_area in soup.select("li.area"):
        div = li_area.select_one("div + div")
        if not div:
            continue
        label_raw = div.get_text(separator="|", strip=True)
        strong = li_area.select_one("strong")
        valor = parse_float(strong.get_text(strip=True)) if strong else None
        if valor is None:
            continue
        label_n = normalize_str(label_raw.split("|")[0])
        if "privativa" in label_n or "privada" in label_n:
            dados["area_privada"] = valor
        elif "lote" in label_n or "total" in label_n or "terreno" in label_n:
            dados["area_total"] = valor
        else:
            # Genérico: preenche o que estiver vazio
            if dados["area_privada"] is None:
                dados["area_privada"] = valor
            elif dados["area_total"] is None:
                dados["area_total"] = valor

    # Fallback entre as duas áreas
    if dados["area_total"] is None and dados["area_privada"]:
        dados["area_total"] = dados["area_privada"]
    elif dados["area_privada"] is None and dados["area_total"]:
        dados["area_privada"] = dados["area_total"]

    # ── Quartos ───────────────────────────────────────────────────────────────
    # <li class="quarto"><div><strong>3 quartos sendo 3 suítes</strong></div></li>
    li_quarto = soup.select_one("li.quarto strong")
    if li_quarto:
        dados["quartos"] = parse_quartos(li_quarto.get_text(strip=True))

    # ── Banheiros ────────────────────────────────────────────────────────────
    # <li class="banheiro"><div><strong>4 banheiros</strong></div></li>
    li_banh = soup.select_one("li.banheiro strong")
    if li_banh:
        dados["banheiros"] = parse_int(li_banh.get_text(strip=True))

    # ── Vagas ────────────────────────────────────────────────────────────────
    # <li class="vaga"><div><strong>4 vagas</strong></div></li>
    li_vaga = soup.select_one("li.vaga strong")
    if li_vaga:
        dados["vagas"] = parse_int(li_vaga.get_text(strip=True))

    # ── Bairro ────────────────────────────────────────────────────────────────
    # Extrai do data-attribute do bloco de similares:
    # <div ... data-codigo="17818"> ou da URL /imovel/17818/casa-venda-N-quartos-BAIRRO-chapeco/
    # sim = soup.select_one("[data-codigo]")
    # Tenta extrair bairro do atributo data-* do bloco de similares
    # (não há campo direto — usamos a descrição ou endereço no h3)
    # Fallback: pega do h3.enderecoimovel (não existe no detalhe, existe na lista)
    # No detalhe, o bairro aparece na descrição "localizado no bairro X em"
    desc_hidden = soup.select_one("p.visually-hidden")
    if desc_hidden:
        m = re.search(r"bairro\s+(.+?)\s+em\s", desc_hidden.get_text(), re.IGNORECASE)
        if m:
            dados["bairro"] = m.group(1).strip()

    if not dados["bairro"]:
        # Fallback: tira da URL /imovel/ID/TIPO-venda-N-quartos-BAIRRO-chapeco/
        m = re.search(r"/imovel/\d+/[^/]+-chapeco", url)
        if m:
            slug = m.group().split("/")[
                -1
            ]  # ex: casa-venda-3-quartos-jardim-europa-chapeco
            slug = re.sub(r"-chapeco$", "", slug)  # remove -chapeco
            # Remove prefixo: "tipo-venda-N-quartos-" ou "tipo-venda-"
            slug = re.sub(r"^[a-z-]+-venda(-\d+-quartos)?-", "", slug)
            dados["bairro"] = slug.replace("-", " ").title()

    return dados


# ── MAIN ─────────────────────────────────────────────────────────────────────

print("=" * 60)
print("PLAZA Imóveis — scraping iniciado")
print("=" * 60)

print("\n[1/2] Coletando URLs da listagem...")
lista = coletar_urls_listagem()
print(f"  Total de imóveis encontrados: {len(lista)}\n")

print("[2/2] Buscando detalhes e inserindo no banco...")
total_inseridos = 0
total_ignorados = 0

for codigo, url in lista:
    html = fetch_page(url)
    if not html:
        print(f"  [PL-{codigo}] Falha ao buscar detalhe — pulando.")
        sleep(1)
        continue

    dados = extrair_detalhe(html, codigo, url)

    # Ignora tipos indesejados
    tipo_n = normalize_str(dados["tipo_imovel"] or "")
    if any(ign in tipo_n for ign in TIPOS_IGNORADOS):
        print(f"  [PL-{codigo}] Ignorado (tipo: {dados['tipo_imovel']})")
        total_ignorados += 1
        sleep(1)
        continue

    # Preço/m²
    area_ref = dados["area_privada"] or dados["area_total"]
    preco_m2 = (dados["preco"] / area_ref) if (dados["preco"] and area_ref) else None

    cursor.execute(
        """
    INSERT OR REPLACE INTO imoveis
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            dados["codigo"],
            dados["preco"],
            preco_m2,
            dados["bairro"],
            dados["cidade"],
            dados["tipo_imovel"],
            dados["area_total"],
            dados["area_privada"],
            dados["quartos"],
            dados["banheiros"],
            dados["vagas"],
            None,
            datetime.now().isoformat(),
        ),
    )
    conn.commit()

    total_inseridos += 1
    print(
        f"  [PL-{codigo}] {dados['tipo_imovel']} | {dados['bairro']} | "
        f"R$ {dados['preco']} | {dados['quartos']}q {dados['banheiros']}b "
        f"{dados['vagas']}v | {area_ref}m²"
    )
    sleep(1)

conn.close()

print(f"\n{'='*60}")
print("Scraping finalizado:")
print(f"  Inseridos : {total_inseridos}")
print(f"  Ignorados : {total_ignorados}")
print(f"{'='*60}")
