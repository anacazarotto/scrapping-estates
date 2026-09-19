import argparse
import hashlib
import json
import re
import pickle
import sqlite3
import statistics
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import TransformedTargetRegressor

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - optional dependency
    XGBRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover - optional dependency
    CatBoostRegressor = None

try:
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover - optional dependency
    Image = None
    ImageOps = None


DEFAULT_RAW_DB = Path("imoveis_03_09_2026.db")
DEFAULT_UNIFIED_DB = Path("imoveis_unificado.db")
DEFAULT_NORMALIZED_DB = Path("imoveis_normalizados.db")
DEFAULT_MODEL_PATH = Path("modelos/preco_imovel_modelo.pkl")
DEFAULT_DB_PATTERN = "imoveis_??_??_????.db"
BASELINE_ALLOWED_TYPES = {"Casa", "Apartamento"}
BASELINE_SEGMENT_TYPES = ("Casa", "Apartamento")
BASELINE_RARE_BAIRRO_MIN_COUNT = 10
BASELINE_CLIP_QUANTILES = (0.02, 0.98)
BASELINE_MODEL_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 3,
    "random_state": 42,
}
IMAGE_HASH_SIZE = 8
IMAGE_HASH_MAX_DISTANCE = 8
IMAGE_AREA_TOLERANCE = 0.18
IMAGE_PRICE_TOLERANCE = 0.25


class ProgressBar:
    """Barra de progresso simples para execução em terminal."""

    def __init__(self, total, label, width=30, enabled=True):
        self.total = max(int(total or 0), 0)
        self.label = label
        self.width = width
        self.enabled = enabled and self.total > 0
        self.current = 0
        if self.enabled:
            self.render(0)

    def render(self, current):
        if not self.enabled:
            return
        current = min(max(int(current), 0), self.total)
        ratio = current / self.total if self.total else 1
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        percent = ratio * 100
        print(
            f"\r{self.label}: [{bar}] {percent:6.2f}% ({current}/{self.total})",
            end="",
            flush=True,
        )
        self.current = current

    def advance(self, step=1):
        self.render(self.current + step)

    def close(self):
        if self.enabled:
            if self.current < self.total:
                self.render(self.total)
            print()


PROPERTY_TYPE_CATALOG = [
    {
        "canonical": "Apartamento",
        "description": "Unidades residenciais em edifícios, incluindo duplex.",
        "aliases": [
            "apartamento",
            "apart",
            "apto",
            "apartamento duplex",
            "duplex",
        ],
    },
    {
        "canonical": "Casa",
        "description": "Casas, geminadas e sobrados.",
        "aliases": [
            "casa",
            "casa geminada",
            "geminada",
            "sobrado",
        ],
    },
    {
        "canonical": "Terreno",
        "description": "Terrenos, lotes e áreas de terra/rurais.",
        "aliases": [
            "terreno",
            "lote",
            "lote urbano",
            "terreno urbano",
            "terreno em condomínio",
            "area",
            "área",
            "area de terra",
            "área de terra",
            "area rural",
            "área rural",
            "lote rural",
            "loteamento",
        ],
    },
    {
        "canonical": "Cobertura",
        "description": "Coberturas e penthouses.",
        "aliases": [
            "cobertura",
            "penthouse",
        ],
    },
    {
        "canonical": "Chácara",
        "description": "Chácaras e sítios.",
        "aliases": [
            "chacara",
            "chácara",
            "sitio",
            "sítio",
            "chácara / sítio",
            "chacara / sitio",
        ],
    },
    {
        "canonical": "Prédio",
        "description": "Prédios e edifícios inteiros.",
        "aliases": [
            "predio",
            "prédio",
            "predios",
            "prédios",
            "predio residencial",
            "prédio residencial",
            "predio comercial",
            "prédio comercial",
        ],
    },
    {
        "canonical": "Studio",
        "description": "Studios e apartamentos compactos.",
        "aliases": [
            "studio",
            "studios",
            "apartamento studio",
            "apartamentos studio",
            "flat",
            "loft",
        ],
    },
    {
        "canonical": "Comercial",
        "description": "Imóveis com uso comercial.",
        "aliases": [
            "comercial",
            "sala comercial",
            "sala",
            "sala/conjunto",
            "casa comercial",
        ],
    },
    {
        "canonical": "Lançamento",
        "description": "Imóveis em lançamento ou pré-lançamento.",
        "aliases": [
            "lancamento",
            "lançamento",
            "lancamentos",
            "lançamentos",
        ],
    },
    {
        "canonical": "Outros",
        "description": "Fallback para tipos sem similaridade suficiente.",
        "aliases": [],
    },
]

ALLOWED_PROPERTY_TYPES = [item["canonical"] for item in PROPERTY_TYPE_CATALOG]
FEATURE_COLUMNS = ["area_total", "area_privada", "quartos", "banheiros", "vagas", "bairro", "tipo_imovel"]
NUMERIC_COLUMNS = ["area_total", "area_privada", "quartos", "banheiros", "vagas"]
CATEGORICAL_COLUMNS = ["bairro", "tipo_imovel"]


def normalize_text(value):
    if value is None:
        return ""
    text = unicodedata.normalize("NFD", str(value)).encode("ascii", "ignore").decode()
    return " ".join(text.lower().strip().split())


def safe_float(value):
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_int(value):
    if value in (None, ""):
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def canonical_tipo(value):
    text = normalize_text(value)
    if not text:
        return ""
    if "lanc" in text:
        return "Lançamento"
    if "comercial" in text or text == "sala" or "sala/" in text:
        return "Comercial"
    if "studio" in text or "loft" in text or "flat" in text:
        return "Studio"
    if "cobert" in text or "penthouse" in text:
        return "Cobertura"
    if "apart" in text:
        return "Apartamento"
    if "sobrado" in text or "geminad" in text or "casa" in text:
        return "Casa"
    if "terreno" in text or "lote" in text:
        return "Terreno"
    if (
        "chacara" in text
        or "sitio" in text
        or "area de terra" in text
        or "área de terra" in text
        or "area rural" in text
        or "área rural" in text
        or text == "area"
        or text == "área"
    ):
        return "Chácara"
    if "predio" in text:
        return "Prédio"
    return "Outros"


def canonical_bairro(value):
    text = normalize_text(value)
    return text if text else "sem bairro"


def canonical_endereco(value, street_only=False):
    text = normalize_text(value)
    if not text:
        return ""

    text = re.sub(r"\bcep\b.*$", "", text)
    text = re.sub(r"\bchapeco\b.*$", "", text)
    text = re.sub(r"\bsc\b.*$", "", text)
    text = re.sub(r"\s+", " ", text).strip(" ,-/")

    if street_only:
        text = re.split(r"\s*,\s*", text, maxsplit=1)[0]
        text = re.split(r"\s+-\s+", text, maxsplit=1)[0]

    return text.strip(" ,-/")


def canonical_area(row):
    total = safe_float(row["area_total"])
    privada = safe_float(row["area_privada"])
    area = max(total, privada)
    return round(area, 2)


def list_non_zero(values):
    return [v for v in values if v not in (None, 0, 0.0, "") and not pd.isna(v)]


def mode_or_none(values):
    filtered = [v for v in values if v not in (None, "") and not pd.isna(v)]
    if not filtered:
        return None
    counts = Counter(filtered)
    return sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))[0][0]


def median_or_zero(values):
    filtered = [float(v) for v in values if v not in (None, "") and not pd.isna(v)]
    if not filtered:
        return 0.0
    return float(statistics.median(filtered))


def earliest_or_none(values):
    filtered = [v for v in values if v not in (None, "") and not pd.isna(v)]
    if not filtered:
        return None
    return min(filtered)


def infer_group_keys(row):
    area = canonical_area(row)
    if area <= 0:
        return []

    base = (
        normalize_text(row["cidade"]),
        canonical_tipo(row["tipo_imovel"]),
        round(area, 1),
        safe_int(row["quartos"]),
        safe_int(row["banheiros"]),
        safe_int(row["vagas"]),
    )

    keys = []
    bairro = normalize_text(row["bairro"])
    if bairro:
        keys.append(("bairro",) + base + (bairro,))

    endereco = canonical_endereco(row.get("endereco"))
    if endereco:
        keys.append(("endereco",) + base + (endereco,))
        street = canonical_endereco(row.get("endereco"), street_only=True)
        if street and street != endereco:
            keys.append(("logradouro",) + base + (street,))

    return keys


def source_prefix(imovel_id):
    if not imovel_id or "-" not in imovel_id:
        return ""
    return imovel_id.split("-", 1)[0]


def extract_codigo(row):
    codigo = row.get("codigo")
    if codigo not in (None, ""):
        return str(codigo).strip()

    imovel_id = row.get("id")
    if imovel_id not in (None, ""):
        imovel_id = str(imovel_id).strip()
        if "-" in imovel_id:
            return imovel_id.split("-", 1)[1]
        return imovel_id

    return ""


def parse_datetime(value):
    if value in (None, ""):
        return None

    text = str(value).strip().replace("Z", "+00:00")
    fmts = (
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
    )
    for fmt in fmts:
        try:
            dt = datetime.strptime(text, fmt)
            return dt.replace(tzinfo=None)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(text)
        return dt.replace(tzinfo=None)
    except ValueError:
        return None


def db_has_column(conn, table_name, column_name):
    cur = conn.cursor()
    cols = [row[1] for row in cur.execute(f"PRAGMA table_info({table_name})")]
    return column_name in cols


def list_source_dbs(pattern=DEFAULT_DB_PATTERN):
    cwd = Path.cwd()
    candidates = sorted(cwd.glob(pattern))
    excluded = {DEFAULT_NORMALIZED_DB.name, DEFAULT_UNIFIED_DB.name}
    return [path for path in candidates if path.name not in excluded and path.is_file()]


def resolve_source_db(source_db=None):
    if source_db:
        return Path(source_db)
    if DEFAULT_UNIFIED_DB.exists():
        return DEFAULT_UNIFIED_DB
    if DEFAULT_RAW_DB.exists():
        return DEFAULT_RAW_DB
    return DEFAULT_RAW_DB


def print_property_types():
    print("Tipos de imovel aceitos:")
    for item in PROPERTY_TYPE_CATALOG:
        aliases = ", ".join(item["aliases"]) if item["aliases"] else "-"
        print(f"- {item['canonical']}: {item['description']} (aliases: {aliases})")


def build_group_id(key):
    raw = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"NR-{digest}"


def build_image_hash(content):
    if Image is None or ImageOps is None:
        return ""

    try:
        image = Image.open(BytesIO(content))
        image = ImageOps.exif_transpose(image).convert("L")
        resize_mode = getattr(Image, "Resampling", Image).LANCZOS
        image = image.resize((IMAGE_HASH_SIZE + 1, IMAGE_HASH_SIZE), resize_mode)
        pixels = np.asarray(image, dtype=np.uint8)
        diff = pixels[:, 1:] > pixels[:, :-1]
        bit_string = "".join("1" if value else "0" for value in diff.flatten())
        return f"{int(bit_string, 2):016x}"
    except Exception:
        return ""


def fetch_image_hash(image_url, session, cache):
    if not image_url or Image is None or ImageOps is None:
        return ""
    if image_url in cache:
        return cache[image_url]

    try:
        response = session.get(image_url, timeout=20)
        response.raise_for_status()
        image_hash = build_image_hash(response.content)
    except Exception:
        image_hash = ""

    cache[image_url] = image_hash
    return image_hash


def attach_image_hashes(rows, progress=False):
    bar = ProgressBar(len(rows), "Hash de imagens", enabled=progress)
    if not rows or Image is None or ImageOps is None:
        for row in rows:
            row["_imagem_hash"] = ""
            bar.advance()
        bar.close()
        return rows

    cache = {}
    with requests.Session() as session:
        session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            }
        )
        for row in rows:
            row["_imagem_hash"] = fetch_image_hash(row.get("imagem_url"), session, cache)
            bar.advance()
    bar.close()
    return rows


def image_hash_distance(left_hash, right_hash):
    if not left_hash or not right_hash:
        return None
    return (int(left_hash, 16) ^ int(right_hash, 16)).bit_count()


def relative_difference(left, right):
    left = float(left or 0)
    right = float(right or 0)
    base = max(abs(left), abs(right))
    if base <= 0:
        return 0.0
    return abs(left - right) / base


def visual_candidate_key(row):
    return (
        normalize_text(row["cidade"]),
        canonical_tipo(row["tipo_imovel"]),
        safe_int(row["quartos"]),
        safe_int(row["banheiros"]),
        min(safe_int(row["vagas"]), 4),
    )


def are_visual_duplicates(left, right):
    left_hash = left.get("_imagem_hash")
    right_hash = right.get("_imagem_hash")
    distance = image_hash_distance(left_hash, right_hash)
    if distance is None or distance > IMAGE_HASH_MAX_DISTANCE:
        return False

    left_area = canonical_area(left)
    right_area = canonical_area(right)
    if left_area <= 0 or right_area <= 0:
        return False
    if relative_difference(left_area, right_area) > IMAGE_AREA_TOLERANCE:
        return False

    left_price = safe_float(left.get("preco"))
    right_price = safe_float(right.get("preco"))
    if left_price > 0 and right_price > 0:
        if relative_difference(left_price, right_price) > IMAGE_PRICE_TOLERANCE:
            return False

    return True


def read_source_rows(source_db):
    conn = sqlite3.connect(source_db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    has_codigo = db_has_column(conn, "imoveis", "codigo")
    has_endereco = db_has_column(conn, "imoveis", "endereco")
    has_imagem_url = db_has_column(conn, "imoveis", "imagem_url")
    codigo_select = "codigo" if has_codigo else "NULL AS codigo"
    endereco_select = "endereco" if has_endereco else "NULL AS endereco"
    imagem_select = "imagem_url" if has_imagem_url else "NULL AS imagem_url"

    rows = cur.execute(
        f"""
        SELECT
            {codigo_select},
            {endereco_select},
            {imagem_select},
            id, preco, preco_m2, bairro, cidade, tipo_imovel,
            area_total, area_privada, quartos, banheiros, vagas,
            date_registration, data_insercao, data_atualizacao
        FROM imoveis
        WHERE preco IS NOT NULL AND preco > 0
        """
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def merge_source_dbs(source_dbs, output_db):
    all_rows = []
    for db_path in source_dbs:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        has_codigo = db_has_column(conn, "imoveis", "codigo")
        has_endereco = db_has_column(conn, "imoveis", "endereco")
        has_imagem_url = db_has_column(conn, "imoveis", "imagem_url")
        codigo_select = "codigo" if has_codigo else "NULL AS codigo"
        endereco_select = "endereco" if has_endereco else "NULL AS endereco"
        imagem_select = "imagem_url" if has_imagem_url else "NULL AS imagem_url"
        rows = cur.execute(
            f"""
            SELECT
                {codigo_select},
                {endereco_select},
                {imagem_select},
                id, preco, preco_m2, bairro, cidade, tipo_imovel,
                area_total, area_privada, quartos, banheiros, vagas,
                date_registration, data_insercao, data_atualizacao
            FROM imoveis
            WHERE preco IS NOT NULL AND preco > 0
            """
        ).fetchall()
        conn.close()
        for row in rows:
            record = dict(row)
            record["_source_db"] = str(db_path)
            record["_codigo"] = extract_codigo(record)
            all_rows.append(record)

    grouped = defaultdict(list)
    for row in all_rows:
        codigo = row.get("_codigo") or ""
        if not codigo:
            continue
        grouped[codigo].append(row)

    def row_rank(item):
        dt = (
            parse_datetime(item.get("data_atualizacao"))
            or parse_datetime(item.get("data_insercao"))
            or parse_datetime(item.get("date_registration"))
            or datetime.min
        )
        completeness = sum(
            1
            for field in (
                "preco",
                "bairro",
                "endereco",
                "imagem_url",
                "cidade",
                "tipo_imovel",
                "area_total",
                "area_privada",
                "quartos",
                "banheiros",
                "vagas",
            )
            if item.get(field) not in (None, "", 0, 0.0)
        )
        return (dt, completeness)

    def has_value(value):
        return value not in (None, "", 0, 0.0) and not pd.isna(value)

    output_db.parent.mkdir(parents=True, exist_ok=True)
    if output_db.exists():
        output_db.unlink()

    conn = sqlite3.connect(output_db)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE imoveis (
            codigo TEXT PRIMARY KEY,
            id TEXT,
            preco REAL,
            preco_m2 REAL,
            bairro TEXT,
            endereco TEXT,
            imagem_url TEXT,
            cidade TEXT,
            tipo_imovel TEXT,
            area_total REAL,
            area_privada REAL,
            quartos INTEGER,
            banheiros INTEGER,
            vagas INTEGER,
            date_registration TEXT,
            data_insercao TEXT,
            data_atualizacao TEXT,
            source_ids TEXT,
            source_dbs TEXT,
            source_count INTEGER
        )
        """
    )

    registros = []
    for codigo, items in grouped.items():
        ordered = sorted(items, key=row_rank, reverse=True)
        source_ids = sorted({str(item.get("id")) for item in items if item.get("id")})
        source_dbs = sorted({str(item.get("_source_db")) for item in items if item.get("_source_db")})

        merged = {
            "codigo": codigo,
            "id": codigo,
            "preco": None,
            "preco_m2": None,
            "bairro": None,
            "endereco": None,
            "imagem_url": None,
            "cidade": None,
            "tipo_imovel": None,
            "area_total": None,
            "area_privada": None,
            "quartos": None,
            "banheiros": None,
            "vagas": None,
            "date_registration": None,
            "data_insercao": None,
            "data_atualizacao": None,
        }

        for item in ordered:
            for field in (
                "preco",
                "preco_m2",
                "bairro",
                "endereco",
                "imagem_url",
                "cidade",
                "tipo_imovel",
                "area_total",
                "area_privada",
                "quartos",
                "banheiros",
                "vagas",
                "date_registration",
                "data_insercao",
                "data_atualizacao",
            ):
                if has_value(merged[field]):
                    continue
                value = item.get(field)
                if has_value(value):
                    merged[field] = value

        merged["quartos"] = safe_int(merged["quartos"])
        merged["banheiros"] = safe_int(merged["banheiros"])
        merged["vagas"] = safe_int(merged["vagas"])
        merged["area_total"] = safe_float(merged["area_total"])
        merged["area_privada"] = safe_float(merged["area_privada"])
        merged["preco"] = safe_float(merged["preco"])
        area_ref = merged["area_privada"] or merged["area_total"]
        if area_ref and merged["preco"] > 0:
            merged["preco_m2"] = round(merged["preco"] / area_ref, 6)
        else:
            merged["preco_m2"] = safe_float(merged["preco_m2"])

        merged["data_insercao"] = (
            min(
                (
                    dt
                    for dt in (
                        parse_datetime(item.get("data_insercao")) for item in items
                    )
                    if dt is not None
                ),
                default=None,
            )
            or datetime.now()
        ).isoformat()
        merged["data_atualizacao"] = (
            max(
                (
                    dt
                    for dt in (
                        parse_datetime(item.get("data_atualizacao")) for item in items
                    )
                    if dt is not None
                ),
                default=None,
            )
            or datetime.now()
        ).isoformat()

        registros.append(
            {
                **merged,
                "source_ids": json.dumps(source_ids, ensure_ascii=False),
                "source_dbs": json.dumps(source_dbs, ensure_ascii=False),
                "source_count": len(items),
            }
        )

    cur.executemany(
        """
        INSERT INTO imoveis (
            codigo, id, preco, preco_m2, bairro, endereco, imagem_url, cidade, tipo_imovel,
            area_total, area_privada, quartos, banheiros, vagas,
            date_registration, data_insercao, data_atualizacao,
            source_ids, source_dbs, source_count
        ) VALUES (
            :codigo, :id, :preco, :preco_m2, :bairro, :endereco, :imagem_url, :cidade, :tipo_imovel,
            :area_total, :area_privada, :quartos, :banheiros, :vagas,
            :date_registration, :data_insercao, :data_atualizacao,
            :source_ids, :source_dbs, :source_count
        )
        """,
        registros,
    )
    conn.commit()
    conn.close()

    return len(all_rows), len(registros)


def cluster_rows(rows, progress=False):
    if not rows:
        return []

    rows = attach_image_hashes(rows, progress=progress)
    parents = list(range(len(rows)))
    key_owner = {}

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left, right):
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parents[root_right] = root_left

    key_bar = ProgressBar(len(rows), "Agrupando por endereco/bairro", enabled=progress)
    for index, row in enumerate(rows):
        for key in infer_group_keys(row):
            owner = key_owner.get(key)
            if owner is None:
                key_owner[key] = index
            else:
                union(index, owner)
        key_bar.advance()
    key_bar.close()

    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[find(index)].append(row)

    visual_groups = defaultdict(list)
    for index, row in enumerate(rows):
        if row.get("_imagem_hash"):
            visual_groups[visual_candidate_key(row)].append(index)

    visual_groups_list = list(visual_groups.values())
    visual_bar = ProgressBar(len(visual_groups_list), "Comparando por imagem", enabled=progress)
    for indexes in visual_groups_list:
        indexes.sort(key=lambda idx: canonical_area(rows[idx]))
        for pos, left_idx in enumerate(indexes):
            left_row = rows[left_idx]
            left_area = canonical_area(left_row)
            for right_idx in indexes[pos + 1 :]:
                right_row = rows[right_idx]
                right_area = canonical_area(right_row)
                if left_area > 0 and right_area > 0:
                    if relative_difference(left_area, right_area) > IMAGE_AREA_TOLERANCE:
                        if right_area > left_area:
                            break
                        continue
                if are_visual_duplicates(left_row, right_row):
                    union(left_idx, right_idx)
        visual_bar.advance()
    visual_bar.close()

    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[find(index)].append(row)

    return [(None, items) for items in grouped.values()]


def aggregate_cluster(key, items):
    price_values = [safe_float(r["preco"]) for r in items if safe_float(r["preco"]) > 0]
    area_total_values = list_non_zero([safe_float(r["area_total"]) for r in items])
    area_privada_values = list_non_zero([safe_float(r["area_privada"]) for r in items])
    area_values = list_non_zero([canonical_area(r) for r in items])

    bairro = mode_or_none([r["bairro"] for r in items]) or ""
    endereco_values = [r.get("endereco") for r in items]
    imagem_values = [r.get("imagem_url") for r in items]
    imagem_hashes = [r.get("_imagem_hash") for r in items if r.get("_imagem_hash")]
    cidade = mode_or_none([r["cidade"] for r in items]) or ""
    tipo = mode_or_none([canonical_tipo(r["tipo_imovel"]) for r in items]) or ""
    quartos_values = [safe_int(r["quartos"]) for r in items]
    banheiros_values = [safe_int(r["banheiros"]) for r in items]
    vagas_values = [safe_int(r["vagas"]) for r in items]
    quartos = mode_or_none([v for v in quartos_values if v > 0]) or 0
    banheiros = mode_or_none([v for v in banheiros_values if v > 0]) or 0
    vagas = mode_or_none([v for v in vagas_values if v > 0]) or 0

    area_m2 = median_or_zero(area_values)
    area_total = median_or_zero(area_total_values)
    area_privada = median_or_zero(area_privada_values)

    if area_m2 <= 0:
        area_m2 = max(area_total, area_privada)

    if area_total <= 0 and area_m2 > 0:
        area_total = area_m2
    if area_privada <= 0 and area_m2 > 0:
        area_privada = area_m2

    preco = median_or_zero(price_values)
    preco_min = min(price_values) if price_values else 0.0
    preco_max = max(price_values) if price_values else 0.0
    preco_m2 = (preco / area_m2) if (preco > 0 and area_m2 > 0) else 0.0

    origem_ids = sorted({r["id"] for r in items if r["id"]})
    origem_prefixos = sorted({source_prefix(i) for i in origem_ids if source_prefix(i)})
    date_registration = earliest_or_none([r["date_registration"] for r in items])
    data_insercao = earliest_or_none([r["data_insercao"] for r in items]) or datetime.now().isoformat()
    data_atualizacao = datetime.now().isoformat()

    endereco = ""
    endereco_map = defaultdict(list)
    for raw in endereco_values:
        if raw in (None, ""):
            continue
        normalized = canonical_endereco(raw)
        if normalized:
            endereco_map[normalized].append(str(raw).strip())
    if endereco_map:
        normalized = sorted(
            endereco_map.items(),
            key=lambda item: (-len(item[1]), -len(item[0]), item[0]),
        )[0][0]
        endereco = max(endereco_map[normalized], key=len)

    imagem_url = mode_or_none([url for url in imagem_values if url]) or None
    imagem_hash = mode_or_none(imagem_hashes) or ""

    canonical_location = canonical_endereco(endereco) or normalize_text(bairro)
    canonical_key = key or (
        normalize_text(cidade),
        canonical_location,
        canonical_tipo(tipo),
        round(area_m2, 1) if area_m2 else 0,
        safe_int(quartos),
        safe_int(banheiros),
        safe_int(vagas),
    )
    normalized_id = build_group_id(canonical_key + (tuple(origem_ids),))

    return {
        "id": normalized_id,
        "preco": round(preco, 2),
        "preco_m2": round(preco_m2, 6) if preco_m2 else 0.0,
        "bairro": bairro,
        "endereco": endereco,
        "imagem_url": imagem_url,
        "imagem_hash": imagem_hash,
        "cidade": cidade,
        "tipo_imovel": tipo,
        "area_total": round(area_total, 2),
        "area_privada": round(area_privada, 2),
        "area_m2": round(area_m2, 2),
        "quartos": safe_int(quartos),
        "banheiros": safe_int(banheiros),
        "vagas": safe_int(vagas),
        "date_registration": date_registration,
        "data_insercao": data_insercao,
        "data_atualizacao": data_atualizacao,
        "origem_ids": json.dumps(origem_ids, ensure_ascii=False),
        "origem_prefixos": json.dumps(origem_prefixos, ensure_ascii=False),
        "origem_quantidade": len(origem_ids),
        "preco_min": round(preco_min, 2),
        "preco_max": round(preco_max, 2),
    }


def write_normalized_db(source_db, output_db, progress=False):
    rows = read_source_rows(source_db)
    clusters = cluster_rows(rows, progress=progress)

    output_db.parent.mkdir(parents=True, exist_ok=True)
    if output_db.exists():
        output_db.unlink()

    conn = sqlite3.connect(output_db)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE imoveis_normalizados (
            id TEXT PRIMARY KEY,
            preco REAL,
            preco_m2 REAL,
            bairro TEXT,
            endereco TEXT,
            imagem_url TEXT,
            imagem_hash TEXT,
            cidade TEXT,
            tipo_imovel TEXT,
            area_total REAL,
            area_privada REAL,
            area_m2 REAL,
            quartos INTEGER,
            banheiros INTEGER,
            vagas INTEGER,
            date_registration TEXT,
            data_insercao TEXT,
            data_atualizacao TEXT,
            origem_ids TEXT,
            origem_prefixos TEXT,
            origem_quantidade INTEGER,
            preco_min REAL,
            preco_max REAL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE normalizacao_resumo (
            chave TEXT PRIMARY KEY,
            valor TEXT
        )
        """
    )

    registros = []
    agg_bar = ProgressBar(len(clusters), "Consolidando registros", enabled=progress)
    for key, items in clusters:
        registros.append(aggregate_cluster(key, items))
        agg_bar.advance()
    agg_bar.close()

    cur.executemany(
        """
        INSERT INTO imoveis_normalizados (
            id, preco, preco_m2, bairro, endereco, imagem_url, imagem_hash, cidade, tipo_imovel,
            area_total, area_privada, area_m2, quartos, banheiros, vagas,
            date_registration, data_insercao, data_atualizacao,
            origem_ids, origem_prefixos, origem_quantidade, preco_min, preco_max
        ) VALUES (
            :id, :preco, :preco_m2, :bairro, :endereco, :imagem_url, :imagem_hash, :cidade, :tipo_imovel,
            :area_total, :area_privada, :area_m2, :quartos, :banheiros, :vagas,
            :date_registration, :data_insercao, :data_atualizacao,
            :origem_ids, :origem_prefixos, :origem_quantidade, :preco_min, :preco_max
        )
        """,
        registros,
    )

    resumo = {
        "source_db": str(source_db),
        "source_rows": str(len(rows)),
        "normalized_rows": str(len(registros)),
        "generated_at": datetime.now().isoformat(),
        "price_rows": str(sum(1 for r in rows if safe_float(r["preco"]) > 0)),
    }
    cur.executemany(
        "INSERT INTO normalizacao_resumo (chave, valor) VALUES (?, ?)",
        list(resumo.items()),
    )
    conn.commit()
    conn.close()

    return len(rows), len(registros)


def load_training_frame(normalized_db):
    conn = sqlite3.connect(normalized_db)
    df = pd.read_sql_query(
        """
        SELECT
            preco,
            bairro,
            tipo_imovel,
            area_total,
            area_privada,
            quartos,
            banheiros,
            vagas
        FROM imoveis_normalizados
        WHERE preco IS NOT NULL AND preco > 0
        """,
        conn,
    )
    conn.close()

    if df.empty:
        raise RuntimeError("Nenhum registro válido encontrado para treinamento.")

    for col in ("area_total", "area_privada", "quartos", "banheiros", "vagas"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df = df[(df["area_total"] > 0) | (df["area_privada"] > 0)]
    df["bairro"] = df["bairro"].fillna("").map(normalize_text)
    df["tipo_imovel"] = df["tipo_imovel"].fillna("").map(canonical_tipo)
    df["preco"] = pd.to_numeric(df["preco"], errors="coerce")
    df = df[df["preco"].notna() & (df["preco"] > 0)]
    return df


def build_feature_matrix(df):
    features = df[FEATURE_COLUMNS].copy()
    numeric = features[NUMERIC_COLUMNS].fillna(0.0)
    categorical = features[CATEGORICAL_COLUMNS].fillna("").astype(str)
    X = pd.concat([numeric, pd.get_dummies(categorical, prefix=["bairro", "tipo"])], axis=1)
    return X


def compute_rare_bairros(df, min_count=10):
    if df.empty or "bairro" not in df.columns:
        return set()

    counts = df["bairro"].fillna("").astype(str).value_counts()
    return set(counts[counts < min_count].index)


def apply_rare_bairros(df, rare_bairros, rare_label="outros"):
    if df.empty or "bairro" not in df.columns or not rare_bairros:
        return df.copy()

    work = df.copy()
    work["bairro"] = work["bairro"].where(~work["bairro"].isin(rare_bairros), rare_label)
    return work


def compute_numeric_clip_bounds(df, columns=NUMERIC_COLUMNS, quantiles=BASELINE_CLIP_QUANTILES):
    bounds = {}
    if df.empty:
        return bounds

    lower_q, upper_q = quantiles
    for column in columns:
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        series = series[series > 0]
        if len(series) < 8:
            continue
        lower = float(series.quantile(lower_q))
        upper = float(series.quantile(upper_q))
        if lower <= upper:
            bounds[column] = (lower, upper)
    return bounds


def apply_numeric_clip_bounds(df, bounds):
    if df.empty or not bounds:
        return df.copy()

    work = df.copy()
    for column, (lower, upper) in bounds.items():
        if column not in work.columns:
            continue
        work[column] = pd.to_numeric(work[column], errors="coerce").clip(lower=lower, upper=upper)
    return work


def make_preprocessor():
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - compatibility fallback
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="")),
            ("onehot", encoder),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLUMNS),
            ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
        ]
    )


def make_model_factories(seed=42):
    factories = {
        "Regressão Linear": lambda: LinearRegression(),
        "Ridge": lambda: Ridge(alpha=5.0),
        "Lasso": lambda: Lasso(alpha=0.0005, max_iter=20000, random_state=seed),
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
        "Gradient Boosting": lambda: GradientBoostingRegressor(
            random_state=seed,
            learning_rate=0.05,
            n_estimators=300,
            max_depth=3,
        ),
    }

    if XGBRegressor is not None:
        factories["XGBoost"] = lambda: XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
        )

    if CatBoostRegressor is not None:
        factories["CatBoost"] = lambda: CatBoostRegressor(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            random_seed=seed,
            verbose=False,
        )

    return factories


def build_regressor(name, seed=42):
    preprocessor = make_preprocessor()
    factories = make_model_factories(seed=seed)
    if name not in factories:
        raise ValueError(f"Modelo desconhecido: {name}")
    regressor = factories[name]()
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", regressor),
        ]
    )
    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    denom = np.where(y_true == 0, np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
    }


def filter_property_subset(df, allowed_types=("Casa", "Apartamento")):
    allowed = {canonical_tipo(t) for t in allowed_types}
    subset = df[df["tipo_imovel"].map(canonical_tipo).isin(allowed)].copy()
    return subset


def remove_outliers_iqr(df, columns=("preco", "preco_m2"), by_type=True, factor=1.5):
    if df.empty:
        return df.copy()

    work = df.copy()
    keep = pd.Series(True, index=work.index)

    groups = [("all", work)] if not by_type else list(work.groupby(work["tipo_imovel"].map(canonical_tipo)))
    for _, group in groups:
        if group.empty:
            continue
        group_keep = pd.Series(True, index=group.index)
        for column in columns:
            if column not in group.columns:
                continue
            series = pd.to_numeric(group[column], errors="coerce").dropna()
            series = series[series > 0]
            if len(series) < 8:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr <= 0:
                continue
            lower = q1 - factor * iqr
            upper = q3 + factor * iqr
            col_values = pd.to_numeric(group[column], errors="coerce")
            group_keep &= col_values.between(lower, upper) | col_values.isna()
        keep.loc[group.index] &= group_keep

    return work[keep].copy()


def benchmark_ridge_variant(df, *, use_log=False, seed=42, test_size=0.2):
    X = df[FEATURE_COLUMNS].copy()
    y = df["preco"].astype(float).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    model = build_regressor("Ridge", seed=seed) if use_log else Pipeline(
        steps=[
            ("preprocess", make_preprocessor()),
            ("model", Ridge(alpha=5.0)),
        ]
    )

    if use_log:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    return calculate_metrics(y_test, preds), model


def benchmark_separate_by_type(df, seed=42, test_size=0.2):
    rows = []
    all_true = []
    all_pred = []
    for tipo in ("Casa", "Apartamento"):
        subset = df[df["tipo_imovel"].map(canonical_tipo) == tipo].copy()
        if len(subset) < 10:
            continue
        metrics, model = benchmark_ridge_variant(subset, use_log=False, seed=seed, test_size=test_size)
        rows.append({"model": f"{tipo} (modelo separado)", **metrics, "n": len(subset)})

        X = subset[FEATURE_COLUMNS].copy()
        y = subset["preco"].astype(float).to_numpy()
        _, X_test, _, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
        preds = model.predict(X_test)
        all_true.extend(y_test.tolist())
        all_pred.extend(preds.tolist())

    combined = calculate_metrics(all_true, all_pred) if all_true else {
        "mae": 0.0, "rmse": 0.0, "r2": 0.0, "mape": 0.0
    }
    return rows, combined


def benchmark_type_scenarios(df, seed=42, test_size=0.2):
    subset = filter_property_subset(df, allowed_types=("Casa", "Apartamento"))
    subset = subset.copy()
    area_ref = subset["area_privada"].where(subset["area_privada"] > 0, subset["area_total"])
    area_ref = area_ref.where(area_ref > 0)
    subset["preco_m2"] = subset["preco"] / area_ref

    scenarios = []

    metrics_1, _ = benchmark_ridge_variant(subset, use_log=False, seed=seed, test_size=test_size)
    scenarios.append(
        {
            "scenario": "1. Casas/Apartamentos - modelo único",
            **metrics_1,
            "n": len(subset),
        }
    )

    separate_rows, combined_metrics = benchmark_separate_by_type(
        subset, seed=seed, test_size=test_size
    )
    scenarios.extend(
        [
            {
                "scenario": "2. Casas/Apartamentos - modelos separados",
                **combined_metrics,
                "n": len(subset),
            }
        ]
    )

    subset_no_out = remove_outliers_iqr(subset, columns=("preco", "preco_m2"), by_type=True)
    metrics_3, _ = benchmark_ridge_variant(
        subset_no_out, use_log=False, seed=seed, test_size=test_size
    )
    scenarios.append(
        {
            "scenario": "3. Casas/Apartamentos - sem outliers",
            **metrics_3,
            "n": len(subset_no_out),
        }
    )

    metrics_4, _ = benchmark_ridge_variant(
        subset_no_out, use_log=True, seed=seed, test_size=test_size
    )
    scenarios.append(
        {
            "scenario": "4. Casas/Apartamentos - sem outliers + log(preco)",
            **metrics_4,
            "n": len(subset_no_out),
        }
    )

    return scenarios, separate_rows


def build_type_scenario_markdown(scenarios, separate_rows):
    lines = [
        "# Experimentos Casas/Apartamentos",
        "",
        "Base do teste: apenas imóveis classificados como **Casa** ou **Apartamento**.",
        "",
        "## Cenários",
        "",
        "| Cenário | N | MAE | RMSE | R² | MAPE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in scenarios:
        lines.append(
            f"| {row['scenario']} | {row['n']} | {format_currency(row['mae'])} | "
            f"{format_currency(row['rmse'])} | {row['r2']:.4f} | {row['mape']:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Modelos separados",
            "",
            "| Tipo | N | MAE | RMSE | R² | MAPE |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in separate_rows:
        lines.append(
            f"| {row['model']} | {row['n']} | {format_currency(row['mae'])} | "
            f"{format_currency(row['rmse'])} | {row['r2']:.4f} | {row['mape']:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Leitura",
            "",
            "- O cenário 1 mede o ganho bruto ao reduzir a variedade de tipos.",
            "- O cenário 2 testa se separar casa e apartamento melhora o ajuste.",
            "- O cenário 3 avalia a remoção de outliers.",
            "- O cenário 4 avalia o efeito do `log(preco)` depois do corte de outliers.",
        ]
    )
    return "\n".join(lines)


def iter_param_grid(grid):
    if not grid:
        yield {}
        return

    items = list(grid.items())

    def rec(index, current):
        if index == len(items):
            yield dict(current)
            return
        key, values = items[index]
        for value in values:
            current[key] = value
            yield from rec(index + 1, current)
        current.pop(key, None)

    yield from rec(0, {})


def tuned_model_specs(seed=42):
    specs = [
        (
            "Ridge",
            Ridge,
            {"alpha": [0.1, 1.0, 5.0, 10.0, 20.0]},
        ),
        (
            "Lasso",
            Lasso,
            {"alpha": [0.0001, 0.0005, 0.001, 0.005], "max_iter": [20000]},
        ),
        (
            "Random Forest",
            RandomForestRegressor,
            {
                "n_estimators": [300],
                "max_depth": [None, 20],
                "min_samples_leaf": [1, 2],
            },
        ),
        (
            "Gradient Boosting",
            GradientBoostingRegressor,
            {
                "n_estimators": [300, 500],
                "learning_rate": [0.03, 0.05],
                "max_depth": [2, 3],
            },
        ),
    ]

    if XGBRegressor is not None:
        specs.append(
            (
                "XGBoost",
                XGBRegressor,
                {
                    "n_estimators": [300],
                    "learning_rate": [0.03, 0.05],
                    "max_depth": [4, 6],
                    "subsample": [0.8, 0.9],
                    "colsample_bytree": [0.8, 0.9],
                    "objective": ["reg:squarederror"],
                    "random_state": [seed],
                    "n_jobs": [-1],
                },
            )
        )

    if CatBoostRegressor is not None:
        specs.append(
            (
                "CatBoost",
                CatBoostRegressor,
                {
                    "iterations": [400],
                    "depth": [4, 6],
                    "learning_rate": [0.03, 0.05],
                    "loss_function": ["RMSE"],
                    "random_seed": [seed],
                    "verbose": [False],
                },
            )
        )

    return specs


def make_candidate_estimator(base_cls, params, use_log, seed=42):
    preprocessor = make_preprocessor()
    model = base_cls(**params)
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    if use_log:
        return TransformedTargetRegressor(
            regressor=pipeline,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )
    return pipeline


def tune_type_models(df, seed=42, test_size=0.2):
    subset = filter_property_subset(df, allowed_types=("Casa", "Apartamento"))
    subset = subset.copy()
    area_ref = subset["area_privada"].where(subset["area_privada"] > 0, subset["area_total"])
    area_ref = area_ref.where(area_ref > 0)
    subset["preco_m2"] = subset["preco"] / area_ref
    subset = remove_outliers_iqr(subset, columns=("preco", "preco_m2"), by_type=True)
    subset = subset[subset["preco"] > 0].copy()
    if subset.empty:
        return [], {"train_rows": 0, "test_rows": 0, "seed": seed, "test_size": test_size}

    X = subset[FEATURE_COLUMNS].copy()
    y = subset["preco"].astype(float).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    results = []
    specs = tuned_model_specs(seed=seed)
    for use_log in (False, True):
        for model_name, model_cls, grid in specs:
            best_row = None
            for params in iter_param_grid(grid):
                estimator = make_candidate_estimator(model_cls, params, use_log=use_log, seed=seed)
                estimator.fit(X_train, y_train)
                preds = estimator.predict(X_test)
                metrics = calculate_metrics(y_test, preds)
                row = {
                    "model": model_name,
                    "target": "log(preco)" if use_log else "preco",
                    "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
                    "n": int(len(subset)),
                    **metrics,
                }
                if best_row is None or row["mae"] < best_row["mae"]:
                    best_row = row
            if best_row:
                results.append(best_row)

    results.sort(key=lambda item: (item["mae"], item["rmse"]))
    split_info = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "seed": seed,
        "test_size": test_size,
        "n": int(len(subset)),
    }
    return results, split_info


def build_tuning_markdown(results, split_info):
    lines = [
        "# Tuning Casas/Apartamentos",
        "",
        f"- base final: {split_info['n']} imóveis após filtro e remoção de outliers",
        f"- treino: {split_info['train_rows']}",
        f"- teste: {split_info['test_rows']}",
        f"- seed: {split_info['seed']}",
        "",
        "| Modelo | Alvo | MAE | RMSE | R² | MAPE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['model']} | {row['target']} | {format_currency(row['mae'])} | "
            f"{format_currency(row['rmse'])} | {row['r2']:.4f} | {row['mape']:.2f}% |"
        )
    lines.append("")
    best = results[0]["model"] if results else "N/A"
    best_target = results[0]["target"] if results else "N/A"
    lines.extend(
        [
            f"Melhor combinação por MAE: **{best} ({best_target})**",
            "",
            "## Próximo uso",
            "",
            "- Se o objetivo for previsibilidade, use o melhor modelo dessa tabela.",
            "- Se o objetivo for estabilidade, compare MAE e R² juntos.",
        ]
    )
    return "\n".join(lines)


class DatasetService:
    """Encapsula a unificação e a normalização dos bancos coletados."""

    def unify(self, pattern, output_db):
        """Une todos os bancos que casam com o padrão informado."""
        source_dbs = list_source_dbs(pattern)
        if not source_dbs:
            raise SystemExit(f"Nenhum banco encontrado com o padrão {pattern}")
        return merge_source_dbs(source_dbs, Path(output_db)), len(source_dbs)

    def normalize(self, source_db, output_db, progress=False):
        """Gera o banco normalizado que alimenta os experimentos."""
        return write_normalized_db(Path(source_db), Path(output_db), progress=progress)


class BaselineService:
    """Treina, salva e executa o baseline padrão do projeto."""

    def train(self, normalized_db, seed=42, test_size=0.2):
        df = load_training_frame(Path(normalized_db))
        return fit_baseline_model(df, seed=seed, test_size=test_size, training_db=normalized_db)

    def save(self, artifact, model_path):
        save_artifact(artifact, Path(model_path))

    def predict(self, model_path, **features):
        return predict_price(Path(model_path), **features)


class ExperimentService:
    """Executa os benchmarks gerais e os experimentos focados em tipos."""

    def general_benchmark(self, normalized_db, seed=42, test_size=0.2):
        df = load_training_frame(Path(normalized_db))
        results, _, split_info = benchmark_models(df, seed=seed, test_size=test_size)
        return results, split_info

    def type_scenarios(self, normalized_db, seed=42, test_size=0.2):
        df = load_training_frame(Path(normalized_db))
        return benchmark_type_scenarios(df, seed=seed, test_size=test_size)

    def type_tuning(self, normalized_db, seed=42, test_size=0.2):
        df = load_training_frame(Path(normalized_db))
        return tune_type_models(df, seed=seed, test_size=test_size)


class MedianEnsembleModel:
    """Ensemble determinístico que prevê pela mediana das saídas dos modelos."""

    def __init__(self, seed=42):
        self.seed = seed
        self.models_ = []

    def _candidate_estimators(self):
        specs = [
            (
                "Gradient Boosting",
                GradientBoostingRegressor(
                    random_state=self.seed,
                    learning_rate=0.05,
                    n_estimators=500,
                    max_depth=3,
                ),
            ),
            (
                "Random Forest",
                RandomForestRegressor(
                    n_estimators=400,
                    random_state=self.seed,
                    n_jobs=-1,
                    min_samples_leaf=2,
                ),
            ),
            ("Ridge", Ridge(alpha=5.0)),
        ]

        if XGBRegressor is not None:
            specs.append(
                (
                    "XGBoost",
                    XGBRegressor(
                        n_estimators=400,
                        learning_rate=0.05,
                        max_depth=6,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="reg:squarederror",
                        random_state=self.seed,
                        n_jobs=-1,
                    ),
                )
            )

        if CatBoostRegressor is not None:
            specs.append(
                (
                    "CatBoost",
                    CatBoostRegressor(
                        iterations=500,
                        depth=6,
                        learning_rate=0.05,
                        loss_function="RMSE",
                        random_seed=self.seed,
                        verbose=False,
                    ),
                )
            )

        return specs

    def fit(self, X, y):
        self.models_ = []
        preprocessor = make_preprocessor()
        for name, estimator in self._candidate_estimators():
            pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
            pipeline.fit(X, y)
            self.models_.append((name, pipeline))
        return self

    def predict(self, X):
        if not self.models_:
            raise RuntimeError("MedianEnsembleModel nao foi treinado.")
        preds = np.vstack([model.predict(X) for _, model in self.models_])
        return np.median(preds, axis=0)


def prepare_segment_baseline_frame(df, property_type):
    subset = filter_property_subset(df, allowed_types=(property_type,))
    subset = subset.copy()
    subset = subset[subset["preco"] > 0].copy()
    rare_bairros = compute_rare_bairros(subset, min_count=BASELINE_RARE_BAIRRO_MIN_COUNT)
    clip_bounds = compute_numeric_clip_bounds(
        subset, columns=NUMERIC_COLUMNS, quantiles=BASELINE_CLIP_QUANTILES
    )
    subset = apply_rare_bairros(subset, rare_bairros)
    subset = apply_numeric_clip_bounds(subset, clip_bounds)
    area_ref = subset["area_privada"].where(subset["area_privada"] > 0, subset["area_total"])
    area_ref = area_ref.where(area_ref > 0)
    subset["preco_m2"] = subset["preco"] / area_ref
    subset = remove_outliers_iqr(subset, columns=("preco", "preco_m2"), by_type=False)
    subset = subset[subset["preco"] > 0].copy()
    return subset, rare_bairros, clip_bounds


def train_segmented_baseline(df, seed=42, test_size=0.2):
    segments = {}
    combined_true = []
    combined_pred = []
    total_rows = 0
    total_train_rows = 0
    total_test_rows = 0

    for property_type in BASELINE_SEGMENT_TYPES:
        subset, rare_bairros, clip_bounds = prepare_segment_baseline_frame(df, property_type)
        if subset.empty:
            continue

        X = subset[FEATURE_COLUMNS].copy()
        y = subset["preco"].astype(float).to_numpy()
        if len(subset) < 5:
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )
        model = MedianEnsembleModel(seed=seed)
        model.fit(X_train, y_train)
        segment_metrics = calculate_metrics(y_test, model.predict(X_test))

        final_model = MedianEnsembleModel(seed=seed)
        final_model.fit(X, y)
        segment_prediction = final_model.predict(X_test)
        combined_true.extend(y_test.tolist())
        combined_pred.extend(segment_prediction.tolist())
        total_rows += len(subset)

        segments[property_type] = {
            "model": final_model,
            "rows": int(len(subset)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "rare_bairros": sorted(rare_bairros),
            "clip_bounds": {k: list(v) for k, v in clip_bounds.items()},
            "metrics": segment_metrics,
        }
        total_train_rows += len(X_train)
        total_test_rows += len(X_test)

    if not segments:
        raise RuntimeError("Nao ha dados suficientes para treinar o baseline.")

    overall_metrics = calculate_metrics(combined_true, combined_pred) if combined_true else {
        "mae": 0.0,
        "rmse": 0.0,
        "r2": 0.0,
        "mape": 0.0,
    }
    return {
        "kind": "baseline_segmented_median_ensemble",
        "allowed_types": list(BASELINE_SEGMENT_TYPES),
        "baseline": {
            "filter": "Casa e Apartamento separados + bairros raros agrupados + clipping de extremos",
            "outlier_method": "IQR",
            "target": "preco",
            "strategy": "mediana das previsoes por segmento",
            "rare_bairro_min_count": BASELINE_RARE_BAIRRO_MIN_COUNT,
            "clip_quantiles": list(BASELINE_CLIP_QUANTILES),
        },
        "segments": segments,
        "metrics": {
            **overall_metrics,
            "rows": int(total_rows),
            "train_rows": int(total_train_rows),
            "test_rows": int(total_test_rows),
            "segments": {
                name: {
                    "rows": value["rows"],
                    "train_rows": value["train_rows"],
                    "test_rows": value["test_rows"],
                    "rare_bairros": value["rare_bairros"],
                    "clip_bounds": value["clip_bounds"],
                    **value["metrics"],
                }
                for name, value in segments.items()
            },
        },
        "trained_at": datetime.now().isoformat(),
        "seed": seed,
    }


def fit_baseline_model(df, seed=42, test_size=0.2, training_db=None):
    artifact = train_segmented_baseline(df, seed=seed, test_size=test_size)
    artifact["training_db"] = str(training_db) if training_db else None
    return artifact


def save_artifact(artifact, model_path):
    """Persiste o baseline em JSON portátil mesmo quando a extensao e .pkl."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(artifact)
    payload.pop("model", None)
    if "segments" in payload:
        payload["segments"] = {
            name: {key: val for key, val in segment.items() if key != "model"}
            for name, segment in payload["segments"].items()
        }
    model_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_artifact(model_path):
    """Carrega o baseline em JSON e mantém compatibilidade com pickles antigos."""
    path = Path(model_path)
    raw = path.read_bytes()
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        pass

    with path.open("rb") as f:
        try:
            return pickle.load(f)
        except Exception as exc:
            json_path = path.with_suffix(".json")
            if json_path.exists():
                try:
                    return json.loads(json_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            raise SystemExit(
                "Nao foi possivel carregar o modelo salvo. Refaça o treino no ambiente atual."
            ) from exc


def benchmark_models(df, seed=42, test_size=0.2):
    X = df[FEATURE_COLUMNS].copy()
    y = df["preco"].astype(float).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    results = []
    fitted = {}
    for name in make_model_factories(seed=seed):
        model = build_regressor(name, seed=seed)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = calculate_metrics(y_test, preds)
        results.append({"model": name, **metrics})
        fitted[name] = model

    results.sort(key=lambda item: (item["mae"], item["rmse"]))
    return results, fitted, {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "seed": seed,
        "test_size": test_size,
    }


def format_currency(value):
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def build_benchmark_markdown(results, split_info):
    lines = [
        "# Benchmark de modelos",
        "",
        f"- treino: {split_info['train_rows']}",
        f"- teste: {split_info['test_rows']}",
        f"- seed: {split_info['seed']}",
        "",
        "## Métricas",
        "",
        "- **MAE**: erro médio absoluto. Quanto menor, melhor.",
        "- **RMSE**: raiz do erro quadrático médio. Penaliza erros grandes. Quanto menor, melhor.",
        "- **R²**: capacidade explicativa do modelo. Quanto mais perto de 1, melhor.",
        "- **MAPE**: erro percentual médio absoluto. Quanto menor, melhor.",
        "",
        "## Resultado",
        "",
        "| Modelo | MAE | RMSE | R² | MAPE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['model']} | {format_currency(row['mae'])} | {format_currency(row['rmse'])} | "
            f"{row['r2']:.4f} | {row['mape']:.2f}% |"
        )
    lines.append("")
    best = results[0]["model"] if results else "N/A"
    lines.extend(
        [
            "## Leitura prática",
            "",
            f"- Melhor modelo por **MAE**: **{best}**",
            "- `R²` muito baixo ou negativo indica que o conjunto atual ainda explica pouco da variação de preços.",
            "- Os erros altos sugerem ruído, outliers e categorias pouco estáveis no banco.",
            "",
            "## Próximos testes",
            "",
            "- remoção de outliers;",
            "- agrupamento mais forte de bairros e tipos;",
            "- validação cruzada;",
            "- ajuste de hiperparâmetros;",
            "- modelos em escala log.",
        ]
    )
    return "\n".join(lines)


def train_ridge_model(df, alpha=5.0, test_ratio=0.2, seed=42):
    X = build_feature_matrix(df)
    y = np.log1p(df["preco"].astype(float).to_numpy())

    rng = np.random.default_rng(seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)
    split = max(1, int(len(indices) * (1 - test_ratio)))
    train_idx = indices[:split]
    test_idx = indices[split:]
    if len(test_idx) == 0:
        test_idx = train_idx

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    feature_columns = list(X.columns)
    X_train_m = X_train.to_numpy(dtype=float)
    X_test_m = X_test.reindex(columns=feature_columns, fill_value=0).to_numpy(dtype=float)

    ones_train = np.ones((X_train_m.shape[0], 1))
    ones_test = np.ones((X_test_m.shape[0], 1))
    X_train_aug = np.hstack([ones_train, X_train_m])
    X_test_aug = np.hstack([ones_test, X_test_m])

    identity = np.eye(X_train_aug.shape[1])
    identity[0, 0] = 0.0
    coef = np.linalg.solve(
        X_train_aug.T @ X_train_aug + alpha * identity,
        X_train_aug.T @ y_train,
    )

    pred_test = X_test_aug @ coef
    pred_price = np.expm1(pred_test)
    true_price = np.expm1(y_test)

    mae = float(np.mean(np.abs(pred_price - true_price)))
    rmse = float(np.sqrt(np.mean((pred_price - true_price) ** 2)))
    ss_tot = float(np.sum((true_price - true_price.mean()) ** 2))
    ss_res = float(np.sum((true_price - pred_price) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    model = {
        "version": 1,
        "trained_at": datetime.now().isoformat(),
        "alpha": alpha,
        "feature_columns": feature_columns,
        "coef": coef.tolist(),
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
        },
    }
    return model


def build_prediction_frame(
    model, area_total, area_privada, bairro, tipo_imovel, quartos, banheiros, vagas
):
    row = pd.DataFrame(
        [
            {
                "area_total": float(area_total or 0),
                "area_privada": float(area_privada or 0),
                "quartos": int(quartos or 0),
                "banheiros": int(banheiros or 0),
                "vagas": int(vagas or 0),
                "bairro": normalize_text(bairro),
                "tipo_imovel": canonical_tipo(tipo_imovel),
            }
        ]
    )
    X = build_feature_matrix(row)
    feature_columns = model["feature_columns"]
    X = X.reindex(columns=feature_columns, fill_value=0.0)
    X_aug = np.hstack([np.ones((len(X), 1)), X.to_numpy(dtype=float)])
    return X_aug


def predict_price(
    model_path, area_total, area_privada, bairro, tipo_imovel, quartos, banheiros, vagas
):
    artifact = load_artifact(model_path)
    if isinstance(artifact, dict) and artifact.get("kind") == "baseline_segmented_median_ensemble":
        tipo_norm = canonical_tipo(tipo_imovel)
        if tipo_norm not in BASELINE_ALLOWED_TYPES:
            raise SystemExit(
                f"Tipo '{tipo_imovel}' fora do baseline. Use somente Casa ou Apartamento."
            )

        training_db = artifact.get("training_db") or DEFAULT_NORMALIZED_DB
        df = load_training_frame(Path(training_db))
        subset, rare_bairros, clip_bounds = prepare_segment_baseline_frame(df, tipo_norm)
        if subset.empty:
            raise SystemExit("Nao ha dados suficientes para gerar a previsao.")

        model = MedianEnsembleModel(seed=int(artifact.get("seed", 42)))
        X = subset[FEATURE_COLUMNS].copy()
        y = subset["preco"].astype(float).to_numpy()
        model.fit(X, y)

        row = pd.DataFrame(
            [
                {
                    "area_total": float(area_total or 0),
                    "area_privada": float(area_privada or 0),
                    "quartos": int(quartos or 0),
                    "banheiros": int(banheiros or 0),
                    "vagas": int(vagas or 0),
                    "bairro": canonical_bairro(bairro),
                    "tipo_imovel": tipo_norm,
                }
            ]
        )
        row = apply_rare_bairros(row, rare_bairros)
        row = apply_numeric_clip_bounds(row, clip_bounds)
        prediction = float(model.predict(row)[0])
        return prediction, artifact.get("metrics", {})

    if isinstance(artifact, dict) and "coef" in artifact:
        X = build_prediction_frame(
            artifact,
            area_total=area_total,
            area_privada=area_privada,
            bairro=bairro,
            tipo_imovel=tipo_imovel,
            quartos=quartos,
            banheiros=banheiros,
            vagas=vagas,
        )
        coef = np.array(artifact["coef"], dtype=float)
        prediction = float(np.expm1(X @ coef)[0])
        return prediction, artifact.get("metrics", {})

    raise SystemExit("Arquivo de modelo invalido.")


def cmd_benchmark(args):
    """Executa o benchmark geral dos modelos."""
    normalized_db = Path(args.normalized_db or DEFAULT_NORMALIZED_DB)
    report_path = Path(args.report_path or "docs/modelo_comparativo.md")
    results, split_info = ExperimentService().general_benchmark(
        normalized_db, seed=args.seed, test_size=args.test_size
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_benchmark_markdown(results, split_info),
        encoding="utf-8",
    )

    print(f"Relatorio salvo em {report_path}")
    print("| Modelo | MAE | RMSE | R² | MAPE |")
    print("|---|---:|---:|---:|---:|")
    for row in results:
        print(
            f"| {row['model']} | {format_currency(row['mae'])} | {format_currency(row['rmse'])} | "
            f"{row['r2']:.4f} | {row['mape']:.2f}% |"
        )


def cmd_type_scenarios(args):
    """Executa os cenários focados em Casas e Apartamentos."""
    normalized_db = Path(args.normalized_db or DEFAULT_NORMALIZED_DB)
    report_path = Path(args.report_path or "docs/experimentos_casas_apartamentos.md")
    scenarios, separate_rows = ExperimentService().type_scenarios(
        normalized_db, seed=args.seed, test_size=args.test_size
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_type_scenario_markdown(scenarios, separate_rows),
        encoding="utf-8",
    )
    print(f"Relatorio salvo em {report_path}")
    print("| Cenário | N | MAE | RMSE | R² | MAPE |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in scenarios:
        print(
            f"| {row['scenario']} | {row['n']} | {format_currency(row['mae'])} | "
            f"{format_currency(row['rmse'])} | {row['r2']:.4f} | {row['mape']:.2f}% |"
        )


def cmd_type_tuning(args):
    """Executa a busca curta de hiperparâmetros no baseline limpo."""
    normalized_db = Path(args.normalized_db or DEFAULT_NORMALIZED_DB)
    report_path = Path(args.report_path or "docs/tuning_casas_apartamentos.md")
    results, split_info = ExperimentService().type_tuning(
        normalized_db, seed=args.seed, test_size=args.test_size
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_tuning_markdown(results, split_info), encoding="utf-8")
    print(f"Relatorio salvo em {report_path}")
    print("| Modelo | Alvo | MAE | RMSE | R² | MAPE |")
    print("|---|---|---:|---:|---:|---:|")
    for row in results:
        print(
            f"| {row['model']} | {row['target']} | {format_currency(row['mae'])} | "
            f"{format_currency(row['rmse'])} | {row['r2']:.4f} | {row['mape']:.2f}% |"
        )


def cmd_normalize(args):
    """Gera o banco normalizado a partir do banco unificado ou bruto."""
    source_db = resolve_source_db(args.source_db)
    output_db = Path(args.output_db or DEFAULT_NORMALIZED_DB)
    total_in, total_out = DatasetService().normalize(
        source_db, output_db, progress=(not args.no_progress)
    )
    print(f"Banco normalizado criado em {output_db}")
    print(f"Registros de origem: {total_in}")
    print(f"Registros normalizados: {total_out}")


def cmd_unify(args):
    """Une todos os bancos coletados em um único SQLite por código."""
    (total_in, total_out), source_count = DatasetService().unify(
        args.pattern, args.output_db
    )
    print(f"Banco unificado criado em {args.output_db}")
    print(f"Bancos de origem: {source_count}")
    print(f"Registros de origem: {total_in}")
    print(f"Registros unificados: {total_out}")


def cmd_types(args):
    print_property_types()


def cmd_train(args):
    """Treina e salva o baseline do projeto."""
    normalized_db = Path(args.normalized_db or DEFAULT_NORMALIZED_DB)
    model_path = Path(args.model_path or DEFAULT_MODEL_PATH)
    artifact = BaselineService().train(normalized_db, seed=args.seed, test_size=args.test_size)
    BaselineService().save(artifact, model_path)
    print(f"Modelo salvo em {model_path}")
    metrics = artifact["metrics"]
    print(
        f"MAE={metrics['mae']:.2f} RMSE={metrics['rmse']:.2f} R2={metrics['r2']:.3f} "
        f"treino={metrics['train_rows']} teste={metrics['test_rows']} base={metrics['rows']}"
    )


def cmd_predict(args):
    """Executa uma previsão usando o baseline salvo."""
    area_total = args.area_total if args.area_total is not None else args.area_m2
    area_privada = args.area_privada if args.area_privada is not None else args.area_m2
    if area_total is None and area_privada is None:
        raise SystemExit("Informe --area-total e/ou --area-privada (ou --area-m2).")
    prediction, metrics = predict_price(
        args.model_path,
        area_total=area_total,
        area_privada=area_privada,
        bairro=args.bairro,
        tipo_imovel=args.tipo_imovel,
        quartos=args.quartos,
        banheiros=args.banheiros,
        vagas=args.vagas,
    )
    print(f"Preço estimado: R$ {prediction:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    if metrics:
        print(
            f"Modelo treinado com MAE={metrics.get('mae', 0):.2f}, "
            f"RMSE={metrics.get('rmse', 0):.2f}, R2={metrics.get('r2', 0):.3f}"
        )


def build_parser():
    parser = argparse.ArgumentParser(description="Normaliza dados de imóveis e treina um preditor de preço.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_unify = subparsers.add_parser("unify", help="Unifica todos os bancos coletados por código.")
    p_unify.add_argument(
        "--pattern",
        default=DEFAULT_DB_PATTERN,
        help="Padrão dos bancos de origem.",
    )
    p_unify.add_argument(
        "--output-db",
        default=str(DEFAULT_UNIFIED_DB),
        help="Arquivo SQLite consolidado.",
    )
    p_unify.set_defaults(func=cmd_unify)

    p_types = subparsers.add_parser("types", help="Lista os tipos de imóvel aceitos.")
    p_types.set_defaults(func=cmd_types)

    p_benchmark = subparsers.add_parser(
        "benchmark", help="Executa e compara os modelos de regressão."
    )
    p_benchmark.add_argument(
        "--normalized-db", default=None, help="Banco SQLite normalizado."
    )
    p_benchmark.add_argument(
        "--report-path",
        default="docs/modelo_comparativo.md",
        help="Arquivo markdown do comparativo.",
    )
    p_benchmark.add_argument("--seed", type=int, default=42, help="Seed do split.")
    p_benchmark.add_argument(
        "--test-size", type=float, default=0.2, help="Proporção de teste."
    )
    p_benchmark.set_defaults(func=cmd_benchmark)

    p_type = subparsers.add_parser(
        "type-scenarios", help="Compara cenários focados em casas e apartamentos."
    )
    p_type.add_argument(
        "--normalized-db", default=None, help="Banco SQLite normalizado."
    )
    p_type.add_argument(
        "--report-path",
        default="docs/experimentos_casas_apartamentos.md",
        help="Arquivo markdown do experimento.",
    )
    p_type.add_argument("--seed", type=int, default=42, help="Seed do split.")
    p_type.add_argument(
        "--test-size", type=float, default=0.2, help="Proporção de teste."
    )
    p_type.set_defaults(func=cmd_type_scenarios)

    p_tune = subparsers.add_parser(
        "type-tuning", help="Tuna modelos em Casas/Apartamentos sem outliers."
    )
    p_tune.add_argument(
        "--normalized-db", default=None, help="Banco SQLite normalizado."
    )
    p_tune.add_argument(
        "--report-path",
        default="docs/tuning_casas_apartamentos.md",
        help="Arquivo markdown do tuning.",
    )
    p_tune.add_argument("--seed", type=int, default=42, help="Seed do split.")
    p_tune.add_argument(
        "--test-size", type=float, default=0.2, help="Proporção de teste."
    )
    p_tune.set_defaults(func=cmd_type_tuning)

    p_norm = subparsers.add_parser("normalize", help="Gera um banco SQLite normalizado.")
    p_norm.add_argument("--source-db", default=None, help="Banco de origem com a tabela imoveis.")
    p_norm.add_argument("--output-db", default=None, help="Arquivo SQLite de saída.")
    p_norm.add_argument(
        "--no-progress",
        action="store_true",
        help="Desativa a barra de progresso da normalização.",
    )
    p_norm.set_defaults(func=cmd_normalize)

    p_train = subparsers.add_parser("train", help="Treina o baseline de preço.")
    p_train.add_argument("--normalized-db", default=None, help="Banco SQLite normalizado.")
    p_train.add_argument("--model-path", default=None, help="Arquivo do modelo baseline.")
    p_train.add_argument("--test-size", type=float, default=0.2, help="Proporção de teste.")
    p_train.add_argument("--seed", type=int, default=42, help="Seed do split.")
    p_train.set_defaults(func=cmd_train)

    p_predict = subparsers.add_parser("predict", help="Prevê o preço de um imóvel.")
    p_predict.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Arquivo do modelo baseline.")
    p_predict.add_argument(
        "--area-total", type=float, default=None, help="Área total em m²."
    )
    p_predict.add_argument(
        "--area-privada", type=float, default=None, help="Área privativa em m²."
    )
    p_predict.add_argument(
        "--area-m2",
        type=float,
        default=None,
        help="Alias legado: preenche area_total e area_privada quando os dois não forem informados.",
    )
    p_predict.add_argument("--bairro", required=True, help="Bairro.")
    p_predict.add_argument("--tipo-imovel", required=True, help="Tipo do imóvel.")
    p_predict.add_argument("--quartos", type=int, default=0, help="Número de quartos.")
    p_predict.add_argument("--banheiros", type=int, default=0, help="Número de banheiros.")
    p_predict.add_argument("--vagas", type=int, default=0, help="Número de vagas.")
    p_predict.set_defaults(
        func=cmd_predict,
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
