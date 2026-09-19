import re
import sqlite3
from datetime import datetime
from urllib.parse import urljoin

from db.config import get_db_path


class BaseScraper:
    PREFIX = ""
    CIDADE_PADRAO = None

    def __init__(self):
        self.conn = sqlite3.connect(get_db_path())
        self.cursor = self.conn.cursor()
        self._ensure_table()

    def _ensure_table(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS imoveis (
                id TEXT PRIMARY KEY,
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
                data_atualizacao TEXT
            )
            """)

        # Migration for existing databases created before data_atualizacao existed.
        self.cursor.execute("PRAGMA table_info(imoveis)")
        columns = {row[1] for row in self.cursor.fetchall()}
        if "endereco" not in columns:
            self.cursor.execute("ALTER TABLE imoveis ADD COLUMN endereco TEXT")
        if "imagem_url" not in columns:
            self.cursor.execute("ALTER TABLE imoveis ADD COLUMN imagem_url TEXT")
        if "data_atualizacao" not in columns:
            self.cursor.execute("ALTER TABLE imoveis ADD COLUMN data_atualizacao TEXT")

        self.conn.commit()

    @staticmethod
    def format_endereco(street, number=None, complement=None):
        def clean_part(value):
            text = str(value or "").strip()
            text = re.sub(r"\s+", " ", text)
            if not text:
                return ""
            if re.fullmatch(r"[*#xX.-/ ]+", text):
                return ""
            return text

        street = clean_part(street)
        number = clean_part(number)
        complement = clean_part(complement)

        if not street:
            return None

        formatted = street
        if number:
            formatted = f"{formatted}, {number}"
        if complement:
            formatted = f"{formatted} - {complement}"
        return formatted

    @staticmethod
    def absolutize_url(base_url, value):
        text = str(value or "").strip()
        if not text:
            return None
        return urljoin(base_url, text)

    def upsert_imovel(self, data: dict):
        if not data.get("id"):
            return

        cid = data.get("cidade") or self.CIDADE_PADRAO
        now_iso = datetime.now().isoformat()
        self.cursor.execute(
            """
            INSERT INTO imoveis (
                id, preco, preco_m2, bairro, endereco, imagem_url, cidade, tipo_imovel,
                area_total, area_privada, quartos, banheiros, vagas,
                date_registration, data_insercao, data_atualizacao
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                preco = excluded.preco,
                preco_m2 = excluded.preco_m2,
                bairro = excluded.bairro,
                endereco = excluded.endereco,
                imagem_url = excluded.imagem_url,
                cidade = excluded.cidade,
                tipo_imovel = excluded.tipo_imovel,
                area_total = excluded.area_total,
                area_privada = excluded.area_privada,
                quartos = excluded.quartos,
                banheiros = excluded.banheiros,
                vagas = excluded.vagas,
                date_registration = excluded.date_registration,
                data_atualizacao = excluded.data_atualizacao
            """,
            (
                f"{self.PREFIX}{data['id']}",
                data.get("preco"),
                data.get("preco_m2"),
                data.get("bairro"),
                data.get("endereco"),
                data.get("imagem_url"),
                cid,
                data.get("tipo_imovel"),
                data.get("area_total"),
                data.get("area_privada"),
                data.get("quartos"),
                data.get("banheiros"),
                data.get("vagas"),
                data.get("date_registration"),
                now_iso,
                now_iso,
            ),
        )
        self.conn.commit()

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
