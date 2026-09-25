import os
from pathlib import Path

# scripts/db/config.py -> raiz do projeto fica dois níveis acima de scripts/db
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"


def _read_env_file(env_path: Path) -> dict:
    """Parseia um arquivo .env simples (KEY=VALUE) e retorna um dict."""
    data = {}
    if not env_path.exists():
        return data
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            data[k] = v
    except (OSError, UnicodeDecodeError):
        return {}
    return data


def get_env_value(key: str, default=None):
    """Lê uma configuração da variável de ambiente ou, se ausente, do .env da raiz."""
    val = os.getenv(key)
    if val:
        return val
    return _read_env_file(ENV_PATH).get(key) or default


def require_env_value(key: str) -> str:
    """Igual a get_env_value, mas encerra com mensagem clara se a chave não existir.

    Usado para tokens/credenciais, que NÃO devem ficar escritos no código-fonte.
    """
    val = get_env_value(key)
    if not val:
        raise SystemExit(
            f"Configuração '{key}' não encontrada. Defina a variável de ambiente "
            f"ou adicione '{key}=...' no arquivo .env na raiz do projeto "
            f"(veja .env.example)."
        )
    return val


def get_db_path() -> str:
    """Retorna o caminho do arquivo de banco a partir da variável de ambiente
    IMOVEIS_DB ou do arquivo .env (procurando chaves IMOVEIS_DB, DB_NAME ou
    DATABASE). Se nada encontrado, retorna 'imoveis.db' (nome padrão).

    Uso: sqlite3.connect(get_db_path())
    """
    for key in ("IMOVEIS_DB", "DB_NAME", "DATABASE"):
        val = get_env_value(key)
        if val:
            return val
    return "imoveis.db"
