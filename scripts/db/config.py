import os
from pathlib import Path


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
    except Exception:
        return {}
    return data


def get_db_path() -> str:
    """Retorna o caminho do arquivo de banco a partir da variável de ambiente
    IMOVEIS_DB ou do arquivo .env (procurando chaves IMOVEIS_DB, DB_NAME ou
    DATABASE). Se nada encontrado, retorna 'imoveis.db' (nome padrão).

    Uso: sqlite3.connect(get_db_path())
    """
    # 1) Checa variável de ambiente
    val = os.getenv("IMOVEIS_DB") or os.getenv("DB_NAME") or os.getenv("DATABASE")
    if val:
        return val

    # 2) Tenta ler .env no diretório do projeto (pai da pasta scripts)
    # Se este arquivo existir, procura chaves conhecidas
    project_root = Path(__file__).resolve().parent
    # se db.py estiver na raiz, project_root é a raiz; caso contrário, sobe
    if (project_root / "scripts").exists():
        # quando db.py for colocado na raiz, this is root
        root = project_root
    else:
        # fallback: usa parent
        root = project_root.parent

    env_path = root / ".env"
    env = _read_env_file(env_path)
    for key in ("IMOVEIS_DB", "DB_NAME", "DATABASE"):
        if key in env and env[key]:
            return env[key]

    # 3) fallback
    return "imoveis.db"
