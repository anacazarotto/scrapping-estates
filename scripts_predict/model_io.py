"""Salvamento e carregamento seguros de modelos serializados (joblib/pickle).

joblib/pickle executam código arbitrário ao carregar um arquivo. Para evitar que um
.pkl adulterado (baixado, recebido por e-mail, trocado na pasta) seja executado,
todo modelo salvo recebe uma assinatura HMAC-SHA256 em um arquivo ao lado
(`<modelo>.sig`). O carregamento só acontece se a assinatura bater.

A chave vem da variável de ambiente/.env `MODEL_SIGNING_KEY`. Sem chave, é usado
apenas o SHA-256 do arquivo (protege contra corrupção/troca acidental, mas não
contra alguém que consiga reescrever também o .sig). Para uso real, defina a chave.
"""

import hashlib
import hmac
import os
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIGNATURE_SUFFIX = ".sig"


def _read_env_key():
    key = os.getenv("MODEL_SIGNING_KEY")
    if key:
        return key
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("MODEL_SIGNING_KEY="):
                    value = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if value:
                        return value
        except (OSError, UnicodeDecodeError):
            return None
    return None


def _signature_path(model_path):
    model_path = Path(model_path)
    return model_path.with_name(model_path.name + SIGNATURE_SUFFIX)


def _compute_signature(data: bytes) -> str:
    key = _read_env_key()
    if key:
        return "hmac-sha256:" + hmac.new(key.encode("utf-8"), data, hashlib.sha256).hexdigest()
    return "sha256:" + hashlib.sha256(data).hexdigest()


def save_model(obj, model_path):
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, model_path)
    signature = _compute_signature(model_path.read_bytes())
    _signature_path(model_path).write_text(signature + "\n", encoding="utf-8")


def load_model(model_path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise SystemExit(f"Modelo não encontrado: {model_path}")

    sig_path = _signature_path(model_path)
    if not sig_path.exists():
        raise SystemExit(
            f"Assinatura ausente para {model_path} ({sig_path.name}). "
            "Por segurança o modelo não será carregado. Refaça o treino."
        )

    data = model_path.read_bytes()
    expected = sig_path.read_text(encoding="utf-8").strip()
    actual = _compute_signature(data)
    if not hmac.compare_digest(expected, actual):
        raise SystemExit(
            f"Assinatura inválida para {model_path}. O arquivo foi alterado ou foi "
            "gerado com outra MODEL_SIGNING_KEY. Por segurança ele não será carregado."
        )

    # Assinatura conferida: o arquivo é exatamente o que foi gerado pelo treino.
    return joblib.load(model_path)  # nosec B301
