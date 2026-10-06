"""TabPFN (v2, v3.5, v3.5-fast e 3.5-Thinking) como candidato nos benchmarks (opcional).

TabPFN é um "foundation model" para dados tabulares: um transformer pré-treinado
que faz a previsão por aprendizado em contexto (não há treino por gradiente nos
nossos dados). A versão v2 (Hollmann et al., Nature 2025) aceita até ~10.000
linhas de treino e ~500 features.

Instalação (opcional, puxa PyTorch):
    pip install -r requirements-tabpfn.txt

Na primeira execução os pesos da v2 (Prior-Labs/TabPFN-v2-reg) são baixados do
Hugging Face e ficam em cache. A v2 não exige aceite de licença; versões mais novas
(v2.5+) exigem, por isso fixamos explicitamente a v2.

Diferente dos outros modelos, o TabPFN recebe as features "cruas": numéricas sem
padronização e categóricas codificadas como inteiros (OrdinalEncoder), informando
ao modelo quais colunas são categóricas. One-hot de bairro prejudicaria o modelo.

Versões novas (variável TABPFN_VERSIONS, separadas por vírgula; padrão "v2"):
- "v3.5" e "v3.5-fast": rodam localmente com o pacote `tabpfn`, mas exigem aceitar a
  licença uma vez em https://ux.priorlabs.ai (o token fica em ~/.cache/tabpfn ou na
  variável TABPFN_TOKEN). Sem token, são puladas, para não abrir o navegador no meio
  de um benchmark.
- "thinking": TabPFN-3.5-Thinking, só pela API da Prior Labs (pacote `tabpfn-client`,
  pago por uso). Precisa de TABPFN_TOKEN. Os dados de treino são enviados ao servidor.
Exemplo: TABPFN_VERSIONS=v2,v3.5,v3.5-fast,thinking
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import OrdinalEncoder

TABPFN_NAME = "TabPFN v2"
TABPFN_MAX_TRAIN_ROWS = 10_000

# nome no ranking -> versão
TABPFN_VARIANTS = {
    "TabPFN v2": "v2",
    "TabPFN v3.5": "v3.5",
    "TabPFN v3.5 Fast": "v3.5-fast",
    "TabPFN 3.5 Thinking (API)": "thinking",
}

try:  # pragma: no cover - cliente da API, opcional
    from tabpfn_client import TabPFNRegressor as _TabPFNClientRegressor
except ImportError:  # pragma: no cover
    _TabPFNClientRegressor = None

try:  # pragma: no cover - dependência opcional
    from tabpfn import TabPFNRegressor as _TabPFNRegressor
except ImportError:  # pragma: no cover
    _TabPFNRegressor = None

try:  # pragma: no cover - só existe em versões recentes do pacote
    from tabpfn.constants import ModelVersion as _ModelVersion
except ImportError:  # pragma: no cover
    _ModelVersion = None


def tabpfn_available():
    return _TabPFNRegressor is not None


def tabpfn_enabled():
    """TabPFN entra nos benchmarks se estiver instalado e DISABLE_TABPFN não estiver definido."""
    return tabpfn_available() and not os.getenv("DISABLE_TABPFN")


def _has_prior_labs_token():
    """Token da Prior Labs já salvo (licença aceita) ou na variável TABPFN_TOKEN."""
    if os.getenv("TABPFN_TOKEN"):
        return True
    for path in (Path.home() / ".cache" / "tabpfn" / "auth_token", Path.home() / ".tabpfn" / "token"):
        if path.is_file() and path.read_text(encoding="utf-8", errors="ignore").strip():
            return True
    return False


def requested_versions():
    raw = os.getenv("TABPFN_VERSIONS", "v2")
    return [v.strip().lower() for v in raw.split(",") if v.strip()]


def enabled_tabpfn_variants(verbose=True):
    """Nomes das variantes do TabPFN que entram no benchmark nesta execução."""
    if os.getenv("DISABLE_TABPFN"):
        return []
    names = []
    for name, version in TABPFN_VARIANTS.items():
        if version not in requested_versions():
            continue
        if version == "thinking":
            ok = _TabPFNClientRegressor is not None and _has_prior_labs_token()
            motivo = "instale tabpfn-client e defina TABPFN_TOKEN"
        elif version == "v2":
            ok = tabpfn_available()
            motivo = "instale requirements-tabpfn.txt"
        else:
            ok = tabpfn_available() and _ModelVersion is not None and _has_prior_labs_token()
            motivo = "aceite a licença em https://ux.priorlabs.ai (token da Prior Labs)"
        if ok:
            names.append(name)
        elif verbose:
            print(f"[aviso] {name} pulado: {motivo}.")
    return names


def make_tabpfn(numeric_columns, categorical_columns, seed=42, output_type="mean", version="v2"):
    return TabPFNv2Regressor(
        numeric_columns=tuple(numeric_columns),
        categorical_columns=tuple(categorical_columns),
        random_state=seed,
        output_type=output_type,
        version=version,
    )


class TabPFNv2Regressor(RegressorMixin, BaseEstimator):
    """Wrapper scikit-learn para o TabPFN v2 com codificação própria das categóricas."""

    def __init__(
        self,
        numeric_columns=("area_total", "area_privada", "quartos", "banheiros", "vagas"),
        categorical_columns=("bairro", "tipo_imovel"),
        random_state=42,
        device="auto",
        n_estimators=8,
        max_train_rows=TABPFN_MAX_TRAIN_ROWS,
        output_type="mean",
        version="v2",
    ):
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.random_state = random_state
        self.device = device
        self.n_estimators = n_estimators
        self.max_train_rows = max_train_rows
        self.output_type = output_type  # "mean" ou "median" da distribuição preditiva
        self.version = version  # "v2", "v3.5", "v3.5-fast" ou "thinking"

    def _build_model(self, categorical_indices):
        if self.version == "thinking":
            if _TabPFNClientRegressor is None:
                raise ImportError("Instale o cliente da API: pip install tabpfn-client")
            # Thinking só existe na API (modelos v3); o servidor escolhe o modelo padrão.
            return _TabPFNClientRegressor(
                thinking_mode=True,
                categorical_features_indices=categorical_indices,
                random_state=self.random_state,
            )
        if _TabPFNRegressor is None:
            raise ImportError(
                "TabPFN não instalado. Rode: pip install -r requirements-tabpfn.txt"
            )
        options = {
            "device": self.device,
            "random_state": self.random_state,
            "n_estimators": self.n_estimators,
            "categorical_features_indices": categorical_indices,
            # O limite de linhas já é controlado por max_train_rows (subamostragem);
            # sem isto o TabPFN recusa > 1000 linhas quando roda em CPU.
            "ignore_pretraining_limits": True,
        }
        if _ModelVersion is not None:
            versions = {
                "v2": _ModelVersion.V2,
                "v3.5": _ModelVersion.V3_5,
                "v3.5-fast": _ModelVersion.V3_5_FAST,
            }
            return _TabPFNRegressor.create_default_for_version(versions[self.version], **options)
        # Versões 2.x do pacote já usam a v2 por padrão.
        return _TabPFNRegressor(**options)

    def _matrix(self, X):
        X = pd.DataFrame(X)
        numeric = X[list(self.numeric_columns)].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.fillna(0.0).to_numpy(dtype=float)
        if not self.categorical_columns:
            return numeric
        categorical = X[list(self.categorical_columns)].fillna("").astype(str)
        encoded = self.encoder_.transform(categorical).astype(float)
        return np.hstack([numeric, encoded])

    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = np.asarray(y, dtype=float)

        # A v2 foi pré-treinada com até ~10k linhas: acima disso, subamostra.
        if len(X) > self.max_train_rows:
            rng = np.random.default_rng(self.random_state)
            idx = np.sort(rng.choice(len(X), size=self.max_train_rows, replace=False))
            X = X.iloc[idx].reset_index(drop=True)
            y = y[idx]

        self.encoder_ = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
        if self.categorical_columns:
            self.encoder_.fit(X[list(self.categorical_columns)].fillna("").astype(str))

        n_num = len(self.numeric_columns)
        categorical_indices = list(range(n_num, n_num + len(self.categorical_columns)))
        self.model_ = self._build_model(categorical_indices)
        try:
            self.model_.fit(self._matrix(X), y)
        except RuntimeError as exc:
            if "download" in str(exc).lower():
                raise RuntimeError(
                    "Não foi possível baixar os pesos do TabPFN v2 (Hugging Face). "
                    "Verifique a internet ou rode com DISABLE_TABPFN=1 para pular o TabPFN."
                ) from exc
            raise
        self.n_train_rows_ = int(len(X))
        return self

    def predict(self, X):
        matrix = self._matrix(X)
        if self.output_type == "mean":
            return np.asarray(self.model_.predict(matrix), dtype=float)
        return np.asarray(self.model_.predict(matrix, output_type=self.output_type), dtype=float)
