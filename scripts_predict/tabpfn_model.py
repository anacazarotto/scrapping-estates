"""TabPFN v2 como candidato nos benchmarks (dependência opcional).

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
"""

import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import OrdinalEncoder

TABPFN_NAME = "TabPFN v2"
TABPFN_MAX_TRAIN_ROWS = 10_000

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


def make_tabpfn(numeric_columns, categorical_columns, seed=42, output_type="mean"):
    return TabPFNv2Regressor(
        numeric_columns=tuple(numeric_columns),
        categorical_columns=tuple(categorical_columns),
        random_state=seed,
        output_type=output_type,
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
    ):
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.random_state = random_state
        self.device = device
        self.n_estimators = n_estimators
        self.max_train_rows = max_train_rows
        self.output_type = output_type  # "mean" ou "median" da distribuição preditiva

    def _build_model(self, categorical_indices):
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
            return _TabPFNRegressor.create_default_for_version(_ModelVersion.V2, **options)
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
