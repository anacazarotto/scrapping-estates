"""Compatibilidade do CatBoost com o scikit-learn >= 1.6.

O CatBoost 1.2.x não implementa `__sklearn_tags__`, exigido pelo scikit-learn novo
quando o modelo está dentro de um Pipeline (erro "'CatBoostRegressor' object has
no attribute '__sklearn_tags__'"). Este wrapper herda de BaseEstimator e delega o
treino/previsão ao CatBoost original.

Uso: `from sklearn_compat import CatBoostRegressor` (ImportError se o catboost
não estiver instalado, igual ao import original).
"""

from catboost import CatBoostRegressor as _CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin


class CatBoostRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
    ):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.random_seed = random_seed
        self.verbose = verbose

    def fit(self, X, y):
        self.model_ = _CatBoostRegressor(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            loss_function=self.loss_function,
            random_seed=self.random_seed,
            verbose=self.verbose,
            allow_writing_files=False,  # não cria a pasta catboost_info/
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)
