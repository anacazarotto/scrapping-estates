# Experimento v2 do modelo de preço

Gerado por `scripts_predict/experimento_preco_v2.py`. Mesmo protocolo do v1: teste
separado antes da limpeza (semente 42), regras só do treino, ranking por validação
cruzada de 5 partes, ensemble dos 3 melhores com peso 1/MAE, teste típico.

Correção pelo IPCA: fator entre 0.9975 e 1.0116 (preços de mar a
ago/2026 levados a ago/2026).

## Casa

| Variante | MAE no teste | Diferença para o v1 | R² | Erro mediano | Ensemble |
|---|---:|---:|---:|---:|---|
| v1 | R$ 291.216,39 | +0.0% | 0.559 | 19.6% | CatBoost, XGBoost, Random Forest |
| bairro | R$ 291.375,79 | +0.1% | 0.575 | 19.4% | CatBoost, Random Forest, XGBoost |
| m2 | R$ 289.732,19 | -0.5% | 0.550 | 19.3% | CatBoost, XGBoost, Random Forest |
| bairro+m2 | R$ 297.547,34 | +2.2% | 0.540 | 18.2% | CatBoost, Random Forest, XGBoost |
| ipca | R$ 292.025,73 | +0.3% | 0.558 | 19.4% | CatBoost, XGBoost, Random Forest |

## Apartamento

| Variante | MAE no teste | Diferença para o v1 | R² | Erro mediano | Ensemble |
|---|---:|---:|---:|---:|---|
| v1 | R$ 119.639,89 | +0.0% | 0.795 | 14.8% | CatBoost (log), XGBoost (log), CatBoost |
| bairro | R$ 119.468,21 | -0.1% | 0.796 | 15.2% | CatBoost (log), CatBoost, XGBoost (log) |
| m2 | R$ 119.781,37 | +0.1% | 0.795 | 14.3% | Gradient Boosting (log), CatBoost (log), Gradient Boosting |
| bairro+m2 | R$ 120.152,13 | +0.4% | 0.801 | 14.6% | Gradient Boosting (log), CatBoost, CatBoost (log) |
| ipca | R$ 119.984,70 | +0.3% | 0.792 | 14.1% | CatBoost (log), Gradient Boosting (log), XGBoost (log) |
