# Relatório do algoritmo híbrido

Este algoritmo combina os aprendizados do baseline:
- segmentação por **Casa** e **Apartamento**;
- agrupamento de bairros raros;
- clipagem de variáveis numéricas;
- remoção de outliers por IQR;
- teste de alvo em `preco` e `log(preco)`.

Depois disso, ele junta os melhores modelos em um ensemble ponderado por MAE.

## Resultado geral (ensemble híbrido)

- MAE: R$ 146.412,39
- RMSE: R$ 223.825,41
- R²: 0.8526
- MAPE: 498343.85%

## Segmento: Casa

- Registros usados: 1254
- MAE do ensemble: R$ 224.729,22
- RMSE do ensemble: R$ 308.751,39
- R² do ensemble: 0.7950
- MAPE do ensemble: 1141608.29%

### Modelos escolhidos no ensemble

| Modelo | Alvo | Peso | MAE validação |
|---|---|---:|---:|
| Gradient Boosting | preco | 0.341 | R$ 324.317,95 |
| Random Forest | preco | 0.340 | R$ 325.314,57 |
| Regressão Linear Múltipla | preco | 0.319 | R$ 347.490,94 |

### Top 10 do benchmark do segmento

| Modelo | Alvo | MAE | RMSE | R² | MAPE |
|---|---|---:|---:|---:|---:|
| Gradient Boosting | preco | R$ 324.317,95 | R$ 455.139,08 | 0.5546 | 1585321.97% |
| Random Forest | preco | R$ 325.314,57 | R$ 468.487,51 | 0.5281 | 1937252.71% |
| Regressão Linear Múltipla | preco | R$ 347.490,94 | R$ 492.647,75 | 0.4782 | 1918714.85% |
| Lasso | preco | R$ 347.490,94 | R$ 492.647,75 | 0.4782 | 1918714.86% |
| Random Forest | log(preco) | R$ 348.430,47 | R$ 502.832,75 | 0.4564 | 1263956.60% |
| Ridge | preco | R$ 350.082,77 | R$ 498.281,56 | 0.4662 | 1987101.99% |
| Gradient Boosting | log(preco) | R$ 360.890,74 | R$ 521.593,71 | 0.4151 | 1655571.74% |
| Máquinas de Vetores de Suporte (SVM) | log(preco) | R$ 376.139,74 | R$ 548.765,88 | 0.3525 | 1516123.55% |
| Ridge | log(preco) | R$ 409.780,34 | R$ 583.326,65 | 0.2684 | 1444311.65% |
| Regressão Linear Múltipla | log(preco) | R$ 410.516,66 | R$ 583.964,73 | 0.2668 | 1413752.86% |

## Segmento: Apartamento

- Registros usados: 1620
- MAE do ensemble: R$ 85.741,01
- RMSE do ensemble: R$ 122.714,43
- R² do ensemble: 0.8959
- MAPE do ensemble: 12.44%

### Modelos escolhidos no ensemble

| Modelo | Alvo | Peso | MAE validação |
|---|---|---:|---:|
| Random Forest | log(preco) | 0.334 | R$ 143.044,92 |
| Random Forest | preco | 0.334 | R$ 143.302,70 |
| Gradient Boosting | log(preco) | 0.332 | R$ 143.786,18 |

### Top 10 do benchmark do segmento

| Modelo | Alvo | MAE | RMSE | R² | MAPE |
|---|---|---:|---:|---:|---:|
| Random Forest | log(preco) | R$ 143.044,92 | R$ 202.301,90 | 0.7172 | 20.89% |
| Random Forest | preco | R$ 143.302,70 | R$ 200.206,74 | 0.7230 | 21.47% |
| Gradient Boosting | log(preco) | R$ 143.786,18 | R$ 204.232,02 | 0.7118 | 20.49% |
| Gradient Boosting | preco | R$ 145.540,93 | R$ 207.426,35 | 0.7027 | 21.27% |
| Ridge | preco | R$ 148.403,91 | R$ 211.824,70 | 0.6899 | 22.16% |
| Lasso | preco | R$ 148.586,02 | R$ 211.762,18 | 0.6901 | 22.16% |
| Regressão Linear Múltipla | preco | R$ 148.586,02 | R$ 211.762,18 | 0.6901 | 22.16% |
| Regressão Linear Múltipla | log(preco) | R$ 150.281,15 | R$ 227.095,93 | 0.6436 | 21.32% |
| Lasso | log(preco) | R$ 150.704,33 | R$ 228.112,57 | 0.6404 | 21.40% |
| Ridge | log(preco) | R$ 150.870,94 | R$ 229.009,44 | 0.6376 | 21.38% |
