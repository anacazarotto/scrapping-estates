# Relatório do algoritmo híbrido

Este algoritmo combina os aprendizados do baseline:
- segmentação por **Casa** e **Apartamento**;
- agrupamento de bairros raros;
- clipagem de variáveis numéricas;
- remoção de outliers por IQR (somente no treino);
- teste de alvo em `preco` e `log(preco)`.

Depois disso, ele junta os melhores modelos em um ensemble ponderado por MAE.

## Como as métricas foram medidas

- O conjunto de teste é separado **antes** de qualquer pré-processamento e nunca é usado
  para escolher modelos, pesos, bairros raros, limites de clipagem ou limites de outlier.
- Os candidatos são ranqueados por **validação cruzada dentro do treino**.
- O ensemble usado para medir desempenho é treinado **somente com o treino**.
  O modelo final (salvo para previsão) é retreinado depois com todos os dados.
- **Teste típico**: imóveis do teste dentro da faixa normal de preço (sem outliers).
- **Teste completo**: todos os imóveis do teste, inclusive anúncios com preço atípico.

## Resultado geral (ensemble híbrido)

| Conjunto de teste | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|
| Típico (sem outliers) | R$ 194.535,59 | R$ 307.015,00 | 0.6787 | 492525.28% |
| Completo (com outliers) | R$ 275.989,72 | R$ 574.982,01 | 0.5026 | 597547.60% |

## Segmento: Casa

- Registros no modelo final: 1298
- Treino (após limpeza): 1038 | Teste típico: 261 | Teste completo: 278
- Validação cruzada: 5 folds

| Conjunto de teste | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|
| Típico (sem outliers) | R$ 293.975,46 | R$ 428.181,98 | 0.5642 | 1166185.31% |
| Completo (com outliers) | R$ 409.960,63 | R$ 792.007,98 | 0.3960 | 1094874.85% |

### Modelos escolhidos no ensemble

| Modelo | Alvo | Peso | MAE validação cruzada | MAE no teste típico | R² no teste típico |
|---|---|---:|---:|---:|---:|
| TabPFN v2 | preco | 0.339 | R$ 279.395,26 | R$ 299.681,14 | 0.5414 |
| CatBoost | preco | 0.331 | R$ 286.418,39 | R$ 286.088,76 | 0.5878 |
| TabPFN v2 | log(preco) | 0.330 | R$ 286.887,04 | R$ 305.772,57 | 0.5200 |

### Top 10 do ranking (validação cruzada no treino)

| Modelo | Alvo | MAE | RMSE | R² | MAPE |
|---|---|---:|---:|---:|---:|
| TabPFN v2 | preco | R$ 279.395,26 | R$ 394.447,06 | 0.6400 | 765289.62% |
| CatBoost | preco | R$ 286.418,39 | R$ 402.538,19 | 0.6251 | 707133.22% |
| TabPFN v2 | log(preco) | R$ 286.887,04 | R$ 413.712,09 | 0.6040 | 677704.50% |
| XGBoost | preco | R$ 289.136,28 | R$ 404.462,12 | 0.6215 | 654844.38% |
| Random Forest | preco | R$ 291.506,90 | R$ 407.882,32 | 0.6151 | 721661.00% |
| Gradient Boosting | preco | R$ 295.209,94 | R$ 413.142,35 | 0.6051 | 707483.96% |
| Ridge | preco | R$ 318.028,42 | R$ 438.106,87 | 0.5559 | 816273.34% |
| Lasso | preco | R$ 319.856,39 | R$ 440.126,84 | 0.5518 | 821542.21% |
| Regressão Linear Múltipla | preco | R$ 319.856,39 | R$ 440.126,85 | 0.5518 | 821542.22% |
| CatBoost | log(preco) | R$ 330.795,92 | R$ 496.859,28 | 0.4288 | 713984.72% |

## Segmento: Apartamento

- Registros no modelo final: 1756
- Treino (após limpeza): 1400 | Teste típico: 357 | Teste completo: 380
- Validação cruzada: 5 folds

| Conjunto de teste | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|
| Típico (sem outliers) | R$ 121.835,85 | R$ 170.679,01 | 0.7854 | 17.53% |
| Completo (com outliers) | R$ 177.979,43 | R$ 336.994,20 | 0.6200 | 233713.46% |

### Modelos escolhidos no ensemble

| Modelo | Alvo | Peso | MAE validação cruzada | MAE no teste típico | R² no teste típico |
|---|---|---:|---:|---:|---:|
| TabPFN v2 | log(preco) | 0.335 | R$ 120.416,79 | R$ 122.096,80 | 0.7819 |
| TabPFN v2 | preco | 0.334 | R$ 120.874,38 | R$ 123.126,50 | 0.7838 |
| CatBoost | log(preco) | 0.330 | R$ 122.256,19 | R$ 122.162,44 | 0.7827 |

### Top 10 do ranking (validação cruzada no treino)

| Modelo | Alvo | MAE | RMSE | R² | MAPE |
|---|---|---:|---:|---:|---:|
| TabPFN v2 | log(preco) | R$ 120.416,79 | R$ 170.133,09 | 0.7812 | 17.18% |
| TabPFN v2 | preco | R$ 120.874,38 | R$ 169.478,70 | 0.7829 | 17.68% |
| CatBoost | log(preco) | R$ 122.256,19 | R$ 171.651,64 | 0.7773 | 17.58% |
| XGBoost | log(preco) | R$ 123.762,11 | R$ 176.833,46 | 0.7636 | 17.92% |
| CatBoost | preco | R$ 124.069,93 | R$ 172.841,85 | 0.7742 | 18.39% |
| Gradient Boosting | log(preco) | R$ 124.261,19 | R$ 175.840,84 | 0.7663 | 17.75% |
| Gradient Boosting | preco | R$ 124.476,49 | R$ 174.862,17 | 0.7689 | 18.49% |
| XGBoost | preco | R$ 125.568,07 | R$ 179.088,10 | 0.7576 | 18.52% |
| Random Forest | preco | R$ 127.820,76 | R$ 181.826,51 | 0.7501 | 19.14% |
| Random Forest | log(preco) | R$ 127.917,23 | R$ 182.510,43 | 0.7482 | 18.71% |
