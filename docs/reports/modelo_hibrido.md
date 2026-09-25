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
| Típico (sem outliers) | R$ 199.440,50 | R$ 312.885,20 | 0.6752 | 485148.12% |
| Completo (com outliers) | R$ 293.258,47 | R$ 612.704,83 | 0.4508 | 578152.76% |

## Segmento: Casa

- Registros no modelo final: 1299
- Treino (após limpeza): 1042 | Teste típico: 256 | Teste completo: 278
- Validação cruzada: 5 folds

| Conjunto de teste | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|
| Típico (sem outliers) | R$ 301.041,07 | R$ 437.489,33 | 0.5608 | 1161677.79% |
| Completo (com outliers) | R$ 443.109,84 | R$ 852.712,78 | 0.3219 | 1069750.80% |

### Modelos escolhidos no ensemble

| Modelo | Alvo | Peso | MAE validação cruzada | MAE no teste típico | R² no teste típico |
|---|---|---:|---:|---:|---:|
| TabPFN v2 | preco | 0.343 | R$ 275.807,22 | R$ 306.696,77 | 0.5358 |
| TabPFN v2 | log(preco) | 0.329 | R$ 287.537,83 | R$ 314.543,80 | 0.5183 |
| CatBoost | preco | 0.328 | R$ 288.015,83 | R$ 299.490,59 | 0.5597 |

### Top 10 do ranking (validação cruzada no treino)

| Modelo | Alvo | MAE | RMSE | R² | MAPE |
|---|---|---:|---:|---:|---:|
| TabPFN v2 | preco | R$ 275.807,22 | R$ 391.049,98 | 0.6435 | 744745.42% |
| TabPFN v2 | log(preco) | R$ 287.537,83 | R$ 421.982,24 | 0.5848 | 592344.10% |
| CatBoost | preco | R$ 288.015,83 | R$ 407.372,31 | 0.6131 | 686993.44% |
| XGBoost | preco | R$ 290.768,59 | R$ 414.055,62 | 0.6003 | 672366.94% |
| Random Forest | preco | R$ 292.331,97 | R$ 413.429,43 | 0.6015 | 701029.75% |
| Gradient Boosting | preco | R$ 295.709,77 | R$ 423.405,92 | 0.5820 | 709739.18% |
| Ridge | preco | R$ 316.012,09 | R$ 436.182,91 | 0.5564 | 809934.29% |
| Lasso | preco | R$ 317.535,23 | R$ 437.851,16 | 0.5530 | 814590.59% |
| Regressão Linear Múltipla | preco | R$ 317.535,23 | R$ 437.851,16 | 0.5530 | 814590.59% |
| CatBoost | log(preco) | R$ 331.011,33 | R$ 497.139,76 | 0.4238 | 631015.41% |

## Segmento: Apartamento

- Registros no modelo final: 1757
- Treino (após limpeza): 1401 | Teste típico: 357 | Teste completo: 380
- Validação cruzada: 5 folds

| Conjunto de teste | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|
| Típico (sem outliers) | R$ 126.584,07 | R$ 175.640,80 | 0.7756 | 17.60% |
| Completo (com outliers) | R$ 183.630,37 | R$ 343.660,03 | 0.5990 | 218509.99% |

### Modelos escolhidos no ensemble

| Modelo | Alvo | Peso | MAE validação cruzada | MAE no teste típico | R² no teste típico |
|---|---|---:|---:|---:|---:|
| TabPFN v2 | log(preco) | 0.335 | R$ 119.323,96 | R$ 127.090,55 | 0.7723 |
| TabPFN v2 | preco | 0.333 | R$ 120.044,04 | R$ 127.108,80 | 0.7761 |
| CatBoost | log(preco) | 0.332 | R$ 120.646,77 | R$ 126.827,03 | 0.7699 |

### Top 10 do ranking (validação cruzada no treino)

| Modelo | Alvo | MAE | RMSE | R² | MAPE |
|---|---|---:|---:|---:|---:|
| TabPFN v2 | log(preco) | R$ 119.323,96 | R$ 169.941,94 | 0.7792 | 17.05% |
| TabPFN v2 | preco | R$ 120.044,04 | R$ 169.143,21 | 0.7813 | 17.57% |
| CatBoost | log(preco) | R$ 120.646,77 | R$ 170.061,93 | 0.7789 | 17.32% |
| Gradient Boosting | log(preco) | R$ 122.060,85 | R$ 174.583,95 | 0.7670 | 17.48% |
| CatBoost | preco | R$ 122.398,48 | R$ 170.889,60 | 0.7767 | 18.20% |
| XGBoost | log(preco) | R$ 123.380,89 | R$ 178.996,44 | 0.7551 | 17.68% |
| XGBoost | preco | R$ 124.502,41 | R$ 181.059,43 | 0.7494 | 18.20% |
| Gradient Boosting | preco | R$ 124.699,02 | R$ 175.940,72 | 0.7634 | 18.40% |
| Random Forest | preco | R$ 128.662,27 | R$ 182.954,53 | 0.7441 | 19.12% |
| Random Forest | log(preco) | R$ 129.157,72 | R$ 184.413,31 | 0.7400 | 18.74% |
