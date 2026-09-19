# Tuning Casas/Apartamentos

- base final: 2864 imóveis após filtro e remoção de outliers
- treino: 2291
- teste: 573
- seed: 42

| Modelo | Alvo | MAE | RMSE | R² | MAPE |
|---|---|---:|---:|---:|---:|
| Random Forest | preco | R$ 202.341,60 | R$ 297.276,45 | 0.7084 | 23.89% |
| Gradient Boosting | preco | R$ 204.591,42 | R$ 298.210,04 | 0.7066 | 24.51% |
| Gradient Boosting | log(preco) | R$ 206.473,19 | R$ 319.506,14 | 0.6632 | 23.14% |
| Random Forest | log(preco) | R$ 212.268,84 | R$ 331.248,54 | 0.6380 | 23.91% |
| Ridge | preco | R$ 236.743,82 | R$ 329.466,71 | 0.6419 | 30.16% |
| Lasso | preco | R$ 237.381,42 | R$ 331.407,38 | 0.6377 | 30.52% |
| Lasso | log(preco) | R$ 261.138,63 | R$ 398.277,25 | 0.4767 | 28.81% |
| Ridge | log(preco) | R$ 265.100,97 | R$ 405.984,95 | 0.4562 | 28.84% |

Melhor combinação por MAE: **Random Forest (preco)**

## Próximo uso

- Se o objetivo for previsibilidade, use o melhor modelo dessa tabela.
- Se o objetivo for estabilidade, compare MAE e R² juntos.