# Modelo de preço v4: histórico simulado com ajuste fuzzy por imóvel

Gerado por `scripts_predict/modelo_v4.py`. Cada imóvel real ganha os anos
2016 a 2025 com o índice IPCA+INCC (média de 10 anos:
5.80% ao ano) mais um ajuste fuzzy próprio (`fuzzy_valorizacao.py`),
centralizado por tipo. Erros medidos só nos preços reais do teste.

| Tipo | Modelo escolhido | Linhas reais (treino) | Linhas simuladas | MAE | RMSE | R² | MdAPE |
|---|---|---:|---:|---:|---:|---:|---:|
| Casa | TabPFN 3.5 (API) (preco) | 1039 | 10390 | R$ 302.404 | R$ 449.918 | 0.540 | 18.6% |
| Apartamento | Ensemble (ensemble) | 1400 | 14000 | R$ 123.631 | R$ 177.385 | 0.779 | 12.8% |

## Valorização aprendida pelo modelo (imóveis de teste; não é medida de acerto)

| Tipo | Período | Valor no início / hoje (mediana) | Taxa anual p10 | mediana | p90 |
|---|---|---:|---:|---:|---:|
| Casa | 2016–2026 | 0.577 | 4.62% | 5.56% | 6.33% |
| Casa | 2021–2026 | 0.740 | 4.70% | 5.99% | 6.91% |
| Apartamento | 2016–2026 | 0.582 | 4.98% | 5.47% | 6.29% |
| Apartamento | 2021–2026 | 0.740 | 5.26% | 6.01% | 6.75% |

## Escolha do modelo (validação cruzada por imóvel, erro só nas linhas reais)

Mesmo critério da v3: menor MdAPE; empate (1 casa decimal) decidido pelo menor MAE.

### Casa

| Posição | Modelo | Alvo | MdAPE CV | MAE CV | No site |
|---:|---|---|---:|---:|---|
| 1 | TabPFN 3.5 (API) ★ (11.429 linhas de treino) | preco | 19.1% | R$ 283.971 | sim |
| 2 | Ensemble | ensemble | 19.5% | R$ 297.944 | sim |
| 3 | XGBoost | preco | 20.0% | R$ 305.067 | sim |
| 4 | Gradient Boosting | preco | 20.1% | R$ 304.548 | sim |
| 5 | Random Forest | preco | 20.2% | R$ 316.427 | sim |
| 6 | CatBoost | preco | 20.6% | R$ 297.564 | sim |
| 7 | Random Forest | log(preco) | 20.6% | R$ 353.112 | sim |
| 8 | CatBoost | log(preco) | 21.8% | R$ 355.671 | sim |
| 9 | XGBoost | log(preco) | 21.9% | R$ 372.197 | sim |
| 10 | Gradient Boosting | log(preco) | 22.8% | R$ 409.701 | sim |
| 11 | Ridge | log(preco) | 25.6% | R$ 383.536 | sim |
| 12 | Ridge | preco | 27.5% | R$ 342.477 | sim |

Reserva que roda no site: XGBoost (preco, peso 0.33), Gradient Boosting (preco, peso 0.33), Random Forest (preco, peso 0.33)

Tentativas do TabPFN (`tabpfn_v4.py`):

| Variante | Linhas de treino | Resultado | Tempo |
|---|---|---|---:|
| TabPFN 3.5 (API) | todas | terminou | 1 min |

### Apartamento

| Posição | Modelo | Alvo | MdAPE CV | MAE CV | No site |
|---:|---|---|---:|---:|---|
| 1 | Ensemble ★ | ensemble | 12.7% | R$ 126.131 | sim |
| 2 | Random Forest | preco | 13.0% | R$ 131.838 | sim |
| 3 | Random Forest | log(preco) | 13.0% | R$ 132.076 | sim |
| 4 | XGBoost | log(preco) | 13.3% | R$ 125.567 | sim |
| 5 | XGBoost | preco | 13.6% | R$ 126.818 | sim |
| 6 | TabPFN 3.5 (API) (15.400 linhas de treino) | preco | 13.9% | R$ 123.259 | sim |
| 7 | CatBoost | log(preco) | 14.0% | R$ 124.810 | sim |
| 8 | Gradient Boosting | log(preco) | 14.0% | R$ 126.501 | sim |
| 9 | CatBoost | preco | 14.4% | R$ 125.280 | sim |
| 10 | Gradient Boosting | preco | 14.6% | R$ 127.718 | sim |
| 11 | Ridge | log(preco) | 15.4% | R$ 143.061 | sim |
| 12 | Ridge | preco | 17.7% | R$ 149.612 | sim |

Reserva que roda no site: Random Forest (preco, peso 0.34), Random Forest (log(preco), peso 0.34), XGBoost (log(preco), peso 0.33)

Tentativas do TabPFN (`tabpfn_v4.py`):

| Variante | Linhas de treino | Resultado | Tempo |
|---|---|---|---:|
| TabPFN 3.5 (API) | todas | terminou | 1 min |

