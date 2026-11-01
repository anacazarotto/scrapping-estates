# Modelo de preço v3: anúncios reais + histórico simulado (IPCA+INCC)

Gerado por `scripts_predict/modelo_v3.py`. Para cada imóvel real foram simulados os
anos 2016 a 2025, deflacionando o preço pelo índice
IPCA+INCC realizado (média de 10 anos: 5.80% ao ano). Erros
medidos só nos preços reais do teste, que o modelo nunca viu.

| Tipo | Linhas reais (treino) | Linhas simuladas (treino) | MAE v3 | MAE v1 | R² v3 | R² v1 | MdAPE v3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Casa | 1039 | 10390 | R$ 302.336 | R$ 291.216 | 0.540 | 0.559 | 18.7% |
| Apartamento | 1400 | 14000 | R$ 123.858 | R$ 119.640 | 0.781 | 0.795 | 13.1% |

## Coerência com o índice (não é medida de acerto)

Quanto o modelo diz que os imóveis de teste valiam em jun/2021, como fração do valor
de hoje (mediana), e quanto o índice diria:

| Tipo | Modelo | Índice IPCA+INCC |
|---|---:|---:|
| Casa | 0.744 | 0.735 |
| Apartamento | 0.741 | 0.735 |

## Escolha do modelo (validação cruzada por imóvel, erro só nas linhas reais)

Critério: menor erro percentual mediano (MdAPE). Empate (mesmo MdAPE com 1 casa
decimal) é decidido pelo menor MAE. O ensemble (média ponderada dos 3 melhores que
rodam no site, peso 1/MdAPE) concorre como mais um candidato. O TabPFN local entra
no ranking, mas não pode ser publicado (PyTorch, minutos por previsão em CPU); o
TabPFN 3.5 pela API da Prior Labs pode. Se a API não responder, o site usa a reserva:
o melhor entre os que rodam no próprio site.

### Casa

| Posição | Modelo | Alvo | MdAPE CV | MAE CV | No site |
|---:|---|---|---:|---:|---|
| 1 | TabPFN v3.5 (3.000 linhas de treino) | preco | 18.6% | R$ 279.870 | não |
| 2 | TabPFN 3.5 (API) ★ (11.429 linhas de treino) | preco | 19.2% | R$ 284.992 | sim |
| 3 | Ensemble | ensemble | 19.6% | R$ 297.174 | sim |
| 4 | Random Forest | preco | 19.6% | R$ 315.031 | sim |
| 5 | TabPFN v3.5 Fast (11.429 linhas de treino) | preco | 19.8% | R$ 289.664 | não |
| 6 | CatBoost | preco | 20.2% | R$ 297.819 | sim |
| 7 | XGBoost | preco | 20.2% | R$ 305.465 | sim |
| 8 | TabPFN v2 (3.000 linhas de treino) | preco | 20.3% | R$ 306.908 | não |
| 9 | Random Forest | log(preco) | 20.4% | R$ 349.864 | sim |
| 10 | Gradient Boosting | preco | 20.7% | R$ 304.471 | sim |
| 11 | XGBoost | log(preco) | 21.6% | R$ 367.254 | sim |
| 12 | CatBoost | log(preco) | 21.8% | R$ 356.432 | sim |
| 13 | Gradient Boosting | log(preco) | 22.3% | R$ 386.933 | sim |
| 14 | Ridge | log(preco) | 26.1% | R$ 385.691 | sim |
| 15 | Ridge | preco | 27.8% | R$ 345.834 | sim |

Publicado (★): TabPFN 3.5 (API), com reserva Ensemble.

Roda no site (★ ou reserva): Random Forest (preco, peso 0.34), CatBoost (preco, peso 0.33), XGBoost (preco, peso 0.33)

Tentativas do TabPFN (`tabpfn_v3.py`):

| Variante | Linhas de treino | Resultado | Tempo |
|---|---|---|---:|
| TabPFN v2 | todas | cortado por tempo: não terminou em 60.0 minutos | 60 min |
| TabPFN v2 | 10000 | pulado: cada parte da validação já treina com 9143 linhas | 0 min |
| TabPFN v2 | 3000 | terminou | 19 min |
| TabPFN v3.5 | todas | cortado por tempo: não terminou em 60.0 minutos | 60 min |
| TabPFN v3.5 | 10000 | pulado: cada parte da validação já treina com 9143 linhas | 0 min |
| TabPFN v3.5 | 3000 | terminou | 18 min |
| TabPFN v3.5 Fast | todas | terminou | 34 min |
| TabPFN 3.5 (API) | todas | terminou | 1 min |

### Apartamento

| Posição | Modelo | Alvo | MdAPE CV | MAE CV | No site |
|---:|---|---|---:|---:|---|
| 1 | Ensemble ★ | ensemble | 12.9% | R$ 126.460 | sim |
| 2 | Random Forest | preco | 13.1% | R$ 131.895 | sim |
| 3 | Random Forest | log(preco) | 13.1% | R$ 132.009 | sim |
| 4 | TabPFN v3.5 (3.000 linhas de treino) | preco | 13.4% | R$ 120.189 | não |
| 5 | XGBoost | log(preco) | 13.5% | R$ 125.746 | sim |
| 6 | XGBoost | preco | 13.5% | R$ 125.922 | sim |
| 7 | CatBoost | log(preco) | 13.6% | R$ 124.860 | sim |
| 8 | Gradient Boosting | log(preco) | 13.7% | R$ 126.509 | sim |
| 9 | TabPFN 3.5 (API) (15.400 linhas de treino) | preco | 13.9% | R$ 123.427 | sim |
| 10 | TabPFN v2 (3.000 linhas de treino) | preco | 14.0% | R$ 126.624 | não |
| 11 | CatBoost | preco | 14.3% | R$ 126.069 | sim |
| 12 | Gradient Boosting | preco | 14.6% | R$ 128.977 | sim |
| 13 | TabPFN v3.5 Fast (15.400 linhas de treino) | preco | 14.6% | R$ 129.842 | não |
| 14 | Ridge | log(preco) | 15.4% | R$ 142.028 | sim |
| 15 | Ridge | preco | 18.2% | R$ 152.500 | sim |

Roda no site (★ ou reserva): Random Forest (preco, peso 0.34), Random Forest (log(preco), peso 0.34), XGBoost (log(preco), peso 0.33)

Tentativas do TabPFN (`tabpfn_v3.py`):

| Variante | Linhas de treino | Resultado | Tempo |
|---|---|---|---:|
| TabPFN v2 | todas | cortado por tempo: não terminou em 60.0 minutos | 60 min |
| TabPFN v2 | 10000 | cortado por tempo: não terminou em 60.0 minutos | 60 min |
| TabPFN v2 | 3000 | terminou | 17 min |
| TabPFN v3.5 | todas | cortado por tempo: não terminou em 60.0 minutos | 60 min |
| TabPFN v3.5 | 10000 | cortado por tempo: não terminou em 60.0 minutos | 60 min |
| TabPFN v3.5 | 3000 | terminou | 18 min |
| TabPFN v3.5 Fast | todas | terminou | 55 min |
| TabPFN 3.5 (API) | todas | terminou | 1 min |

