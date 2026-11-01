# Modelo de preço v3: anúncios reais + histórico simulado (IPCA+INCC)

Gerado por `scripts_predict/modelo_v3.py`. Para cada imóvel real foram simulados os
anos 2016 a 2025, deflacionando o preço pelo índice
IPCA+INCC realizado (média de 10 anos: 5.80% ao ano). Erros
medidos só nos preços reais do teste, que o modelo nunca viu.

| Tipo | Linhas reais (treino) | Linhas simuladas (treino) | MAE v3 | MAE v1 | R² v3 | R² v1 | MdAPE v3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Casa | 1039 | 10390 | R$ 296.426 | R$ 291.216 | 0.576 | 0.559 | 19.2% |
| Apartamento | 1400 | 14000 | R$ 121.532 | R$ 119.640 | 0.799 | 0.795 | 14.5% |

## Coerência com o índice (não é medida de acerto)

Quanto o modelo diz que os imóveis de teste valiam em jun/2021, como fração do valor
de hoje (mediana), e quanto o índice diria:

| Tipo | Modelo | Índice IPCA+INCC |
|---|---:|---:|
| Casa | 0.737 | 0.735 |
| Apartamento | 0.736 | 0.735 |

## Escolha do modelo (validação cruzada por imóvel, erro só nas linhas reais)

Critério: menor erro percentual mediano (MdAPE). Empate (mesmo MdAPE com 1 casa
decimal) é decidido pelo menor MAE. O ensemble (média ponderada dos 3 melhores que
cabem no site, peso 1/MdAPE) concorre como mais um candidato. O TabPFN entra no
ranking, mas não pode ser publicado: precisa de PyTorch e leva minutos por previsão
em CPU, o que o site gratuito não comporta.

### Casa

| Posição | Modelo | Alvo | MdAPE CV | MAE CV | No site |
|---:|---|---|---:|---:|---|
| 1 | TabPFN v3.5 (3.000 linhas de treino) | preco | 18.6% | R$ 279.870 | não |
| 2 | TabPFN v3.5 Fast (11.429 linhas de treino) | preco | 19.8% | R$ 289.664 | não |
| 3 | Ensemble ★ | ensemble | 20.1% | R$ 297.391 | sim |
| 4 | CatBoost | preco | 20.2% | R$ 297.819 | sim |
| 5 | XGBoost | preco | 20.2% | R$ 305.465 | sim |
| 6 | TabPFN v2 (3.000 linhas de treino) | preco | 20.3% | R$ 306.908 | não |
| 7 | Gradient Boosting | preco | 20.7% | R$ 304.471 | sim |
| 8 | XGBoost | log(preco) | 21.6% | R$ 367.254 | sim |
| 9 | CatBoost | log(preco) | 21.8% | R$ 356.432 | sim |
| 10 | Gradient Boosting | log(preco) | 22.3% | R$ 386.933 | sim |
| 11 | Ridge | log(preco) | 26.1% | R$ 385.691 | sim |
| 12 | Ridge | preco | 27.8% | R$ 345.834 | sim |

Publicado (★): CatBoost (preco, peso 0.34), XGBoost (preco, peso 0.34), Gradient Boosting (preco, peso 0.33)

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

### Apartamento

| Posição | Modelo | Alvo | MdAPE CV | MAE CV | No site |
|---:|---|---|---:|---:|---|
| 1 | Ensemble ★ | ensemble | 13.3% | R$ 123.587 | sim |
| 2 | TabPFN v3.5 (3.000 linhas de treino) | preco | 13.4% | R$ 120.189 | não |
| 3 | XGBoost | log(preco) | 13.5% | R$ 125.746 | sim |
| 4 | XGBoost | preco | 13.5% | R$ 125.922 | sim |
| 5 | CatBoost | log(preco) | 13.6% | R$ 124.860 | sim |
| 6 | Gradient Boosting | log(preco) | 13.7% | R$ 126.509 | sim |
| 7 | TabPFN v2 (3.000 linhas de treino) | preco | 14.0% | R$ 126.624 | não |
| 8 | CatBoost | preco | 14.3% | R$ 126.069 | sim |
| 9 | Gradient Boosting | preco | 14.6% | R$ 128.977 | sim |
| 10 | TabPFN v3.5 Fast (15.400 linhas de treino) | preco | 14.6% | R$ 129.842 | não |
| 11 | Ridge | log(preco) | 15.4% | R$ 142.028 | sim |
| 12 | Ridge | preco | 18.2% | R$ 152.500 | sim |

Publicado (★): XGBoost (log(preco), peso 0.33), XGBoost (preco, peso 0.33), CatBoost (log(preco), peso 0.33)

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

