# Modelo de preço v3: anúncios reais + histórico simulado (IPCA+INCC)

Gerado por `scripts_predict/modelo_v3.py`. Para cada imóvel real foram simulados os
anos 2016 a 2025, deflacionando o preço pelo índice
IPCA+INCC realizado (média de 10 anos: 5.80% ao ano). Erros
medidos só nos preços reais do teste, que o modelo nunca viu.

| Tipo | Linhas reais (treino) | Linhas simuladas (treino) | MAE v3 | MAE v1 | R² v3 | R² v1 | Erro mediano v3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Casa | 1038 | 10380 | R$ 290.535 | R$ 291.216 | 0.571 | 0.559 | 19.2% |
| Apartamento | 1400 | 14000 | R$ 118.600 | R$ 119.640 | 0.800 | 0.795 | 15.0% |

## Coerência com o índice (não é medida de acerto)

Quanto o modelo diz que os imóveis de teste valiam em jun/2021, como fração do valor
de hoje (mediana), e quanto o índice diria:

| Tipo | Modelo | Índice IPCA+INCC |
|---|---:|---:|
| Casa | 0.755 | 0.754 |
| Apartamento | 0.757 | 0.754 |

## Ensemble escolhido (validação cruzada por imóvel, erro só nas linhas reais)

- **Casa:** CatBoost (preco, peso 0.34), XGBoost (preco, peso 0.33), Gradient Boosting (preco, peso 0.33)
- **Apartamento:** CatBoost (log(preco), peso 0.34), XGBoost (log(preco), peso 0.33), CatBoost (preco, peso 0.33)
