# Experimentos Casas/Apartamentos

Base do teste: apenas imóveis classificados como **Casa** ou **Apartamento**.

## Cenários

| Cenário | N | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|---:|
| 1. Casas/Apartamentos - modelo único | 3079 | R$ 1.068.612,62 | R$ 2.217.219,06 | -4.9272 | 947420.53% |
| 2. Casas/Apartamentos - modelos separados | 3079 | R$ 1.068.815,17 | R$ 2.739.778,84 | -6.1296 | 371838.14% |
| 3. Casas/Apartamentos - sem outliers | 2864 | R$ 236.969,00 | R$ 330.277,83 | 0.6401 | 30.10% |
| 4. Casas/Apartamentos - sem outliers + log(preco) | 2864 | R$ 272.057,27 | R$ 412.478,32 | 0.4387 | 29.65% |

## Modelos separados

| Tipo | N | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|---:|
| Casa (modelo separado) | 1342 | R$ 1.755.735,86 | R$ 3.897.508,65 | -7.9500 | 852782.76% |
| Apartamento (modelo separado) | 1737 | R$ 537.833,37 | R$ 1.251.639,29 | -2.4243 | 73.48% |

## Leitura

- O cenário 1 mede o ganho bruto ao reduzir a variedade de tipos.
- O cenário 2 testa se separar casa e apartamento melhora o ajuste.
- O cenário 3 avalia a remoção de outliers.
- O cenário 4 avalia o efeito do `log(preco)` depois do corte de outliers.