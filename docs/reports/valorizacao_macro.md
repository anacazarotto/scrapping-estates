# Valorização v3: os juros ajudam a prever a valorização de imóveis?

Gerado por `scripts_predict/valorizacao_macro.py`. Séries mensais reais do Banco
Central: IVG-R (21340), Selic (4189), IPCA (433) e INCC (192). Alvo: valorização do
IVG-R nos 12 meses seguintes (%). Variáveis: Selic, variação da Selic em 12 meses, IPCA
e INCC dos últimos 12 meses e valorização dos últimos 12 meses. Cada corte treina com
os meses até a data e testa nos meses seguintes, que o modelo nunca viu.

## Treino até 2016-12

Treino: 2002-03 a 2016-12 (178 meses). Teste: 2017-01 a 2025-07 (103 meses).

| Modelo | RMSE (p.p.) | MAE (p.p.) | Acerto da direção |
|---|---:|---:|---:|
| Gradient Boosting | 2.93 | 2.45 | 95% |
| Random Forest | 3.06 | 2.55 | 87% |
| Referência: repete os últimos 12 meses | 3.08 | 2.67 | 88% |
| Referência: IPCA dos últimos 12 meses | 4.25 | 3.17 | 95% |
| Referência: média dos últimos 10 anos | 4.74 | 3.55 | 95% |
| Ridge | 5.71 | 4.69 | 45% |
| Regressão Linear | 5.91 | 4.85 | 43% |

## Treino até 2018-12

Treino: 2002-03 a 2018-12 (202 meses). Teste: 2019-01 a 2025-07 (79 meses).

| Modelo | RMSE (p.p.) | MAE (p.p.) | Acerto da direção |
|---|---:|---:|---:|
| Referência: média dos últimos 10 anos | 2.31 | 2.11 | 100% |
| Gradient Boosting | 3.07 | 2.80 | 100% |
| Referência: repete os últimos 12 meses | 3.16 | 2.63 | 100% |
| Random Forest | 3.29 | 2.99 | 95% |
| Ridge | 4.13 | 3.57 | 78% |
| Regressão Linear | 4.25 | 3.67 | 78% |
| Referência: IPCA dos últimos 12 meses | 4.63 | 3.55 | 100% |

## Treino até 2020-12

Treino: 2002-03 a 2020-12 (226 meses). Teste: 2021-01 a 2025-07 (55 meses).

| Modelo | RMSE (p.p.) | MAE (p.p.) | Acerto da direção |
|---|---:|---:|---:|
| Ridge | 1.85 | 1.53 | 100% |
| Regressão Linear | 1.88 | 1.58 | 100% |
| Referência: média dos últimos 10 anos | 2.22 | 2.04 | 100% |
| Random Forest | 2.48 | 2.07 | 96% |
| Gradient Boosting | 2.61 | 2.24 | 100% |
| Referência: repete os últimos 12 meses | 3.03 | 2.41 | 100% |
| Referência: IPCA dos últimos 12 meses | 4.52 | 3.12 | 100% |

## Efeito de cada variável (regressão linear, todos os meses)

Descritivo apenas: mostra a associação nos dados, não uma relação que preveja bem.

| Variável | Efeito na valorização dos 12 meses seguintes |
|---|---:|
| Selic (+1 p.p.) | +0.52 p.p. |
| Variação da Selic em 12 meses (+1 p.p.) | -0.04 p.p. |
| IPCA dos últimos 12 meses (+1 p.p.) | -0.61 p.p. |
| INCC dos últimos 12 meses (+1 p.p.) | -0.06 p.p. |
| Valorização dos últimos 12 meses (+1 p.p.) | +0.83 p.p. |
