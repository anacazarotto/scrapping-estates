# Valorização v3: os juros ajudam a prever a valorização de imóveis?

Gerado por `scripts_predict/valorizacao_macro.py`. Séries mensais reais do Banco
Central: IVG-R (21340), Selic (4189), IPCA (433) e INCC (192). Alvo: valorização do
IVG-R nos 12 meses seguintes (%). Variáveis: Selic, variação da Selic em 12 meses, IPCA
e INCC dos últimos 12 meses e valorização dos últimos 12 meses. Cada corte treina com
os meses até a data e testa nos meses seguintes, que o modelo nunca viu.

## Treino até 2016-12

Treino: 2002-03 a 2015-12 (166 meses). Teste: 2017-01 a 2025-07 (103 meses).

| Modelo | RMSE (p.p.) | MAE (p.p.) | Acerto da direção |
|---|---:|---:|---:|
| Random Forest | 2.67 | 2.22 | 94% |
| Referência: repete os últimos 12 meses | 3.08 | 2.67 | 88% |
| Gradient Boosting | 3.18 | 2.56 | 95% |
| Referência: IPCA dos últimos 12 meses | 4.25 | 3.17 | 95% |
| Referência: média dos últimos 10 anos | 4.74 | 3.55 | 95% |
| Ridge | 5.75 | 4.73 | 45% |
| Regressão Linear | 5.95 | 4.89 | 43% |

## Treino até 2018-12

Treino: 2002-03 a 2017-12 (190 meses). Teste: 2019-01 a 2025-07 (79 meses).

| Modelo | RMSE (p.p.) | MAE (p.p.) | Acerto da direção |
|---|---:|---:|---:|
| Referência: média dos últimos 10 anos | 2.31 | 2.11 | 100% |
| Referência: repete os últimos 12 meses | 3.16 | 2.63 | 100% |
| Gradient Boosting | 3.57 | 3.30 | 100% |
| Random Forest | 3.81 | 3.17 | 99% |
| Referência: IPCA dos últimos 12 meses | 4.63 | 3.55 | 100% |
| Ridge | 5.19 | 4.26 | 66% |
| Regressão Linear | 5.34 | 4.38 | 65% |

## Treino até 2020-12

Treino: 2002-03 a 2019-12 (214 meses). Teste: 2021-01 a 2025-07 (55 meses).

| Modelo | RMSE (p.p.) | MAE (p.p.) | Acerto da direção |
|---|---:|---:|---:|
| Referência: média dos últimos 10 anos | 2.22 | 2.04 | 100% |
| Ridge | 2.24 | 2.02 | 85% |
| Regressão Linear | 2.33 | 2.10 | 82% |
| Random Forest | 2.64 | 2.20 | 95% |
| Gradient Boosting | 2.78 | 2.11 | 100% |
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
