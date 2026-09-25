# Modelo de valorização mensal

- Meses de coleta: 2026-03, 2026-04, 2026-05, 2026-06, 2026-08
- Horizonte previsto: 3 mês(es) à frente
- Imóveis no painel: 3752
- Pares treino / teste: 2266 / 2259
- Mês(es)-alvo do teste: 2026-08 (nunca vistos no treino)
- Pares em que o preço anunciado não mudou: 91.1%

## Como foi avaliado

- Alvo: variação do preço anunciado do mesmo imóvel entre t e t+h (em log).
- Features só com informação disponível em t (sem olhar o futuro).
- Teste = último(s) mês(es); ranking por validação cruzada agrupada por imóvel no treino.
- Erros em **pontos percentuais** (p.p.): errar 1,0 p.p. = prever +2% quando foi +1%.
- Critério principal: **RMSE** (o modelo prevê a valorização *esperada*). O MAE
  favorece prever 0% porque a maioria dos anúncios não muda num mês.
- Acurácia de direção: sobe / estável (±0.5%) / cai.

## Resultados no teste temporal

| Modelo | RMSE CV treino (p.p.) | RMSE teste (p.p.) | MAE teste (p.p.) | R² (log) | Acurácia direção | MAE (R$) |
|---|---:|---:|---:|---:|---:|---:|
| CatBoost | 4.624 | 3.119 | 1.034 | 0.2351 | 69.3% | R$ 10.804,51 |
| Gradient Boosting | 4.684 | 3.216 | 1.336 | 0.0966 | 47.6% | R$ 14.531,14 |
| Random Forest | 4.607 | 3.275 | 1.117 | 0.0641 | 63.7% | R$ 11.903,94 |
| XGBoost | 4.659 | 3.316 | 1.168 | 0.0625 | 64.1% | R$ 12.873,05 |
| Baseline: preço não muda | - | 3.381 | 0.664 | -0.0012 | 92.2% | R$ 6.718,70 |
| Ridge | 4.554 | 3.381 | 1.024 | -0.0051 | 66.6% | R$ 10.650,48 |
| TabPFN v2 | 4.558 | 3.382 | 0.716 | -0.0016 | 92.3% | R$ 7.263,03 |
| Baseline: média do segmento | - | 3.434 | 1.033 | -0.0345 | 66.7% | R$ 11.058,07 |

## Chance de redução de preço (> 0.5%) em 3 mês(es)

Taxa de reduções: 3.6% no treino, 2.9% no teste. AUC 0,5 = sorteio; Brier menor é melhor.

| Classificador | AUC | Brier |
|---|---:|---:|
| Baseline: taxa histórica | 0.500 | 0.0280 |
| Regressão Logística | 0.604 | 0.0279 |
| Gradient Boosting | 0.727 | 0.0250 |

## Leitura

- Modelo escolhido (menor RMSE de CV): **Ridge** — RMSE no teste 3.381 p.p. contra 3.381 p.p. do melhor baseline (Baseline: preço não muda).
- O modelo **não** superou o baseline no teste: com os meses disponíveis, a variação futura ainda não é previsível além da tendência do segmento. Isso é um resultado válido para o TCC; reavalie com mais meses de coleta.
- Limitações: preço anunciado ≠ preço de venda; imóveis vendidos saem do painel (viés de sobrevivência); poucos meses = pouca variação observada.

## Índice de valorização acumulada (base 100 no 1º mês)

Segmentos com poucos imóveis repetidos usam o índice do tipo (Casa/Apartamento).

| bairro | tipo_imovel | imoveis | 2026-03 | 2026-04 | 2026-05 | 2026-06 | 2026-08 | variacao_total_% |
|---|---|---|---|---|---|---|---|---|
| sao lucas | Apartamento | 40 | 100.0 | 102.61 | 102.71 | 102.87 | 102.87 | 2.87 |
| universitario | Apartamento | 128 | 100.0 | 100.0 | 101.24 | 101.24 | 101.24 | 1.24 |
| lider | Apartamento | 183 | 100.0 | 99.82 | 100.14 | 101.16 | 101.16 | 1.16 |
| pinheirinho | Apartamento | 164 | 100.0 | 99.83 | 100.04 | 101.11 | 101.11 | 1.11 |
| maria goretti | Apartamento | 339 | 100.0 | 100.43 | 100.95 | 100.99 | 100.99 | 0.99 |
| santo antonio | Apartamento | 104 | 100.0 | 100.5 | 100.87 | 100.96 | 100.96 | 0.96 |
| efapi | Apartamento | 349 | 100.0 | 100.07 | 100.18 | 100.63 | 100.63 | 0.63 |
| desbravador | Apartamento | 99 | 100.0 | 100.0 | 100.61 | 100.61 | 100.61 | 0.61 |
| paraiso | Apartamento | 199 | 100.0 | 100.05 | 100.05 | 100.6 | 100.6 | 0.6 |
| palmital | Apartamento | 120 | 100.0 | 100.1 | 100.53 | 100.59 | 100.59 | 0.59 |
| sao cristovao | Apartamento | 338 | 100.0 | 100.1 | 100.03 | 100.47 | 100.47 | 0.47 |
| bom pastor | Apartamento | 45 | 100.0 | 100.27 | 100.27 | 100.42 | 100.42 | 0.42 |
| jardim italia | Apartamento | 580 | 100.0 | 99.97 | 100.05 | 100.33 | 100.33 | 0.33 |
| belvedere | Apartamento | 22 | 100.0 | 100.04 | 100.14 | 100.29 | 100.29 | 0.29 |
| engenho braun | Apartamento | 24 | 100.0 | 100.04 | 100.14 | 100.29 | 100.29 | 0.29 |
| jardim america | Apartamento | 21 | 100.0 | 100.04 | 100.14 | 100.29 | 100.29 | 0.29 |
| jardins lunardi | Apartamento | 25 | 100.0 | 100.04 | 100.14 | 100.29 | 100.29 | 0.29 |
| santa paulina | Apartamento | 27 | 100.0 | 100.04 | 100.14 | 100.29 | 100.29 | 0.29 |
| passo dos fortes | Apartamento | 333 | 100.0 | 100.16 | 100.16 | 100.19 | 100.19 | 0.19 |
| centro | Apartamento | 3473 | 100.0 | 100.05 | 100.05 | 100.17 | 100.17 | 0.17 |
| santa maria | Apartamento | 303 | 100.0 | 99.99 | 100.06 | 100.17 | 100.17 | 0.17 |
|  | Apartamento | 85 | 100.0 | 100.04 | 100.14 | 100.14 | 100.14 | 0.14 |
| bom retiro | Apartamento | 50 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 |
| presidente medice | Apartamento | 49 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 |
| seminario | Apartamento | 73 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 |
| vila real | Apartamento | 132 | 100.0 | 98.83 | 100.0 | 100.0 | 100.0 | -0.0 |
| presidente medici | Apartamento | 559 | 100.0 | 100.01 | 100.08 | 99.95 | 99.95 | -0.05 |
| esplanada | Apartamento | 158 | 100.0 | 99.85 | 99.85 | 99.85 | 99.85 | -0.15 |
| bela vista | Apartamento | 116 | 100.0 | 100.0 | 99.74 | 99.74 | 99.74 | -0.26 |
| dom geronimo | Apartamento | 54 | 100.0 | 99.24 | 99.24 | 99.24 | 99.24 | -0.76 |
