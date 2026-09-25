# Modelo de valorização mensal

- Meses de coleta: 2026-03, 2026-04, 2026-05, 2026-06, 2026-08
- Horizonte previsto: 1 mês(es) à frente
- Imóveis no painel: 3752
- Pares treino / teste: 5977 / 2589
- Mês(es)-alvo do teste: 2026-06 (nunca vistos no treino)
- Pares em que o preço anunciado não mudou: 96.0%

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
| Baseline: preço não muda | - | 2.356 | 0.338 | -0.0043 | 95.5% | R$ 3.659,57 |
| TabPFN v2 | 2.109 | 2.357 | 0.405 | -0.0062 | 95.0% | R$ 4.202,58 |
| Baseline: média do segmento | - | 2.365 | 0.452 | -0.0137 | 91.5% | R$ 4.881,62 |
| Ridge | 2.173 | 2.385 | 0.552 | -0.0326 | 88.5% | R$ 5.780,24 |
| Random Forest | 2.121 | 2.393 | 0.570 | -0.0431 | 85.2% | R$ 5.914,34 |
| XGBoost | 2.119 | 2.411 | 0.551 | -0.0597 | 87.1% | R$ 5.784,86 |
| Gradient Boosting | 2.140 | 2.414 | 0.662 | -0.0660 | 75.7% | R$ 6.748,89 |
| CatBoost | 2.121 | 2.507 | 0.503 | -0.1419 | 91.6% | R$ 5.280,04 |

## Chance de redução de preço (> 0.5%) em 1 mês(es)

Taxa de reduções: 1.4% no treino, 1.4% no teste. AUC 0,5 = sorteio; Brier menor é melhor.

| Classificador | AUC | Brier |
|---|---:|---:|
| Baseline: taxa histórica | 0.500 | 0.0133 |
| Regressão Logística | 0.648 | 0.0138 |
| Gradient Boosting | 0.695 | 0.0131 |

## Leitura

- Modelo escolhido (menor RMSE de CV): **TabPFN v2** — RMSE no teste 2.357 p.p. contra 2.356 p.p. do melhor baseline (Baseline: preço não muda).
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
