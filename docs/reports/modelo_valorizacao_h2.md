# Modelo de valorização mensal

- Meses de coleta: 2026-03, 2026-04, 2026-05, 2026-06, 2026-08
- Horizonte previsto: 2 mês(es) à frente
- Imóveis no painel: 3752
- Pares treino / teste: 5259 / 2412
- Mês(es)-alvo do teste: 2026-08 (nunca vistos no treino)
- Pares em que o preço anunciado não mudou: 93.9%

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
| Baseline: preço não muda | - | 2.317 | 0.324 | -0.0000 | 96.0% | R$ 2.628,60 |
| Baseline: média do segmento | - | 2.394 | 0.612 | -0.0790 | 77.3% | R$ 6.190,53 |
| Random Forest | 3.506 | 2.481 | 0.794 | -0.1587 | 68.1% | R$ 10.986,35 |
| Gradient Boosting | 3.581 | 2.516 | 0.890 | -0.1988 | 60.3% | R$ 8.997,52 |
| XGBoost | 3.509 | 2.520 | 0.729 | -0.1954 | 75.8% | R$ 7.961,36 |
| Ridge | 3.470 | 2.601 | 0.742 | -0.2722 | 72.6% | R$ 31.780,70 |
| CatBoost | 3.493 | 2.708 | 0.696 | -0.4298 | 79.6% | R$ 6.980,67 |

## Chance de redução de preço (> 0.5%) em 2 mês(es)

Taxa de reduções: 2.5% no treino, 1.9% no teste. AUC 0,5 = sorteio; Brier menor é melhor.

| Classificador | AUC | Brier |
|---|---:|---:|
| Baseline: taxa histórica | 0.500 | 0.0183 |
| Regressão Logística | 0.655 | 0.0192 |
| Gradient Boosting | 0.644 | 0.0186 |

## Leitura

- Modelo escolhido (menor RMSE de CV): **Ridge** — RMSE no teste 2.601 p.p. contra 2.317 p.p. do melhor baseline (Baseline: preço não muda).
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
