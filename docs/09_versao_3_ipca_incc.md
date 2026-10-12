# 9. Versão 3: histórico simulado e valorização pelo índice IPCA+INCC

## 9.1 O problema

Cinco meses de anúncios não mostram a valorização: 96% dos preços anunciados não mudaram
de um mês para o outro, e a taxa medida (+0,89% ao ano em apartamentos) ficou longe da
realidade. Um caso real deixou isso claro: um apartamento de Chapecó comprado por
R$ 160 mil há 5 anos foi avaliado por um correspondente bancário em cerca de R$ 235 mil,
**+7,99% ao ano**.

| Método | Taxa por ano | Valor previsto hoje para esse apartamento | Erro |
|---|---:|---:|---:|
| v1 (só anúncios) | +0,89% | R$ 167 mil | −29% |
| v2 com IVG-R | +3,07% | R$ 186 mil | −21% |
| v2 com IPCA | +3,75% | R$ 192 mil | −18% |
| IVG-R realizado nos últimos 5 anos | +4,42% | R$ 198 mil | −16% |
| IPCA realizado nos últimos 5 anos | +5,30% | R$ 207 mil | −12% |
| **IPCA+INCC realizado nos últimos 5 anos** | **+5,95%** | **R$ 214 mil** | **−9%** |
| INCC realizado nos últimos 5 anos | +6,60% | R$ 220 mil | −6% |

Um caso só não serve para escolher o método (a compra pode ter sido abaixo do mercado, a
avaliação é aproximada). Por isso o índice da v3 foi definido por um critério que não
depende desse caso (9.2), e o caso foi usado só para conferir.

## 9.2 O índice IPCA+INCC

**Lógica:** o preço de um imóvel tem duas partes. O terreno e o bem como um todo tendem a
acompanhar a inflação geral (**IPCA**); a construção tende a acompanhar o custo de
construir (**INCC**). O índice combinado é a média simples das duas variações mensais,
peso 50% cada. Séries reais do Banco Central (SGS 433 e 192), em
`scripts_predict/indice_ipca_incc.py`.

| | IPCA | INCC | **IPCA+INCC** |
|---|---:|---:|---:|
| Últimos 5 anos | +5,30%/ano | +6,60%/ano | **+5,95%/ano** |
| Últimos 10 anos | +4,96%/ano | — | **+5,80%/ano** |
| Faixa de 12 meses (p10 a p90, 10 anos) | — | — | **+3,52% a +11,42%** |

Variação do índice em cada ano: 2016 +6,20%; 2017 +3,60%; 2018 +3,79%; 2019 +4,23%;
2020 +6,65%; 2021 +11,95%; 2022 +7,53%; 2023 +4,06%; 2024 +5,68%; 2025 +5,10%.

## 9.3 O histórico simulado

Para cada um dos 3.283 imóveis reais (casas e apartamentos com preço e área), foram criados 10 registros,
um por ano de 2016 a 2025, com o preço levado para o meio de cada ano pelo índice:

```
preço simulado em jun/Y = preço anunciado ÷ (quanto o IPCA+INCC subiu de jun/Y até o mês do anúncio)
```

Exemplo: um apartamento anunciado por R$ 500 mil em ago/2026 aparece por R$ 368 mil em
jun/2021 (o índice subiu 36% nesse intervalo: 500 ÷ 1,36 ≈ 368) e por R$ 283 mil em jun/2016.

Cada registro tem a coluna `simulado` (0 = real, 1 = simulado) e o ano. O dataset completo
(3.283 reais + 32.830 simulados) fica em `imoveis_simulados_v3.db` e em
`dados/dataset_simulado_v3.csv`, aberto no modo avançado do site.

**O que é e o que não é:** os índices de cada ano são reais; o passado de cada imóvel é
uma **hipótese** (o imóvel teria acompanhado o índice), não um preço observado.

## 9.4 O modelo de preço v3

O modelo v3 é o mesmo tipo de ensemble do v1, treinado com os anúncios reais **e** o
histórico simulado, com o ano como variável a mais (`scripts_predict/modelo_v3.py`).

**Como foi treinado e avaliado sem se enganar:**

1. Treino e teste separados por imóvel **antes** de simular, com a mesma divisão do v1
   (semente 42): os simulados de um imóvel de teste nunca entram no treino.
2. Validação cruzada agrupada por imóvel (`GroupKFold`): o real e os simulados de um mesmo
   imóvel ficam sempre na mesma parte. O erro da validação é medido **só nas linhas reais**.
3. Candidatos: CatBoost, XGBoost, Gradient Boosting e Ridge, com alvo em preço e em log.
   Random Forest, MLP e SVM ficaram de fora: com cerca de 15 mil linhas, o Random Forest
   geraria um arquivo de centenas de MB, e MLP e SVM foram os piores no v1.
4. Erro final medido **só nos preços reais do teste**.

| Tipo | Linhas no treino (reais + simuladas) | MAE v3 | MAE v1 | R² v3 | R² v1 | Erro mediano v3 |
|---|---:|---:|---:|---:|---:|---:|
| Apartamento | 1.400 + 14.000 | R$ 118.600 | R$ 119.640 | 0,800 | 0,795 | 15,0% |
| Casa | 1.038 + 10.380 | R$ 290.535 | R$ 291.216 | 0,571 | 0,559 | 19,2% |

Ensembles: apartamento = CatBoost (log), XGBoost (log) e CatBoost; casa = CatBoost, XGBoost
e Gradient Boosting.

**Leitura:** no preço de hoje, o v3 empata com o v1 (diferença de 1%, dentro da variação
normal). Era o esperado: o histórico simulado não traz informação nova sobre *qual*
imóvel vale mais. O ganho do v3 é outro: ele sabe responder **quanto o imóvel valia em
anos anteriores**, de forma coerente com o índice. Os imóveis de teste valiam, em
jun/2021, 75,7% do valor de hoje segundo o modelo, contra 75,4% pelo índice. Essa conta
mostra coerência, não acerto, porque o passado foi gerado pelo próprio índice.

No site, o valor de hoje continua vindo do modelo v1, que foi validado com mais
candidatos e tem o mesmo desempenho. O v3 aparece no quadro "Quanto este imóvel valia nos
anos anteriores". Relatório: `docs/reports/modelo_v3.md`.

## 9.5 A valorização v3

A projeção para o futuro usa a mesma fórmula da v2 (seção 8.5), agora com o IPCA+INCC
como ponto de partida: os anúncios pesam T / (T + 12) = 29% e o índice 71%.

| Tipo | v1 | v2 com IVG-R | **v3 com IPCA+INCC** |
|---|---|---|---|
| Apartamento | +0,89%/ano | +3,07% | **+4,33% (faixa +2,64% a +8,32%)** |
| Casa | +0,38%/ano | +2,92% | **+4,17% (faixa +2,47% a +8,17%)** |

A valorização do apartamento do caso real (+7,99% ao ano) fica **dentro da faixa** da v3,
perto do cenário otimista. Nenhuma versão anterior chegava a isso.

No apartamento de exemplo do site (80 m² no Centro, R$ 630.776 hoje): R$ 956.931 em 10
anos no cenário central da v3 (+4,26% ao ano no bairro; faixa de R$ 808.069 a
R$ 1.397.788).

## 9.6 Como explicar na banca

1. "Cinco meses de anúncios não medem valorização; isso é comum em dados coletados por
   pouco tempo."
2. "Testei a hipótese de que os juros explicam a valorização com 25 anos de dados reais do
   Banco Central: o resultado foi inconsistente (seção 9.7)."
3. "Adotei um índice com lógica econômica, metade inflação e metade custo de construção,
   e simulei o passado de cada imóvel com os valores reais desse índice."
4. "Os erros do modelo são medidos só nos preços reais; o passado simulado é declarado
   como hipótese."
5. "Conferi com um caso real: a valorização ficou dentro da faixa da v3. Para validar de
   verdade, o próximo passo são dados de venda (ITBI) da Prefeitura."

## 9.7 Estudo complementar: os juros explicam a valorização?

`scripts_predict/valorizacao_macro.py` testou modelos (Regressão Linear, Ridge, Random
Forest, Gradient Boosting) para prever a valorização do IVG-R nos 12 meses seguintes a
partir de Selic, variação da Selic, IPCA, INCC e valorização passada, com 25 anos de dados
mensais reais e três cortes de validação no tempo.

| Treino até | Melhor | Melhor referência simples |
|---|---|---|
| dez/2016 | Gradient Boosting (2,93 p.p.) | repete os últimos 12 meses (3,08 p.p.) |
| dez/2018 | **média de 10 anos (2,31 p.p.)** | — (os modelos perderam) |
| dez/2020 | Ridge (1,85 p.p.) | média de 10 anos (2,22 p.p.) |

O melhor modelo muda a cada período, e nenhum vence as referências nos três. Conclusão:
com esses dados, os juros não melhoram a previsão de forma consistente, e o site não usa
cenários de juros. Relatório completo em `docs/reports/valorizacao_macro.md`.

## 9.8 Como reproduzir

```bash
make valorizacao-macro      # baixa as séries do BCB (dados/series_macro.json) e avalia os juros
make modelo-v3              # histórico simulado + modelo de preço v3
make retreinar-tudo         # retreina todos os modelos atuais (cerca de 1 hora com TabPFN)
python scripts_predict/imoveis_projecao.py projetar --bairro Centro --tipo-imovel Apartamento \
    --area-privada 80 --area-total 100 --quartos 2 --banheiros 2 --vagas 1 --anos 10 --versao v3
```
