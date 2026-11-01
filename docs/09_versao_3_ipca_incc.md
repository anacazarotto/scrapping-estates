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

O modelo v3 é treinado com os anúncios reais **e** o histórico simulado, com o ano como
variável a mais (`scripts_predict/modelo_v3.py`). O modelo publicado é escolhido pelo
**MdAPE** (erro percentual absoluto mediano) na validação cruzada; em empate (mesmo MdAPE
com 1 casa decimal), vence o de menor MAE. O ensemble concorre como mais um candidato.

**Como foi treinado e avaliado sem se enganar:**

1. Treino e teste separados por imóvel **antes** de simular, com a mesma divisão do v1
   (semente 42): os simulados de um imóvel de teste nunca entram no treino.
2. Validação cruzada agrupada por imóvel (`GroupKFold`): o real e os simulados de um mesmo
   imóvel ficam sempre na mesma parte. O erro da validação é medido **só nas linhas reais**.
3. Candidatos que podem ir ao site:
   - CatBoost, XGBoost, Gradient Boosting, Random Forest e Ridge, com alvo em preço e em log;
   - o ensemble dos 3 melhores deles (peso 1/MdAPE);
   - o TabPFN 3.5 pela API da Prior Labs (seção 9.8).

   O Random Forest usa 100 árvores (e não 400): o MdAPE mudou menos de 0,2 ponto percentual
   e o modelo ficou 4 vezes menor, com cerca de 20 MB por arquivo comprimido e cerca de
   230 MB de memória somando os dois tipos.

   MLP, SVM, regressões lineares sem penalização e o TabPFN rodando no computador também
   entram no ranking (`resultados_v3.py`), mas não podem ser publicados.
4. Erro final medido **só nos preços reais do teste**.

| Tipo | Modelo do site | Linhas no treino (reais + simuladas) | MAE v3 | MAE v1 | R² v3 | R² v1 | MdAPE v3 |
|---|---|---:|---:|---:|---:|---:|---:|
| Apartamento | Ensemble | 1.400 + 14.000 | R$ 123.858 | R$ 119.640 | 0,781 | 0,795 | 13,1% |
| Casa | TabPFN 3.5 (API) | 1.039 + 10.390 | R$ 302.336 | R$ 291.216 | 0,540 | 0,559 | 18,7% |

Escolha pelo MdAPE na validação cruzada:

- **Casa:** o TabPFN 3.5 pela API ficou com 19,2%. A reserva é o ensemble de Random Forest,
  CatBoost e XGBoost (19,6%), que roda no próprio site e é usado se a API não responder.
- **Apartamento:** o ensemble de Random Forest (preço e log) e XGBoost (log) ficou com 12,9%.

Os pesos ficaram em cerca de 1/3 cada. Na v3 os preços reais do teste estão levados a
ago/2026 (seção 9.8), então o MAE em reais fica um pouco maior.

**Leitura:** pelo MdAPE do teste, os modelos escolhidos erram menos no imóvel típico que a
escolha anterior, que só considerava CatBoost, XGBoost, Gradient Boosting e Ridge:

- **Apartamentos:** 13,1% contra 14,5%.
- **Casas:** 18,7% contra 19,2%.

O R² ficou um pouco menor (0,781 contra 0,799; 0,540 contra 0,576), porque o Random Forest e
o TabPFN erram mais nos imóveis caros, que pesam muito no R². É a troca esperada ao escolher
pelo MdAPE.

O histórico simulado não traz informação nova sobre *qual* imóvel vale mais. O ganho do v3
é outro: ele sabe responder **quanto o imóvel valia em anos anteriores**, de forma coerente
com o índice. Segundo o modelo, os imóveis de teste valiam em jun/2021 cerca de 74% do
valor de hoje (ago/2026), contra 73,5% pelo índice. Essa conta mostra coerência, não acerto,
porque o passado foi gerado pelo próprio índice.

No site, o valor de hoje, o passado e a projeção vêm do modelo v3 (seção 9.8). Relatório:
`docs/reports/modelo_v3.md`.

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
mensais reais e três cortes de validação no tempo. Como o alvo de cada mês é a
valorização dos 12 meses seguintes, o treino termina 12 meses antes de cada corte, para
nenhum alvo de treino usar preços do período de teste.

| Teste a partir de | Melhor | Melhor modelo com juros |
|---|---|---|
| jan/2017 | **Random Forest (2,67 p.p.)** | — |
| jan/2019 | **média de 10 anos (2,31 p.p.)** | Gradient Boosting (3,57 p.p.) |
| jan/2021 | **média de 10 anos (2,22 p.p.)** | Ridge (2,24 p.p.) |

Os modelos com juros só vencem no primeiro período; nos dois mais recentes, a média de 10
anos do IVG-R é a melhor. Conclusão: com esses dados, os juros não melhoram a previsão de
forma consistente, e o site não usa cenários de juros. Relatório completo em
`docs/reports/valorizacao_macro.md`. (A primeira versão desta avaliação não separava os 12
meses finais do treino e mostrava o Ridge vencendo a partir de 2021; o vazamento foi
corrigido na revisão do código.)

## 9.8 Interface final: só a versão 3

A pedido da orientação, o site passou a mostrar **somente a v3**:

- **Estimativa:** o valor de hoje, o passado (2016 a 2026) e a projeção vêm todos do
  modelo v3 e da valorização v3. Não há mais seletor de versões.
- **Modelos:** todos os candidatos treinados no protocolo da v3 (9 algoritmos × 2 alvos,
  `scripts_predict/resultados_v3.py`), cada um com a explicação de como funciona e do porquê
  do resultado com os nossos dados, e a comparação lado a lado.
- **Dataset:** o dataset de treino da v3, com os imóveis reais e o histórico simulado, o
  mês e o preço originais de cada anúncio, filtros e download.

**Correção feita ao montar a interface:** os anúncios reais foram coletados de março a
agosto de 2026, e o modelo usava o mês do anúncio como "ano". Com isso, ele confundia a
diferença entre os imóveis anunciados em cada mês com valorização (somava de 4% a 6% entre
março e agosto). Agora, todos os preços reais são levados a ago/2026 pelo próprio índice
IPCA+INCC (ajuste de no máximo cerca de 2,5%) antes de simular o passado. Assim, o único
efeito do tempo que o modelo aprende vem do índice. O dataset mostra o preço anunciado
original e o preço no mês de referência.

**Todos os candidatos no protocolo da v3** (`dados/resultados_modelos_v3.json`, aba
Modelos do site). Posição pelo MdAPE na validação cruzada (agrupada por imóvel, erro só
nas linhas reais), com MdAPE e R² no teste com preços reais (★ = publicado):

| Posição | Apartamento | MdAPE CV | MdAPE teste | R² teste |
|---:|---|---:|---:|---:|
| 1 | ★ Ensemble (Random Forest preço e log, XGBoost log) | 12,9% | 13,1% | 0,781 |
| 2 | Random Forest (preço) | 13,1% | 13,8% | 0,767 |
| 3 | Random Forest (log) | 13,1% | 13,5% | 0,752 |
| 4 | TabPFN 3.5 no computador (3 mil linhas) | 13,4% | 13,6% | 0,799 |
| 5 | XGBoost (log) | 13,5% | 14,4% | 0,788 |
| 9 | TabPFN 3.5 pela API (todas as linhas) | 13,9% | 14,0% | 0,800 |
| último | SVM (preço) | 37,1% | 39,3% | −0,333 |

| Posição | Casa | MdAPE CV | MdAPE teste | R² teste |
|---:|---|---:|---:|---:|
| 1 | TabPFN 3.5 no computador (3 mil linhas, fora do site) | 18,6% | 18,0% | 0,559 |
| 2 | ★ TabPFN 3.5 pela API (todas as linhas) | 19,2% | 18,7% | 0,540 |
| 3 | Ensemble (Random Forest, CatBoost, XGBoost; reserva) | 19,6% | 18,9% | 0,558 |
| 4 | Random Forest (preço) | 19,6% | 19,6% | 0,478 |
| 5 | TabPFN 3.5 Fast no computador (todas as linhas) | 19,8% | 19,5% | 0,512 |
| 6 | CatBoost (preço) | 20,2% | 19,8% | 0,583 |
| último | Rede Neural (log) | 45,8% | 45,2% | −15,7 |

**TabPFN com corte de tempo** (`scripts_predict/tabpfn_v3.py`, `make tabpfn-v3`). O TabPFN
foi pré-treinado com até cerca de 10 mil linhas, e o treino da v3 tem 11.429 (casas) e
15.400 (apartamentos). Cada variante (v2, 3.5 e 3.5 Fast, locais, em CPU) rodou no mesmo
protocolo, em tentativas cada vez menores (todas as linhas, 10 mil, 3 mil sorteadas), com
corte de 60 minutos cada:

| Variante | Casa | Apartamento |
|---|---|---|
| v2 | todas: cortado; 3 mil: 19 min | todas e 10 mil: cortados; 3 mil: 17 min |
| 3.5 | todas: cortado; 3 mil: 18 min | todas e 10 mil: cortados; 3 mil: 18 min |
| 3.5 Fast | todas: 34 min | todas: 55 min |

Nas casas, o degrau de 10 mil foi pulado: cada parte da validação já treina com 9.143
linhas. O TabPFN 3.5 foi o melhor em casas mesmo com só 3 mil linhas, mas no computador
ele não pode ir ao site.

**TabPFN 3.5 pela API** (`TABPFN_VERSIONS=api`, pacote `tabpfn-client`). O mesmo modelo
3.5 roda no servidor da Prior Labs:
- usa todas as linhas de treino;
- cada parte da validação levou cerca de 6 segundos, e uma consulta do site leva cerca de
  9 segundos;
- cada consulta gasta 10 mil créditos de uma cota mensal de 20 milhões, ou seja, cerca de
  2 mil consultas por mês.

No site, as estimativas ficam em cache, para a mesma consulta não gastar cota de novo. A
chave fica nos Secrets do Streamlit (`TABPFN_TOKEN`). Sem chave, sem internet ou sem cota,
o site usa a reserva e avisa o visitante.

O que os resultados mostram, com os nossos dados:

- **Árvores (CatBoost, XGBoost, Gradient Boosting, Random Forest) e TabPFN são os
  melhores:** o preço não é uma soma simples (o bairro muda o efeito da área; o preço/m²
  não sobe com os quartos). Os primeiros ficam a menos de 1 ponto percentual de MdAPE uns
  dos outros.
- **O Random Forest acerta bem o imóvel típico** (melhor MdAPE em apartamentos), mas tem
  R² menor, porque erra mais nos imóveis caros.
- **Lineares erram de 2 a 8 pontos percentuais a mais** (MdAPE), pelo mesmo motivo; Lasso
  e Ridge ficam iguais à regressão linear.
- **Em casas, o alvo em log piora todos os algoritmos:** o modelo em log prevê o valor
  típico e subestima as casas caras, que dominam o erro em reais. Em apartamentos, o log
  ajuda um pouco.
- **Rede Neural e SVM ficam entre os piores:** precisam de mais imóveis do que temos e de
  ajuste fino; o SVM é sensível à escala do alvo (com o preço em reais, R² negativo), e a
  Rede Neural em log explode ao converter de volta para reais.
- **MdAPE e R² medem coisas diferentes:** o Random Forest e o TabPFN acertam melhor o
  imóvel típico (MdAPE), e o CatBoost acerta melhor os imóveis caros (R²). O critério do
  trabalho é o MdAPE.

## 9.9 Como reproduzir

```bash
make valorizacao-macro      # baixa as séries do BCB (dados/series_macro.json) e avalia os juros
make modelo-v3              # histórico simulado + modelo de preço v3
make resultados-v3          # todos os candidatos no protocolo da v3 (aba Modelos)
make tabpfn-v3             # TabPFN no computador, com corte de 60 min (várias horas)
TABPFN_VERSIONS=api python scripts_predict/tabpfn_v3.py   # TabPFN 3.5 pela API (TABPFN_TOKEN)
make retreinar-tudo         # retreina todos os modelos atuais (cerca de 1 hora com TabPFN)
python scripts_predict/imoveis_projecao.py projetar --bairro Centro --tipo-imovel Apartamento \
    --area-privada 80 --area-total 100 --quartos 2 --banheiros 2 --vagas 1 --anos 10 --versao v3
```
