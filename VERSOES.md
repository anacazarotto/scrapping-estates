# Linha do tempo das versões

O trabalho passou por quatro versões do modelo. Cada uma nasceu de um problema da anterior.
Todas usam os mesmos anúncios reais: 12 coletas de 8 imobiliárias de Chapecó, entre
23/03/2026 e 02/08/2026, que resultaram em 3.283 casas e apartamentos com preço e área.

| Versão | Quando | O que mudou | Melhor modelo publicado | MdAPE no teste | Valorização (apto / casa) |
|---|---|---|---|---|---|
| v1 | set/2026 | Só os anúncios reais; escolha pelo MAE | Apto: *ensemble* CatBoost + XGBoost (log) · Casa: *ensemble* CatBoost + XGBoost + Random Forest | 14,8% / 19,6% | +0,89% / +0,38% ao ano (só anúncios) |
| v2 | out/2026 | Sugestões do orientador: novas variáveis no preço (sem ganho) e valorização combinada com o IPCA e o IVG-R | igual à v1 | igual à v1 | +3,07% / +2,92% ao ano (anúncios + IVG-R) |
| v3 | out–nov/2026 | Índice IPCA+INCC, histórico simulado 2016–2025, escolha pelo MdAPE, 22 candidatos + *ensemble*, TabPFN pela API | Apto: *ensemble* Random Forest (preço e log) + XGBoost (log) · Casa: TabPFN 3.5 pela API | 13,1% / 18,7% | +4,33% / +4,17% ao ano |
| v4 | nov/2026 | Histórico simulado com taxa própria por imóvel (lógica *fuzzy*, ±1,5 p.p. ao ano em torno do índice) | iguais aos da v3 | 12,8% / 18,6% | +4,33% / +4,17% ao ano (média) |

MdAPE: erro percentual absoluto mediano. Metade dos imóveis de teste teve a estimativa a menos
desse percentual do preço anunciado.

## v1: só os anúncios reais (setembro de 2026)

- **Dados:** os anúncios reais, sem histórico (`dados/dataset_imoveis.csv`).
- **Modelos:** regressões lineares, árvores (Random Forest, Gradient Boosting, XGBoost,
  CatBoost), rede neural, SVM e TabPFN v2. A escolha foi pelo menor MAE na validação cruzada.
- **Valorização:** variação do preço do mesmo anúncio entre as coletas.
- **Problema:** 95,9% dos preços não mudaram entre coletas. A valorização medida ficou muito
  abaixo da inflação (+0,89% e +0,38% ao ano), e o modelo não sabia dizer quanto um imóvel
  valia antes de 2026.
- Documentação: [docs/01](docs/01_coleta_e_preparacao_dos_dados.md) a
  [docs/07](docs/07_resultados_e_conclusoes.md).

## v2: sugestões do orientador (outubro de 2026)

- **Preço:** quatro ideias testadas (preço/m² do bairro, prever o preço/m², as duas juntas e
  preços corrigidos pelo IPCA). Nenhuma mudou o erro em mais de 0,5%, e o modelo de preço
  continuou o da v1.
- **Valorização:** taxa dos anúncios combinada com um índice pela teoria da credibilidade.
  Primeiro com o IPCA (+3,75% e +3,59% ao ano), depois com o IVG-R do Banco Central
  (+3,07% e +2,92% ao ano).
- **Também testados:** TabPFN 3.5, idade do prédio pelas fotos (deixada para trabalhos futuros)
  e o modo avançado do site.
- Documentação: [docs/08](docs/08_versao_2_orientacao.md).

## v3: histórico simulado pelo IPCA+INCC (outubro e novembro de 2026)

- **Índice:** média das variações mensais do IPCA e do INCC (+5,80% ao ano em 10 anos).
- **Histórico simulado:** os preços reais foram levados a ago/2026 pelo índice e, para cada
  imóvel, foram criadas 10 linhas sintéticas (junho de 2016 a 2025). São 3.283 linhas reais e
  32.830 simuladas (`dados/dataset_simulado_v3.csv`).
- **Modelos:** 9 algoritmos × alvo em preço e em log, TabPFN (v2, 3.5 e 3.5 Fast no
  computador, com corte de 60 min, e 3.5 pela API) e um *ensemble* dos 3 melhores que rodam no
  site. Escolha pelo menor MdAPE na validação cruzada agrupada por imóvel, com erro medido só
  nos preços reais, e desempate pelo MAE.
- **Resultado:** é a versão publicada no site.
- Documentação: [docs/09](docs/09_versao_3_ipca_incc.md).

## v4: histórico com lógica *fuzzy* (novembro de 2026)

- **Ideia:** na v3, todos os imóveis têm o mesmo passado. Na v4, cada imóvel recebe a taxa do
  índice mais um ajuste de −1,5 a +1,5 ponto percentual ao ano, decidido por regras *fuzzy* a
  partir do preço do bairro, do tamanho do imóvel e da tendência do bairro nas coletas
  (`scripts_predict/fuzzy_valorizacao.py`). O ajuste tem média zero em cada tipo.
- **Dados:** `dados/dataset_simulado_v4.csv` (mesmas linhas da v3, com a coluna `ajuste_pp`).
- **Resultado:** mesmos modelos escolhidos e mesmo erro da v3. A diferença é ruído, segundo um
  *bootstrap* pareado. O que muda é o passado de cada imóvel. Fica como análise de robustez:
  o resultado do trabalho não depende de todos os imóveis terem valorizado igual.
- Documentação: [docs/10](docs/10_versao_4_fuzzy.md).

## Como rodar cada versão

```bash
make train-price-model-rapido   # v1 (modelo de preço só com anúncios)
make experimento-v2             # v2 (experimentos de preço)
make modelo-v3 resultados-v3    # v3
make tabpfn-v4 modelo-v4 resultados-v4 comparar-v3-v4   # v4
```
