# 8. Versão 2: sugestões do orientador

Na reunião de orientação de outubro de 2026, o orientador sugeriu cinco mudanças. Este
capítulo registra o que foi feito com cada uma, sem apagar a versão 1: os modelos, os
números e as telas anteriores continuam disponíveis.

| Sugestão | O que foi feito | Situação |
|---|---|---|
| Estimar a idade do prédio pelas imagens, se for possível | Avaliado: não é possível com os dados coletados (8.1) | Trabalho futuro |
| Treinar também com TabPFN v3, Fast e Thinking | Suporte às versões 3.5, 3.5-Fast e 3.5-Thinking (8.2) | Pronto no código; depende do aceite da licença e do token da Prior Labs |
| Modo avançado no site, com todos os modelos e o dataset | Nova aba "Modo avançado" (8.3 e capítulo 6) | Publicado |
| Listas suspensas nos campos com restrição | Quartos, banheiros e vagas viraram listas (8.3) | Publicado |
| Aproximar o valor, usando o IPCA no treino, numa v2 | Experimento no modelo de preço (8.4) e valorização v2 com IPCA (8.5) e com o IVG-R, índice de imóveis (8.6) | Valorização v2 com IVG-R publicada como padrão; modelo de preço mantido no v1 |

## 8.1 Idade do prédio pelas fotos

Para treinar um modelo que estima a idade a partir da foto, é preciso um gabarito: imóveis
com a foto **e** o ano de construção conhecido, para aprender e para medir o erro.

- Nenhum dos 8 robôs de coleta captura o ano de construção, e ele não aparece nos campos
  dos anúncios salvos.
- Nenhuma foto foi guardada: o link da foto principal entrou nos robôs depois das 12
  coletas (seção 1.4). E mesmo a foto principal varia sem padrão (fachada, sala, vista).
- Um modelo de imagem pronto (que classifica fotos sem treino específico) daria uma
  resposta, mas sem gabarito não haveria como dizer se ela está certa. Um número sem
  medida de erro não pode entrar no modelo de preço nem no TCC.

**Conclusão:** como o orientador previu, fica como trabalho futuro. Seria preciso coletar o
ano de construção onde a imobiliária o informar e várias fotos por imóvel, e então medir
o erro da estimativa antes de usá-la como variável.

## 8.2 TabPFN 3.5, Fast e Thinking

A Prior Labs lançou o TabPFN-3.5 em setembro de 2026, em três variantes. O pacote
`tabpfn` do projeto (versão 9.0) já traz as duas primeiras:

| Variante | Onde roda | O que precisa |
|---|---|---|
| TabPFN 3.5 | localmente | aceitar a licença uma vez em https://ux.priorlabs.ai (licença não comercial, que permite uso acadêmico) |
| TabPFN 3.5 Fast | localmente | o mesmo aceite; até 6 vezes mais rápido, um pouco menos preciso |
| TabPFN 3.5 Thinking | só na API da Prior Labs | conta, token (`TABPFN_TOKEN`) e créditos pagos; os dados de treino vão para o servidor |

`scripts_predict/tabpfn_model.py` passou a aceitar a versão pelo parâmetro `version` e
pela variável `TABPFN_VERSIONS` (ex.: `v2,v3.5,v3.5-fast,thinking`). Cada variante entra
no ranking com nome próprio ("TabPFN v3.5", "TabPFN v3.5 Fast", "TabPFN 3.5 Thinking
(API)"), no modelo de preço e no de valorização. Sem a licença ou o token, a variante é
pulada com um aviso, e o resto do treino segue igual.

**O que falta:** o aceite da licença e, para o Thinking, a conta com créditos. Os dois são
feitos pela autora do trabalho. Depois disso, basta rodar
`TABPFN_VERSIONS=v2,v3.5,v3.5-fast make benchmark-models-hibrido` e
`make resultados-modelos` com a mesma variável, e os resultados aparecem no relatório e no
modo avançado.

## 8.3 Mudanças no site

- **Modo avançado**: escolha de qualquer modelo candidato, com descrição, prós e contras,
  métricas, gráfico de previsto × real e distribuição do erro; comparação de todos os
  modelos; e o dataset com filtros e download (detalhes na seção 6.4.1).
- **Listas suspensas**: quartos, banheiros e vagas só aceitam valores de 0 até o máximo
  visto no treino para o tipo de imóvel. As áreas continuam livres, porque um valor fora
  da faixa ainda pode ser real; nesse caso o site mostra o aviso da seção 6.5.
- **Versão da valorização**: v2 com IVG-R (padrão), v2 com IPCA ou v1, na barra lateral; a
  aba de bairros mostra as três taxas.

## 8.4 Experimento: o modelo de preço pode ficar mais próximo?

**Resposta curta: não, com os dados atuais.** Quatro ideias foram testadas com exatamente o
protocolo do v1, e nenhuma mudou o erro em mais de meio por cento. O modelo de preço do
site continua o v1.

| Variante | O que muda | Casa: MAE (vs v1) | Apartamento: MAE (vs v1) |
|---|---|---:|---:|
| v1 (atual) | — | R$ 291.216 | R$ 119.640 |
| bairro | + mediana do preço/m² do bairro, aprendida só no treino | R$ 291.376 (+0,1%) | R$ 119.468 (−0,1%) |
| m2 | prevê o preço/m² e multiplica pela área | R$ 289.732 (−0,5%) | R$ 119.781 (+0,1%) |
| bairro+m2 | as duas juntas | R$ 297.547 (+2,2%) | R$ 120.152 (+0,4%) |
| ipca | preços corrigidos pelo IPCA até ago/2026 | R$ 292.026 (+0,3%) | R$ 119.985 (+0,3%) |

Relatório completo, com R² e erro mediano: `docs/reports/experimento_preco_v2.md`
(`make experimento-v2`).

**Por que nada mudou:**

- **IPCA no treino:** os preços foram coletados entre março e agosto de 2026. Corrigir
  pela inflação desse período muda cada preço entre −0,25% e +1,16%, muito menos que o
  erro do modelo (15% a 20%). O IPCA não traz informação nova sobre *qual* imóvel vale mais.
- **Preço do bairro e preço/m²:** o CatBoost e o XGBoost já aprendem o efeito do bairro
  pelo one-hot. Dar a mesma informação em outro formato não acrescenta nada.
- **Confirma a conclusão do capítulo 7:** o limite do modelo está na informação disponível
  (sem endereço, idade, acabamento), não no método. Com um único teste de 261 casas e 357
  apartamentos, diferenças de ±0,5% são pequenas demais para dizer que uma variante é
  melhor.

Por isso, a mudança "para o valor ficar mais próximo" foi feita onde ela tem efeito real:
na valorização (8.5).

## 8.5 Valorização v2: anúncios combinados com o IPCA

O ponto fraco apontado pelo orientador está na valorização: em 5 meses, 96% dos anúncios
não mudaram de preço, e a taxa medida (+0,89% ao ano em apartamentos, +0,38% em casas)
ficou muito abaixo da inflação. Com tão poucos meses, a medida tem pouca informação.

A v2 trata o IPCA como o ponto de partida e deixa os dados corrigirem esse ponto na
proporção da informação que têm (uma média ponderada no estilo bayesiano, a mesma ideia de
"credibilidade" já usada para bairros com poucos imóveis):

```
peso_dados = T / (T + 12)
taxa_v2    = peso_dados × taxa_medida + (1 − peso_dados) × IPCA médio de 10 anos (+4,89%)
```

- **T** são os meses de coleta (março a agosto de 2026: 5). Hoje a taxa medida pesa
  5 / 17 = 29% e o IPCA, 71%.
- **12** é a força do IPCA, em "meses de dados equivalentes": com 12 meses de coleta, os
  anúncios e o IPCA pesariam metade cada. É uma escolha do método, configurável em
  `IPCA_PRIOR_MESES` (`imoveis_projecao.py`).
- **Diferenças entre bairros** são mantidas, reduzidas pelo mesmo peso: um bairro que
  valorizou acima da média continua acima na v2.
- **Faixa pessimista–otimista**: combina, com os mesmos pesos, o intervalo de 80% da taxa
  medida (bootstrap) com a variação do próprio IPCA em 12 meses nos últimos 10 anos
  (percentis 10 e 90: +2,80% e +9,06%).

| Tipo | v1 central (faixa) | v2 central (faixa) |
|---|---|---|
| Apartamento | +0,89%/ano (+0,57% a +1,24%) | +3,70%/ano (+2,14% a +6,70%) |
| Casa | +0,38%/ano (+0,01% a +0,75%) | +3,54%/ano (+1,97% a +6,55%) |

Exemplo, apartamento de 80 m² no Centro, valor hoje R$ 630.776:

| Em 10 anos | Valor |
|---|---:|
| v1 (só os anúncios, +0,65%/ano no bairro) | R$ 672.838 |
| **v2 (anúncios + IPCA, +3,62%/ano)** | **R$ 900.322** (faixa R$ 769.466 a R$ 1.202.309) |
| Só o IPCA (+4,89%/ano) | R$ 1.016.485 |

**Como explicar:** a v2 não "inventa" valorização: ela assume, como hipótese declarada, que
imóveis tendem a acompanhar a inflação no longo prazo, e usa os anúncios para ajustar essa
hipótese bairro a bairro. A v1 continua no site para comparação. A v2 não pode ser
validada com os dados atuais, porque não há anos de histórico para conferir; com mais
meses de coleta, o peso dos anúncios cresce sozinho e a comparação passa a ser possível.

**Limitação:** o número 12 é uma escolha, não algo medido. Com 12 meses de coleta, vale
repetir a análise e verificar se a taxa medida se aproxima do IPCA.

## 8.6 Valorização v2 com o IVG-R (padrão do site)

O IPCA mede a inflação em geral, não o preço de imóveis. O Banco Central publica um
índice de imóveis, o **IVG-R** (Índice de Valores de Garantia de Imóveis Residenciais
Financiados, série SGS 21340), calculado a partir das avaliações feitas pelos bancos nos
financiamentos. Ele passou a ser o ponto de partida padrão da v2, com a mesma fórmula da
seção 8.5 (o IPCA continua disponível como alternativa).

| Índice (BCB) | Últimos 12 meses | Média de 10 anos | Faixa de 12 meses (p10 a p90) |
|---|---:|---:|---|
| IPCA (inflação) | +4,22% | +4,89%/ano | +2,80% a +9,06% |
| **IVG-R (imóveis)** | **+5,28%** | **+3,99%/ano** | **−2,04% a +8,23%** |

Dados até jul/2026 (IVG-R) e ago/2026 (IPCA); `scripts_predict/ivgr.py` baixa a série e
guarda uma cópia em `dados/ivgr.json`.

| Tipo | v1 | v2 com IPCA | **v2 com IVG-R** |
|---|---|---|---|
| Apartamento | +0,89%/ano | +3,70% (+2,14% a +6,70%) | **+3,07% (−1,28% a +6,13%)** |
| Casa | +0,38%/ano | +3,54% (+1,97% a +6,55%) | **+2,92% (−1,45% a +5,98%)** |

No apartamento de exemplo (R$ 630.776 hoje), a v2 com IVG-R projeta **R$ 847.541 em 10
anos** (+3,00% ao ano no Centro; faixa de R$ 547.286 a R$ 1.139.090).

**Por que o IVG-R é melhor que o IPCA aqui:**

- **Mede o que o trabalho quer medir:** valor de imóveis, não preço de alimentos,
  transporte e serviços.
- **Reforça um achado do TCC:** em 10 anos, os imóveis financiados no Brasil valorizaram
  +3,99% ao ano, menos que a inflação (+4,89%). A valorização baixa vista nos anúncios de
  Chapecó vai na mesma direção.
- **Faixa mais honesta:** o cenário pessimista pode ser de queda, porque o próprio índice
  caiu em alguns períodos da última década.

**Limitação:** o IVG-R é nacional; não existe recorte para Chapecó nem para Santa Catarina
nessa série. Ele mede o valor de avaliação bancária, que também não é o preço de venda.

## 8.7 Como reproduzir

```bash
make experimento-v2         # docs/reports/experimento_preco_v2.md
make resultados-modelos     # dados do modo avançado
python scripts_predict/imoveis_projecao.py projetar --bairro Centro --tipo-imovel Apartamento \
    --area-privada 80 --area-total 100 --quartos 2 --banheiros 2 --vagas 1 --anos 10 --versao v2-ivgr
python scripts_predict/ivgr.py   # atualiza dados/ivgr.json
```
