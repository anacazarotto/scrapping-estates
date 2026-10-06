# 6. Interface

Aplicação web em **Streamlit** (`interface/app.py`) onde qualquer pessoa informa as
características de um imóvel e recebe o valor estimado hoje e a projeção ano a ano.

```bash
python -m streamlit run interface/app.py     # abre em http://localhost:8501
```

## 6.1 Tela de estimativa

![Tela de estimativa](figures/15_interface_estimativa.png)

**Entrada (barra lateral):**

- tipo (Apartamento ou Casa) e bairro (lista dos bairros com anúncios na base);
- área privativa e área total (campos livres, com aviso fora da faixa de treino);
- quartos, banheiros e vagas em **listas suspensas**, de 0 (não informado) até o máximo
  visto no treino para o tipo (apartamento: 4 quartos, 4 banheiros, 3 vagas; casa: 5, 5
  e 5);
- horizonte da projeção (1 a 30 anos);
- versão da valorização: **v2** (padrão, anúncios combinados com o IPCA) ou **v1** (só os
  anúncios) — capítulo 8;
- cenário de comparação: IPCA média de 10 anos (padrão), IPCA de 12 meses, outra taxa
  ou nenhum;
- opções avançadas: informar o preço atual em vez de usar o valor estimado.

**Saída:**

- valor estimado hoje, com o erro típico do modelo para o tipo de imóvel;
- valor no fim do horizonte (cenário central) e a variação acumulada;
- taxa de valorização anual do bairro, com a faixa pessimista–otimista;
- quanto o imóvel valeria se apenas acompanhasse a inflação;
- gráfico interativo com o cenário central, a faixa de incerteza e a linha do IPCA;
- aviso quando o bairro tem poucos dados e foi usada a taxa geral do tipo;
- aviso de **estimativa pouco confiável** quando o imóvel foge do padrão dos dados de
  treino (seção 6.5).

## 6.2 Tabela ano a ano e explicação do método

![Tabela e método](figures/16_interface_tabela_metodo.png)

A tabela mostra os quatro cenários para cada ano. A seção "Como a estimativa é feita e
limitações" explica, em linguagem simples, de onde vem cada número e os cuidados na
interpretação.

## 6.3 Valorização por bairro

![Valorização por bairro](figures/17_interface_bairros.png)

Ranking dos bairros pela valorização anual estimada, com o número de imóveis
acompanhados e a faixa de incerteza, para apartamentos e para casas.

## 6.4 Sobre os modelos

![Sobre os modelos](figures/18_interface_modelos.png)

Desempenho do modelo de preço no conjunto de teste (MAE, RMSE e R² por tipo) e os
modelos que compõem o ensemble.

## 6.4.1 Modo avançado

Aba para quem quer ver os modelos por dentro, pedida pelo orientador. Tem três partes:

- **Modelos**: o usuário escolhe o tipo de imóvel e qualquer um dos candidatos (os 9
  algoritmos × 2 alvos, mais o ensemble usado no site e o TabPFN quando incluído). Para o
  modelo escolhido, mostra família, ideia, parâmetros, pontos positivos e negativos;
  métricas na validação cruzada e no teste (MAE, R², erro mediano, previsões a até ±20%);
  o gráfico de preço previsto × real de cada imóvel do teste, com bairro e área no
  cursor; e a distribuição do erro.
- **Comparação**: todos os candidatos lado a lado pelo erro médio no teste, com o ensemble
  em destaque, e a tabela completa de métricas.
- **Dataset**: a base de casas e apartamentos, com filtros por tipo, bairro e faixa de
  preço, mediana de preço e de preço/m² e botão para baixar em CSV. Não inclui código do
  anúncio, endereço, foto nem imobiliária.

Os dados vêm de `dados/resultados_modelos.json` e `dados/dataset_imoveis.csv`, gerados por
`scripts_predict/resultados_modelos.py` (`make resultados-modelos`) com exatamente o
protocolo de avaliação do capítulo 4. As descrições dos modelos ficam em
`interface/modelos_info.py`.

## 6.5 Aviso de dados fora do padrão

O modelo de preço limita cada valor numérico à faixa de 2% a 98% do treino (seção 4.2).
Sem aviso, uma casa de 3.000 m² seria estimada como se tivesse cerca de 1.049 m², e o
site mostraria o valor normalmente. A função `input_warnings`
(`scripts_predict/imoveis_ml_hibrido.py`) compara o que foi digitado com essas faixas,
guardadas no próprio modelo, e a interface mostra um alerta amarelo quando:

- algum campo (área total, área privativa, quartos, banheiros, vagas) está fora da faixa
  do tipo de imóvel — o aviso diz a faixa e o valor que o modelo usou no cálculo;
- a área privativa é maior que a área total;
- o bairro tem menos de 10 imóveis do tipo no treino e entrou no grupo "outros".

| Tipo | Área total no treino | Quartos | Vagas |
|---|---|---|---|
| Apartamento | 39 a 438 m² | 1 a 4 | 1 a 3 |
| Casa | 50 a 1.049 m² | 2 a 5 | 1 a 4,6 |

Valores 0 contam como "não informado" e não geram aviso.

## 6.6 Publicação

O site está publicado no **Streamlit Community Cloud** (gratuito), ligado ao repositório
do GitHub: cada push na branch `main` atualiza o site automaticamente.

- **Modelos versionados**: só o modelo rápido (`preco_imovel_modelo_hibrido_rapido.pkl`,
  24 MB) e o de projeção vão para o GitHub; o modelo com TabPFN (179 MB) fica local.
- **Dados**: os bancos `.db` não são publicados; o site usa `dados/bairros.json` e
  `dados/ipca.json`.
- **Dependências**: `interface/requirements.txt`, com 9 bibliotecas em versões fixas.
- **Versão do Python**: o servidor usa Python 3.14. O CatBoost 1.2.8, usado no treino,
  não tem pacote para essa versão; o site usa o CatBoost 1.2.10, e foi verificado que o
  modelo salvo dá exatamente as mesmas previsões nas duas versões.
- **Desempenho**: cerca de 300 MB de memória. No plano gratuito o site "dorme" sem
  acessos e leva cerca de 30 segundos para voltar.

## 6.7 Decisões de projeto

- **Modelo sem TabPFN na interface**: resposta imediata, com precisão equivalente
  (seção 4.6).
- **Modelos carregados uma única vez** e mantidos em memória (`st.cache_resource`).
- **Segurança**: os modelos são arquivos `.pkl` (que executam código ao serem abertos);
  cada um tem uma assinatura HMAC-SHA256 (`.sig`) e só é carregado se a assinatura
  conferir (`scripts_predict/model_io.py`).
- **Formato brasileiro**: valores em R$ e porcentagens com vírgula decimal.
- **Transparência**: a interface sempre mostra a incerteza (faixa e erro típico) e as
  limitações, em vez de um número único.
