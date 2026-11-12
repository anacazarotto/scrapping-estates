# Dados do TCC: casas e apartamentos à venda em Chapecó/SC (2026)

Conjunto de dados do TCC "Modelagem preditiva da valorização imobiliária em Chapecó/SC:
comparação de algoritmos de aprendizado de máquina" (Sistemas de Informação, UNOESC Chapecó).
Autora: Ana Carla Londero Cazarotto. Orientador: Prof. Álvaro Gianni Pagliari.

## Origem

- **Coleta:** anúncios públicos de venda de 8 imobiliárias de Chapecó, lidos por
  *web scraping*. Foram 12 coletas entre 23/03/2026 e 02/08/2026, com 46.383 registros.
- **Limpeza:** unificação das coletas, padronização de bairros e tipos de imóvel e
  deduplicação de anúncios repetidos entre imobiliárias. Restaram 3.283 casas e apartamentos
  com preço e área.
- **Índices:** IPCA (série 433) e INCC (série 192) do Sistema Gerenciador de Séries
  Temporais do Banco Central do Brasil.

Não há endereço, fotos, links dos anúncios nem dados de pessoas. A localização é só o bairro.
As coletas brutas não são publicadas.

## Arquivos

| Arquivo | Versão do modelo | Linhas | Conteúdo |
|---|---|---:|---|
| `dataset_imoveis.csv` | v1 e v2 | 3.283 | Imóveis reais, com o preço anunciado |
| `dataset_simulado_v3.csv` | v3 | 36.113 | Imóveis reais (preço levado a ago/2026) + 10 linhas simuladas por imóvel (jun/2016 a jun/2025), pelo índice IPCA+INCC |
| `dataset_simulado_v4.csv` | v4 | 36.113 | Igual à v3, mas cada imóvel tem a sua taxa: índice + ajuste *fuzzy* |
| `ipca.json`, `series_macro.json` | v2 a v4 | — | Séries do IPCA, INCC e outros índices usados |
| `taxas_fuzzy_v4.json` | v4 | — | Resumo dos ajustes *fuzzy* por tipo e bairro |
| `resultados_modelos*.json` | v1, v3, v4 | — | Ranking e métricas de todos os modelos |

**As linhas simuladas não são preços observados.** São uma hipótese: o imóvel teria
acompanhado o índice (v3) ou o índice mais o ajuste *fuzzy* (v4). A linha do tempo das
versões está em `VERSOES.md`, na raiz do repositório.

## Colunas

| Coluna | Descrição |
|---|---|
| `imovel` | Número do imóvel (liga a linha real às simuladas do mesmo imóvel) |
| `simulado` | 0 = preço real do anúncio; 1 = linha simulada |
| `ano` | Ano com fração do mês (ex.: 2016,4583 = junho de 2016) |
| `mes` | Mês da linha (AAAA-MM) |
| `tipo_imovel` | Apartamento ou Casa |
| `bairro` | Bairro padronizado (vazio quando a imobiliária não informou) |
| `area_total`, `area_privada` | Áreas em m² |
| `quartos`, `banheiros`, `vagas` | Quantidades (0 quando não informado) |
| `preco` | Preço em reais. Linha real: preço levado a ago/2026 pelo índice. Linha simulada: preço simulado |
| `preco_m2` | Preço por m² (só em `dataset_imoveis.csv`) |
| `mes_anuncio` | Mês da coleta do anúncio (só nas linhas reais) |
| `preco_anunciado` | Preço original do anúncio (só nas linhas reais) |
| `ajuste_pp` | Ajuste *fuzzy* da taxa anual, em pontos percentuais (só na v4) |

## Licença

Dados sob a licença Creative Commons Atribuição 4.0 (CC BY 4.0). Ao usar, cite o TCC e
este conjunto de dados.
