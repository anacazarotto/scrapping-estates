# Documentação do TCC — Estimativa do valor de imóveis em Chapecó/SC

| Capítulo | Conteúdo |
|---|---|
| [1. Coleta e preparação dos dados](01_coleta_e_preparacao_dos_dados.md) | Fontes, coletas, unificação, deduplicação e normalização |
| [2. Análise exploratória de dados](02_analise_exploratoria.md) | Origem, qualidade, distribuição de preços, correlações, bairros e dinâmica no tempo |
| [3. Evolução dos modelos de preço](03_evolucao_dos_modelos_de_preco.md) | Experimentos, lições aprendidas e a correção da avaliação (vazamento de dados) |
| [4. Modelo de preço atual](04_modelo_de_preco_hibrido.md) | Arquitetura do híbrido, como funciona cada modelo, ranking com TabPFN, resultados e erro por faixa de preço |
| [5. Valorização e projeção](05_valorizacao_e_projecao.md) | Previsão de 1 e 3 meses, taxa anual por bairro, cenários e IPCA |
| [6. Interface](06_interface.md) | Telas da aplicação Streamlit, aviso de dados fora do padrão, publicação e decisões de projeto |
| [7. Resultados e conclusões](07_resultados_e_conclusoes.md) | Principais descobertas, respostas às perguntas, limitações e trabalhos futuros |

## Pastas

- `figures/` — gráficos usados nos capítulos (`eda_*`: capítulo 2; `01`–`06`: capítulo 3;
  `11`–`14`: capítulos 4 e 5; `15`–`18`: capítulo 6).
- `reports/` — relatórios gerados automaticamente pelos scripts de treino e benchmark.

## Como gerar novamente as figuras

```bash
python scripts_predict/eda.py                                   # figuras eda_*
DISABLE_TABPFN=1 python scripts_predict/figuras_resultados.py   # figuras 11 a 14
```
