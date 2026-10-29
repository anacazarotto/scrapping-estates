# TCC — Estimativa e valorização de imóveis em Chapecó/SC

TCC de Sistemas de Informação (UNOESC Chapecó) da Ana Carla Londero Cazarotto, com orientação do Prof. Álvaro Gianni Pagliari. A usuária escreve em pt-BR e não é especialista em ML: explique de forma simples.

## Estado atual
- A coleta (scrapers em `scripts/`) **terminou**: não rodar de novo.
- O site publicado mostra só a **v3**: https://scrapping-estates-ayzubn6w4uizag9pbeh5yq.streamlit.app (`interface/app.py`).
- Modelo v3 (`scripts_predict/modelo_v3.py`):
  - preços reais levados a ago/2026 pelo índice IPCA+INCC, com histórico simulado de 2016 a 2025;
  - erro medido só nos preços reais.
- Resultados v3:
  - apartamento: R² 0,799, erro mediano 14,5%;
  - casa: R² 0,576, erro mediano 19,2%;
  - valorização: +4,33%/ano (apartamento) e +4,17%/ano (casa).
- Documentação técnica em `docs/01`–`docs/09` (a v3 está no `docs/09`).

## Artigo do TCC II (`artigo_tcc/`)
- `main.tex` + `referencias.bib`, no modelo UNOESC (abnTeX2 + biber). As figuras vêm de `gerar_figuras.py`.
- Regras do orientador:
  - de 12 a 20 páginas, sem capa e sem abstract;
  - introdução de até 4 parágrafos, sem citações;
  - conclusão sem citações;
  - sem citações diretas.
- O Overleaf gratuito estoura o tempo de compilação. Compilar localmente com o TinyTeX (`%APPDATA%\TinyTeX\bin\windows`): `pdflatex`, `biber`, `pdflatex`, `pdflatex`.
- Depois de compilar, atualizar `artigo_tcc/artigo_tcc.pdf` e o zip `artigo_tcc_overleaf.zip`.

## Combinados
- A usuária faz os merges dos PRs.
- Não enviar o e-mail pessoal dela para nenhum serviço.
- Declarar ao orientador o uso de IA.
