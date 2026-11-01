# TCC — Estimativa e valorização de imóveis em Chapecó/SC

TCC de Sistemas de Informação (UNOESC Chapecó) da Ana Carla Londero Cazarotto, com orientação do Prof. Álvaro Gianni Pagliari. A usuária escreve em pt-BR e não é especialista em ML: explique de forma simples.

## Estado atual
- A coleta (scrapers em `scripts/`) **terminou**: não rodar de novo.
- O site publicado mostra só a **v3**: https://scrapping-estates-ayzubn6w4uizag9pbeh5yq.streamlit.app (`interface/app.py`).
- Modelo v3 (`scripts_predict/modelo_v3.py`):
  - preços reais levados a ago/2026 pelo índice IPCA+INCC, com histórico simulado de 2016 a 2025;
  - erro medido só nos preços reais.
- Escolha do modelo: menor MdAPE na validação cruzada; empate (mesmo MdAPE com 1 casa) decidido pelo menor MAE. O ensemble (3 melhores que rodam no site, peso 1/MdAPE) concorre como candidato.
- TabPFN local (v2, 3.5, 3.5 Fast) testado com corte de 60 min por tentativa (`scripts_predict/tabpfn_v3.py`, resultados em `dados/tabpfn_v3/`); entra no ranking, mas não no site.
- TabPFN 3.5 pela API da Prior Labs (`TABPFN_VERSIONS=api`, `tabpfn-client`): pode ir ao site. A chave fica nos Secrets do Streamlit (`TABPFN_TOKEN`), e a usuária é quem cadastra. Sem API, o site usa a reserva (melhor modelo local).
- Random Forest no site com 100 árvores; cada estimador fica num arquivo comprimido em `modelos/v3/`.
- Modelos do site (teste):
  - apartamento: ensemble de Random Forest (preço e log) + XGBoost (log), R² 0,781, MdAPE 13,1%;
  - casa: TabPFN 3.5 pela API, R² 0,540, MdAPE 18,7% (reserva: ensemble RF + CatBoost + XGBoost);
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
