# 🏠 Scraping de Imóveis — Chapecó/SC

Scripts de coleta automática de imóveis à venda em Chapecó, com dados salvos em banco SQLite (`imoveis.db`).

## 📋 Fontes coletadas

| Script | Site / Fonte                                                    | Prefixo no banco |
|---|-----------------------------------------------------------------|---|
| `scripts/casaimoveis_v2.py` | [casaimoveis.net](https://www.casaimoveis.net) (HTML) | `C-` |
| `scripts/nostracasa_v2.py` | [nostracasa.com.br](https://nostracasa.com.br) (HTML) | `N-` |
| `scripts/santamaria_v2.py` | Santa Maria Imóveis (API) | `S-` |
| `scripts/padra_v2.py` | Padrá Imóveis (API — markize.simob.com.br) | `P-` |
| `scripts/sim_v2.py` | SIM Imóveis (API) | `SI-` |
| `scripts/markize_v2.py` | Markize Imóveis (API) | `M-` |
| `scripts/katedral_v2.py` | [katedralimoveis.com.br](https://katedralimoveis.com.br) (HTML) | `K-` |
| `scripts/plaza_v2.py` | [plazachapeco.com.br](https://plazachapeco.com.br) (HTML) | `PL-` |

## 🗄️ Estrutura do banco (`imoveis.db`)

| Campo | Tipo | Descrição |
|---|---|---|
| `id` | TEXT | Código único (ex: `C-15299`) |
| `preco` | REAL | Preço de venda (R$) |
| `preco_m2` | REAL | Preço por m² calculado |
| `bairro` | TEXT | Bairro do imóvel |
| `endereco` | TEXT | Endereço textual do imóvel quando exposto pela fonte |
| `imagem_url` | TEXT | URL da imagem principal usada na deduplicação visual |
| `cidade` | TEXT | Cidade |
| `tipo_imovel` | TEXT | Casa, Apartamento, etc. |
| `area_total` | REAL | Área total em m² |
| `area_privada` | REAL | Área privativa em m² |
| `quartos` | INTEGER | Número de quartos |
| `banheiros` | INTEGER | Número de banheiros |
| `vagas` | INTEGER | Vagas de garagem |
| `date_registration` | TEXT | Data de registro na fonte |
| `data_insercao` | TEXT | Data de inserção no banco |

---

## ⚙️ Configuração do ambiente

### 1. Criar o ambiente virtual

```bash
python3 -m venv venv
```

### 2. Ativar o ambiente virtual

```bash
# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

Para o benchmark comparativo, também são necessários:

```bash
pip install scikit-learn xgboost catboost
```

### 4. (Opcional) TabPFN v2

O TabPFN v2 (foundation model para dados tabulares) entra automaticamente nos
benchmarks (`benchmark`, `benchmark-alt`, `benchmark-hibrido`) quando está instalado:

```bash
pip install -r requirements-tabpfn.txt
```

Na primeira execução os pesos `Prior-Labs/TabPFN-v2-reg` são baixados do Hugging Face.
O modelo usa no máximo 10.000 linhas de treino (acima disso, subamostra) e recebe
bairro/tipo como categorias (sem one-hot). Para rodar sem ele: `DISABLE_TABPFN=1`.

### 5. Configuração (`.env`) e tokens

Tokens das APIs das imobiliárias **não ficam mais no código**. Copie o exemplo e preencha:

```bash
cp .env.example .env
```

| Chave | Usada por |
|---|---|
| `MARKIZE_API_TOKEN` | `scripts/markize_v2.py` |
| `PADRA_API_TOKEN` | `scripts/padra_v2.py` |
| `SANTAMARIA_API_TOKEN` | `scripts/santamaria_v2.py` |
| `MODEL_SIGNING_KEY` | assinatura dos modelos `.pkl` (recomendado) |

Os modelos salvos por `train-alt` e `train-hibrido` recebem um arquivo `.sig` ao lado.
O carregamento só acontece se a assinatura conferir, evitando executar um `.pkl`
adulterado (pickle executa código ao ser aberto). Modelos antigos sem `.sig`
precisam ser treinados de novo.

### 6. Ferramentas de qualidade (Black, isort, Flake8)

As ferramentas `black`, `isort` e `flake8` ja estao no `requirements.txt`.

```bash
make format        # aplica isort + black
make lint          # roda flake8
make check-format  # valida formatacao sem alterar arquivos
```

---

## ▶️ Rodar os scripts

Execute cada script a partir da **raiz do projeto** (onde normalmente fica o arquivo do banco).

Nome do arquivo de banco
------------------------
Os scripts agora leem o nome do arquivo de banco a partir da variavel de ambiente `IMOVEIS_DB` ou das chaves `IMOVEIS_DB`, `DB_NAME` / `DATABASE` em um arquivo `.env` na raiz do projeto. Se nenhuma dessas opcoes for encontrada o scraper usara por padrao `imoveis.db`.

Exemplos de `.env` e variavel de ambiente
```bash
# arquivo .env na raiz do projeto
IMOVEIS_DB=meubanco.db

# ou executar diretamente com variavel de ambiente
IMOVEIS_DB=meubanco.db python3 scripts/markize_v2.py
```

Executando os scripts
---------------------
Recomendado: ative um ambiente virtual e use `python3`.

Comandos individuais com Makefile:
```bash
make run-casaimoveis
make run-katedral
make run-markize
make run-nostracasa
make run-padra
make run-plaza
make run-santamaria
make run-sim
```

Comando em lote com Makefile:
```bash
make run-all
```

Comando em lote com variavel de ambiente:
```bash
make run-all IMOVEIS_DB=imoveis2.db
```

## 🤖 Normalização e previsão

### Histórico de preços (base para valorização)

O `unify` agora grava, além da tabela `imoveis` (registro mais recente de cada imóvel),
a tabela `historico_precos` em `imoveis_unificado.db`, com uma linha por imóvel e data de
coleta (extraída do nome `imoveis_DD_MM_AAAA.db`). É a partir dela que se estuda a
valorização mensal. Rode os scrapers periodicamente (`make run-all`) para acumular meses.

### Modelo de valorização (`scripts_predict/imoveis_valorizacao.py`)

Prevê a variação do preço anunciado de um imóvel em `h` meses (padrão 1) a partir do
histórico. Precisa de pelo menos `h + 2` meses de coletas.

1. Copie todos os bancos `imoveis_DD_MM_AAAA.db` das coletas para a raiz do projeto.
2. Rode:

```bash
make benchmark-valorizacao                 # unify + comparação de modelos (teste temporal)
make benchmark-valorizacao ARGS="--horizonte 2"
make train-valorizacao
make predict-valorizacao ARGS="--bairro Centro --tipo-imovel Apartamento --area-privada 90 --quartos 3 --banheiros 2 --vagas 1 --preco-atual 900000 --meses-anunciado 3"
```

Saídas: `docs/reports/modelo_valorizacao.md` (modelos vs. baselines "preço não muda" e
"média do segmento", classificador de chance de redução de preço e índice acumulado por
bairro) e `docs/reports/indice_valorizacao_bairros.csv`.

Validação é temporal: o último mês é o teste e nunca entra no treino. Limitações: preço
anunciado não é preço de venda; imóveis vendidos saem do painel (viés de sobrevivência).

### Projeção ao longo dos anos e interface (`imoveis_projecao.py`, `interface/app.py`)

Estima quanto um imóvel vale hoje e quanto pode valer nos próximos anos:

1. **Valor hoje**: modelo de preço híbrido (`train-hibrido`), a partir de bairro, tipo,
   áreas, quartos, banheiros e vagas.
2. **Valorização anual**: variação do preço anunciado dos *mesmos* imóveis entre as
   coletas, anualizada por bairro + tipo. Bairros com poucos imóveis são puxados para a
   taxa do tipo (peso = n / (n + 30)); a faixa pessimista/otimista vem de bootstrap por
   imóvel (percentis 10 e 90). Relatório: `docs/reports/taxas_valorizacao_anual.md`.
3. **Referência (IPCA)**: cenário "acompanhando a inflação", com o IPCA oficial baixado da API
   do Banco Central (SGS série 433) e guardado em `dados/ipca.json`. Padrão: média anual dos
   últimos 10 anos; opção: acumulado de 12 meses (`--referencia ipca-12m`). Não há FipeZap
   para Chapecó.

```bash
python scripts_predict/imoveis_ml.py normalize --source-db imoveis_unificado.db --output-db imoveis_normalizados.db
DISABLE_TABPFN=1 python scripts_predict/imoveis_ml_hibrido.py train-hibrido --normalized-db imoveis_normalizados.db --model-path modelos/preco_imovel_modelo_hibrido_rapido.pkl
python scripts_predict/imoveis_projecao.py train-projecao
python scripts_predict/imoveis_projecao.py projetar --bairro Centro --tipo-imovel Apartamento --area-privada 90 --area-total 120 --quartos 3 --banheiros 2 --vagas 2 --anos 10
streamlit run interface/app.py
```

O modelo usado na interface e no `projetar` é treinado **sem TabPFN** (`_rapido.pkl`):
na CPU o TabPFN leva ~2 min por previsão, e na mesma base a diferença de erro foi < 1%
(MAE R$ 200,7 mil sem TabPFN contra R$ 199,4 mil com TabPFN). O modelo com TabPFN fica
como resultado do benchmark.

Grafias diferentes do mesmo bairro são unificadas no `unify` (`BAIRRO_ALIASES` em
`scripts_predict/imoveis_ml.py`, ex.: "Presidente Medice" → "Presidente Medici").

A interface abre no navegador (http://localhost:8501) com três abas: estimativa
(formulário + gráfico ano a ano), valorização por bairro e desempenho dos modelos.

### Como as métricas são medidas

Em todos os scripts o teste é separado **antes** de qualquer pré-processamento: bairros
raros, clipagem e limites de outlier (IQR) são aprendidos só no treino. A escolha de
modelos/hiperparâmetros usa validação cruzada no treino; o teste é usado uma única vez.

```bash
make unify-db
make normalize-db
make benchmark-models
make benchmark-models-alt
make benchmark-models-hibrido
make type-scenarios
make type-tuning
make train-price-model
make train-price-model-alt
make train-price-model-hibrido
make types
make predict-price ARGS="--area-total 80.2 --area-privada 80.2 --bairro 'Vila Real' --tipo-imovel Apartamento --quartos 3 --banheiros 1 --vagas 2"
make predict-price-alt ARGS="--area-total 80.2 --area-privada 80.2 --bairro 'Vila Real' --tipo-imovel Apartamento --quartos 3 --banheiros 1 --vagas 2"
make predict-price-hibrido ARGS="--area-total 80.2 --area-privada 80.2 --bairro 'Vila Real' --tipo-imovel Apartamento --quartos 3 --banheiros 1 --vagas 2"
```

Também é possível usar diretamente:

```bash
python3 scripts_predict/imoveis_ml.py unify --pattern "imoveis_??_??_????.db" --output-db imoveis_unificado.db
python3 scripts_predict/imoveis_ml.py normalize --source-db imoveis_unificado.db --output-db imoveis_normalizados.db
python3 scripts_predict/imoveis_ml.py benchmark --normalized-db imoveis_normalizados.db --report-path docs/modelo_comparativo.md
python3 scripts_predict/imoveis_ml_alternativo.py benchmark-alt --normalized-db imoveis_normalizados.db --report-path docs/reports/modelo_alternativo.md
python3 scripts_predict/imoveis_ml_hibrido.py benchmark-hibrido --normalized-db imoveis_normalizados.db --report-path docs/reports/modelo_hibrido.md
python3 scripts_predict/imoveis_ml.py type-scenarios --normalized-db imoveis_normalizados.db --report-path docs/reports/experimentos_casas_apartamentos.md
python3 scripts_predict/imoveis_ml.py type-tuning --normalized-db imoveis_normalizados.db --report-path docs/reports/tuning_casas_apartamentos.md
python3 scripts_predict/imoveis_ml.py train --normalized-db imoveis_normalizados.db --model-path modelos/preco_imovel_modelo.pkl
python3 scripts_predict/imoveis_ml_alternativo.py train-alt --normalized-db imoveis_normalizados.db --model-path modelos/preco_imovel_modelo_alternativo.pkl
python3 scripts_predict/imoveis_ml_hibrido.py train-hibrido --normalized-db imoveis_normalizados.db --model-path modelos/preco_imovel_modelo_hibrido.pkl
python3 scripts_predict/imoveis_ml.py types
python3 scripts_predict/imoveis_ml.py predict --model-path modelos/preco_imovel_modelo.pkl --area-total 80.2 --area-privada 80.2 --bairro "Vila Real" --tipo-imovel Apartamento --quartos 3 --banheiros 1 --vagas 2
python3 scripts_predict/imoveis_ml_alternativo.py predict-alt --model-path modelos/preco_imovel_modelo_alternativo.pkl --area-total 80.2 --area-privada 80.2 --bairro "Vila Real" --tipo-imovel Apartamento --quartos 3 --banheiros 1 --vagas 2
python3 scripts_predict/imoveis_ml_hibrido.py predict-hibrido --model-path modelos/preco_imovel_modelo_hibrido.pkl --area-total 80.2 --area-privada 80.2 --bairro "Vila Real" --tipo-imovel Apartamento --quartos 3 --banheiros 1 --vagas 2
```

Comandos individuais sem Makefile (executar a partir da raiz do projeto):
```bash
python3 scripts/casaimoveis_v2.py   # Casa Imoveis (HTML)
python3 scripts/nostracasa_v2.py    # Nostra Casa (HTML)
python3 scripts/santamaria_v2.py    # Santa Maria Imoveis (API)
python3 scripts/padra_v2.py         # Padra Imoveis (API)
python3 scripts/sim_v2.py           # SIM Imoveis (API)
python3 scripts/markize_v2.py       # Markize Imoveis (API)
python3 scripts/katedral_v2.py      # Katedral Imoveis (HTML)
python3 scripts/plaza_v2.py         # Plaza Imoveis (HTML)
```

Observações
- Os scripts percorrem todas as páginas/registros automaticamente e encerram ao encontrar o fim.
- Os dados são salvos incrementalmente no arquivo de banco (SQLite) usando `INSERT OR REPLACE`.

---

## 📁 Estrutura do projeto

```
scrapping/
├── imoveis.db                  # Banco de dados SQLite (gerado automaticamente)
├── requirements.txt            # Dependencias Python
├── pyproject.toml              # Configuracao do black e isort
├── .flake8                     # Configuracao do flake8
├── Makefile                    # Atalhos de format, lint e execucao
├── README.md
├── apis/                       # Exemplos de retorno das APIs (referencia)
│   ├── padra_markize.json      # Listagem — Padra/Markize
│   ├── padra_markize_detail.json  # Detalhe — Padra/Markize
│   ├── santamaria.json         # Santa Maria Imoveis
│   └── sim.json                # SIM Imoveis
├── scripts/
│   ├── v1                      # Scripts antigos
│   ├── casaimoveis_v2.py       # Scraper — casaimoveis.net
│   ├── nostracasa_v2.py        # Scraper — nostracasa.com.br
│   ├── santamaria_v2.py        # Scraper — Santa Maria Imoveis (API)
│   ├── padra_v2.py             # Scraper — Padra Imoveis (API)
│   ├── sim_v2.py               # Scraper — SIM Imoveis (API)
│   ├── markize_v2.py           # Scraper — Markize Imoveis (API)
│   ├── katedral_v2.py          # Scraper — katedralimoveis.com.br (HTML)
│   └── plaza_v2.py             # Scraper — plazachapeco.com.br (HTML)
└── htmls/                      # Paginas HTML de referencia para desenvolvimento
    ├── casaimoveis/
    │   ├── casaimoveis.html
    │   ├── casaimoveis_ap.html
    │   ├── casaimoveis_imovel.html
    │   └── casaimoveis_sem_imp.html
    ├── nostracasa/
    │   ├── nostracasa_pag_lista_imoveis.html
    │   ├── nostracasa_pag_imovel.html
    │   └── nostracasa_ap.html
    ├── katedral/
    │   ├── katedral.html
    │   ├── katedral_imovel.html
    │   └── katedral_apartamento.html
    └── plaza/
        ├── plaza_list.html
        └── plaza_detail.html
```
