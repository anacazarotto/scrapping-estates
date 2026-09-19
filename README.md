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

### 4. Ferramentas de qualidade (Black, isort, Flake8)

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
