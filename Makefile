PYTHON ?= python3
SCRIPTS_DIR := scripts
IMOVEIS_DB ?= imoveis_$(shell date +%d_%m_%Y).db

# ...

run-casaimoveis:
	IMOVEIS_DB=$(IMOVEIS_DB) $(PYTHON) scripts/casaimoveis_v2.py

run-katedral:
	IMOVEIS_DB=$(IMOVEIS_DB) $(PYTHON) scripts/katedral_v2.py

run-markize:
	IMOVEIS_DB=$(IMOVEIS_DB) $(PYTHON) scripts/markize_v2.py

run-nostracasa:
	IMOVEIS_DB=$(IMOVEIS_DB) $(PYTHON) scripts/nostracasa_v2.py

run-padra:
	IMOVEIS_DB=$(IMOVEIS_DB) $(PYTHON) scripts/padra_v2.py

run-plaza:
	IMOVEIS_DB=$(IMOVEIS_DB) $(PYTHON) scripts/plaza_v2.py

run-santamaria:
	IMOVEIS_DB=$(IMOVEIS_DB) $(PYTHON) scripts/santamaria_v2.py

run-sim:
	IMOVEIS_DB=$(IMOVEIS_DB) $(PYTHON) scripts/sim_v2.py

run-all:
	IMOVEIS_DB=$(IMOVEIS_DB) $(PYTHON) scripts/casaimoveis_v2.py
	IMOVEIS_DB=$(IMOVEIS_DB) $(PYTHON) scripts/katedral_v2.py
	IMOVEIS_DB=$(IMOVEIS_DB) $(PYTHON) scripts/markize_v2.py
	IMOVEIS_DB=$(IMOVEIS_DB) $(PYTHON) scripts/nostracasa_v2.py
	IMOVEIS_DB=$(IMOVEIS_DB) $(PYTHON) scripts/padra_v2.py
	IMOVEIS_DB=$(IMOVEIS_DB) $(PYTHON) scripts/plaza_v2.py
	IMOVEIS_DB=$(IMOVEIS_DB) $(PYTHON) scripts/santamaria_v2.py
	IMOVEIS_DB=$(IMOVEIS_DB) $(PYTHON) scripts/sim_v2.py

unify-db:
	$(PYTHON) scripts_predict/imoveis_ml.py unify --pattern 'imoveis_??_??_????.db' --output-db imoveis_unificado.db

normalize-db:
	$(PYTHON) scripts_predict/imoveis_ml.py unify --pattern 'imoveis_??_??_????.db' --output-db imoveis_unificado.db && $(PYTHON) scripts_predict/imoveis_ml.py normalize --source-db imoveis_unificado.db --output-db imoveis_normalizados.db

benchmark-models:
	$(PYTHON) scripts_predict/imoveis_ml.py benchmark --normalized-db imoveis_normalizados.db --report-path docs/modelo_comparativo.md

benchmark-models-alt:
	$(PYTHON) scripts_predict/imoveis_ml_alternativo.py benchmark-alt --normalized-db imoveis_normalizados.db --report-path docs/modelo_alternativo.md

benchmark-models-hibrido:
	$(PYTHON) scripts_predict/imoveis_ml_hibrido.py benchmark-hibrido --normalized-db imoveis_normalizados.db --report-path docs/modelo_hibrido.md

type-scenarios:
	$(PYTHON) scripts_predict/imoveis_ml.py type-scenarios --normalized-db imoveis_normalizados.db --report-path docs/experimentos_casas_apartamentos.md

type-tuning:
	$(PYTHON) scripts_predict/imoveis_ml.py type-tuning --normalized-db imoveis_normalizados.db --report-path docs/tuning_casas_apartamentos.md

train-price-model:
	$(PYTHON) scripts_predict/imoveis_ml.py train --normalized-db imoveis_normalizados.db --model-path modelos/preco_imovel_modelo.pkl

train-price-model-alt:
	$(PYTHON) scripts_predict/imoveis_ml_alternativo.py train-alt --normalized-db imoveis_normalizados.db --model-path modelos/preco_imovel_modelo_alternativo.pkl

train-price-model-hibrido:
	$(PYTHON) scripts_predict/imoveis_ml_hibrido.py train-hibrido --normalized-db imoveis_normalizados.db --model-path modelos/preco_imovel_modelo_hibrido.pkl

predict-price:
	$(PYTHON) scripts_predict/imoveis_ml.py predict --model-path modelos/preco_imovel_modelo.pkl $(ARGS)

predict-price-alt:
	$(PYTHON) scripts_predict/imoveis_ml_alternativo.py predict-alt --model-path modelos/preco_imovel_modelo_alternativo.pkl $(ARGS)

predict-price-hibrido:
	$(PYTHON) scripts_predict/imoveis_ml_hibrido.py predict-hibrido --model-path modelos/preco_imovel_modelo_hibrido.pkl $(ARGS)

types:
	$(PYTHON) scripts_predict/imoveis_ml.py types

benchmark-valorizacao:
	$(PYTHON) scripts_predict/imoveis_ml.py unify --pattern 'imoveis_??_??_????.db' --output-db imoveis_unificado.db && $(PYTHON) scripts_predict/imoveis_valorizacao.py benchmark-valorizacao --unified-db imoveis_unificado.db $(ARGS)

train-valorizacao:
	$(PYTHON) scripts_predict/imoveis_valorizacao.py train-valorizacao --unified-db imoveis_unificado.db --model-path modelos/valorizacao_modelo.pkl $(ARGS)

predict-valorizacao:
	$(PYTHON) scripts_predict/imoveis_valorizacao.py predict-valorizacao --model-path modelos/valorizacao_modelo.pkl $(ARGS)

train-price-model-rapido:
	DISABLE_TABPFN=1 $(PYTHON) scripts_predict/imoveis_ml_hibrido.py train-hibrido --normalized-db imoveis_normalizados.db --model-path modelos/preco_imovel_modelo_hibrido_rapido.pkl

train-projecao:
	$(PYTHON) scripts_predict/imoveis_projecao.py train-projecao --unified-db imoveis_unificado.db

projetar:
	$(PYTHON) scripts_predict/imoveis_projecao.py projetar $(ARGS)

interface:
	$(PYTHON) -m streamlit run interface/app.py
