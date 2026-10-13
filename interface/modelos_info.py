"""Descrição de cada modelo candidato, mostrada no modo avançado da interface."""

MODELOS = {
    "Regressão Linear Múltipla": {
        "familia": "Linear",
        "ideia": "O preço é uma soma de pesos: cada m², quarto, vaga e bairro soma (ou subtrai) um valor fixo.",
        "parametros": "sem parâmetros",
        "pros": "Fácil de explicar; rápida; boa linha de base.",
        "contras": "Só capta relações em linha reta; muito sensível a valores absurdos.",
    },
    "Ridge": {
        "familia": "Linear com penalização L2",
        "ideia": "Igual à regressão linear, mas impede que os pesos fiquem grandes demais.",
        "parametros": "alpha = 5,0",
        "pros": "Mais estável quando as variáveis andam juntas (área e quartos).",
        "contras": "Continua linear: não capta que o bairro muda o efeito da área.",
    },
    "Lasso": {
        "familia": "Linear com penalização L1",
        "ideia": "Regressão linear que pode zerar o peso de variáveis inúteis.",
        "parametros": "alpha = 0,0005",
        "pros": "Ajuda a interpretar quais variáveis importam.",
        "contras": "Continua linear; com penalização pequena fica igual à regressão linear.",
    },
    "Random Forest": {
        "familia": "Árvores em paralelo (bagging)",
        "ideia": "Treina 400 árvores de decisão, cada uma com um sorteio dos dados, e tira a média.",
        "parametros": "400 árvores, mínimo de 2 imóveis por folha",
        "pros": "Robusto a valores estranhos e a parâmetros; capta relações não lineares.",
        "contras": "Não extrapola além do que viu; arquivo grande.",
    },
    "Gradient Boosting": {
        "familia": "Árvores em sequência (boosting)",
        "ideia": "Cada nova árvore tenta corrigir o erro deixado pelas anteriores.",
        "parametros": "500 árvores, taxa de aprendizado 0,05, profundidade 3",
        "pros": "Costuma ser dos melhores em dados de tabela.",
        "contras": "Treino mais lento; pode decorar o treino com árvores demais.",
    },
    "XGBoost": {
        "familia": "Boosting otimizado",
        "ideia": "Boosting com amostragem de linhas e colunas e regularização.",
        "parametros": "400 árvores, taxa 0,05, profundidade 6, 90% das linhas e colunas por árvore",
        "pros": "Rápido e robusto; muito usado em competições de dados.",
        "contras": "Muitos parâmetros para ajustar.",
    },
    "CatBoost": {
        "familia": "Boosting com árvores simétricas",
        "ideia": "Boosting criado pela Yandex que trata muito bem variáveis de categoria, como o bairro.",
        "parametros": "500 iterações, taxa 0,05, profundidade 6",
        "pros": "Melhor com o bairro; bons resultados sem muito ajuste.",
        "contras": "Treino mais lento; precisou de um adaptador para o scikit-learn 1.6+.",
    },
    "Redes Neurais Artificiais (MLP)": {
        "familia": "Rede neural",
        "ideia": "Camadas de neurônios (128 e 64) que combinam as variáveis de forma flexível.",
        "parametros": "camadas 128 e 64, ativação ReLU, parada antecipada",
        "pros": "Capta padrões muito complexos quando há muitos dados.",
        "contras": "Precisa de milhares de exemplos e ajuste fino; instável em bases pequenas.",
    },
    "Máquinas de Vetores de Suporte (SVM)": {
        "familia": "Vetores de suporte",
        "ideia": "Procura uma curva que passe perto da maioria dos imóveis (kernel RBF).",
        "parametros": "C = 20, epsilon = 0,1",
        "pros": "Bom em bases pequenas e médias com padrão claro.",
        "contras": "Muito sensível à escala do alvo: com o preço em reais falha.",
    },
    "TabPFN v2": {
        "familia": "Modelo de fundação (transformer)",
        "ideia": "Pré-treinado em milhões de tabelas sintéticas; prevê pelo contexto, sem treino nos nossos dados (Nature, 2025).",
        "parametros": "nenhum ajuste",
        "pros": "Venceu a validação cruzada sem nenhuma configuração.",
        "contras": "Lento em CPU (cerca de 2 min por previsão), arquivo de 179 MB.",
    },
    "TabPFN v3.5": {
        "familia": "Modelo de fundação (transformer)",
        "ideia": "Versão 3.5 do TabPFN (set/2026), roda localmente após aceite da licença da Prior Labs.",
        "parametros": "nenhum ajuste",
        "pros": "Estado da arte em dados de tabela.",
        "contras": "Exige aceite de licença; lento em CPU.",
    },
    "TabPFN v3.5 Fast": {
        "familia": "Modelo de fundação (transformer)",
        "ideia": "Variante mais rápida do TabPFN 3.5, com um pouco menos de precisão.",
        "parametros": "nenhum ajuste",
        "pros": "Até 6 vezes mais rápido que o 3.5.",
        "contras": "Exige aceite de licença; um pouco menos preciso.",
    },
    "TabPFN 3.5 Thinking (API)": {
        "familia": "Modelo de fundação (transformer), na nuvem",
        "ideia": "Usa mais processamento no servidor da Prior Labs para melhorar a previsão.",
        "parametros": "thinking_mode = True",
        "pros": "A variante mais precisa do TabPFN 3.5.",
        "contras": "Só pela API paga; os dados de treino são enviados ao servidor.",
    },
    "Ensemble": {
        "familia": "Combinação de modelos",
        "ideia": "Média ponderada dos 3 melhores candidatos da validação cruzada; quem erra menos pesa mais (peso = 1/MAE).",
        "parametros": "3 modelos, pesos por 1/MAE",
        "pros": "Reduz a variação de um modelo isolado; é o modelo usado no site.",
        "contras": "Mais difícil de interpretar que um modelo só.",
    },
}

# Por que cada modelo teve o resultado que teve, com os NOSSOS dados (versão 3:
# anúncios reais + histórico simulado; resultados em dados/resultados_modelos_v3.json).
# A interface junta a posição no ranking (calculada) com estes textos.
RESULTADOS = {
    "CatBoost": {
        "resultado": (
            "É o melhor nos dois tipos de imóvel. O preço depende muito do bairro (até 70% de "
            "diferença no preço/m²), e o CatBoost trata categorias como o bairro melhor que os "
            "outros. Também capta efeitos que não são somas simples, como o preço/m² que não sobe "
            "com o número de quartos."
        ),
        "resultado_log": (
            "Com o alvo em log, foi o melhor em apartamentos, mas caiu bastante em casas: o modelo "
            "em log tende a prever o valor típico e subestima as casas caras, que pesam muito no "
            "erro em reais."
        ),
        "resultado_preco": "Com o alvo em reais, foi o melhor em casas e o 4º em apartamentos.",
    },
    "XGBoost": {
        "resultado": (
            "Fica logo atrás do CatBoost: é o mesmo tipo de modelo (árvores em sequência), mas "
            "trata o bairro como colunas separadas, o que funciona um pouco pior com bairros que "
            "têm poucos anúncios."
        ),
        "resultado_log": "Com o alvo em log, foi bem em apartamentos e mal em casas, pelo mesmo motivo do CatBoost.",
        "resultado_preco": "Com o alvo em reais, ficou entre os 3 primeiros em casas.",
    },
    "Gradient Boosting": {
        "resultado": (
            "Mesma ideia do XGBoost, com árvores menores (profundidade 3). Fica perto dos melhores, "
            "o que mostra que, para estes dados, o tipo de modelo (árvores em sequência) importa "
            "mais que a implementação."
        ),
        "resultado_log": "Com o alvo em log, ficou instável em casas (validação cruzada com R² perto de zero).",
        "resultado_preco": "Com o alvo em reais, entrou no ensemble de casas.",
    },
    "Random Forest": {
        "resultado": (
            "Bom e estável, mas atrás dos modelos de boosting: tira a média de árvores independentes "
            "e não corrige os próprios erros como o boosting."
        ),
        "fora_do_site": (
            "Não entra no site: com o histórico simulado (cerca de 15 mil linhas), o arquivo do modelo "
            "passaria de centenas de MB."
        ),
    },
    "Redes Neurais Artificiais (MLP)": {
        "resultado": (
            "Precisa de muito mais exemplos do que temos (cerca de 1.000 a 1.400 imóveis reais por "
            "tipo) e de ajuste fino. O histórico simulado aumenta as linhas, mas não traz imóveis "
            "novos, então não resolve."
        ),
        "resultado_log": (
            "Com o alvo em log, falhou de forma grave em casas: pequenos erros no logaritmo viram "
            "erros enormes em reais quando convertidos de volta."
        ),
        "resultado_preco": "Com o alvo em reais, ficou atrás dos modelos de árvore com o mesmo alvo.",
        "fora_do_site": "Não entra no site: ficou atrás dos modelos de árvore.",
    },
    "Regressão Linear Múltipla": {
        "resultado": (
            "Cerca de 14% a 22% pior que os melhores: o preço não é uma soma de pesos fixos. A EDA "
            "mostrou que o preço/m² não sobe com o número de quartos e que o efeito da área muda "
            "conforme o bairro, e uma reta não capta isso."
        ),
        "fora_do_site": "Não entra no site: ficou atrás dos modelos de árvore.",
    },
    "Ridge": {
        "resultado": (
            "Praticamente igual à regressão linear: a penalização ajuda quando as variáveis andam "
            "juntas (área e quartos), mas não resolve o problema principal, que é a relação não linear."
        ),
        "fora_do_site": "Foi testado para o site, mas não ficou entre os 3 melhores.",
    },
    "Lasso": {
        "resultado": (
            "Igual à regressão linear: com a penalização pequena usada, nenhuma variável foi "
            "descartada, então o resultado é o mesmo."
        ),
        "fora_do_site": "Não entra no site: ficou atrás dos modelos de árvore.",
    },
    "Máquinas de Vetores de Suporte (SVM)": {
        "resultado": (
            "Ficou entre os piores. O SVM é muito sensível à escala do alvo: os parâmetros usados (C = 20, "
            "epsilon = 0,1) fazem sentido para números pequenos, não para preços de centenas de "
            "milhares de reais."
        ),
        "resultado_log": "Com o alvo em log (números pequenos), melhorou bastante em apartamentos.",
        "resultado_preco": "Com o alvo em reais, errou mais do que prever sempre a média (R² negativo).",
        "fora_do_site": "Não entra no site: ficou entre os piores.",
    },
    "Ensemble": {
        "resultado": (
            "É o modelo do site: a média ponderada dos 3 melhores da validação cruzada entre os que "
            "cabem no site. Em apartamentos erra um pouco menos que o melhor modelo sozinho; em casas "
            "empata com o CatBoost (diferença de 0,3%). A vantagem é a estabilidade: os erros de "
            "modelos diferentes se compensam em parte, e a média não depende de um modelo só."
        ),
    },
}

POR_QUE_GERAL = """
**Por que os resultados são esses, com os dados que temos**

- **Os modelos de árvores em sequência (CatBoost, XGBoost, Gradient Boosting) são os
  melhores.** O preço não é uma soma simples: o bairro muda o efeito da área, e o preço/m²
  não sobe com o número de quartos. Árvores captam isso; modelos lineares não.
- **Os melhores ficam muito perto uns dos outros.** Quando algoritmos diferentes chegam ao
  mesmo erro, o limite está na informação disponível: os anúncios não têm endereço, idade
  nem padrão de acabamento.
- **Apartamentos (R² ≈ 0,80) são bem mais previsíveis que casas (R² ≈ 0,57).** A
  correlação entre área e preço é 0,86 nos apartamentos e 0,69 nas casas, e a categoria
  "casa" mistura padrões muito diferentes.
- **Em casas, prever o logaritmo do preço piora todos os modelos.** O modelo em log
  tende a prever o valor típico e subestima as casas caras, que dominam o erro em reais.
- **O histórico simulado não muda quem é o melhor.** Ele ensina ao modelo o efeito do tempo
  (coerente com o IPCA+INCC), não qual imóvel vale mais. Por isso a precisão de hoje fica
  praticamente igual à da versão anterior (R² 0,80 e 0,58).
- **Todos os erros são medidos só em preços reais**, de imóveis que nenhum modelo viu.
- **O TabPFN não entra na v3:** aceita no máximo 10 mil linhas de treino e, com o histórico
  simulado, levaria horas no computador. Na v1 ele empatou com o ensemble.
"""

ALVOS = {
    "preco": "Prevê o preço em reais.",
    "log(preco)": "Prevê o logaritmo do preço e converte de volta; reduz o peso dos imóveis muito caros.",
    "preco/m2": "Prevê o preço por m² e multiplica pela área.",
    "log(preco/m2)": "Prevê o logaritmo do preço por m², converte e multiplica pela área.",
}


def info(nome):
    """Descrição e resultado do modelo; nomes com sufixo de versão caem na família base."""
    base = nome if nome in MODELOS else next((b for b in MODELOS if nome.startswith(b)), None)
    if base is None:
        return None
    return {**MODELOS[base], **RESULTADOS.get(base, {})}
