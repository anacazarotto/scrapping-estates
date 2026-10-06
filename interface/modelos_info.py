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

ALVOS = {
    "preco": "Prevê o preço em reais.",
    "log(preco)": "Prevê o logaritmo do preço e converte de volta; reduz o peso dos imóveis muito caros.",
    "preco/m2": "Prevê o preço por m² e multiplica pela área.",
    "log(preco/m2)": "Prevê o logaritmo do preço por m², converte e multiplica pela área.",
}


def info(nome):
    """Descrição do modelo; nomes com sufixo de versão caem na família base."""
    if nome in MODELOS:
        return MODELOS[nome]
    for base, dados in MODELOS.items():
        if nome.startswith(base):
            return dados
    return None
