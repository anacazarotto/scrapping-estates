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
        "ideia": "Treina 100 árvores de decisão, cada uma com um sorteio dos dados, e tira a média.",
        "parametros": "100 árvores, mínimo de 2 imóveis por folha",
        "pros": "Robusto a valores estranhos e a parâmetros; capta relações não lineares.",
        "contras": "Não extrapola além do que viu; arquivo grande (por isso 100 árvores no site).",
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
        "pros": "Bons resultados sem nenhuma configuração.",
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
    "TabPFN 3.5 (API)": {
        "familia": "Modelo de fundação (transformer), na nuvem",
        "ideia": "O mesmo TabPFN 3.5, rodando no servidor da Prior Labs: usa todas as linhas de treino e responde em segundos.",
        "parametros": "nenhum ajuste; modelo v3.5 padrão da API",
        "pros": "Usa todas as linhas e cabe no site, porque o cálculo pesado fica no servidor.",
        "contras": "Depende da internet e da cota da API; os dados de treino e do imóvel vão para o servidor.",
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
        "ideia": "Média ponderada dos 3 melhores candidatos da validação cruzada; quem erra menos pesa mais (peso = 1/MdAPE). Concorre com os outros modelos pelo mesmo critério.",
        "parametros": "3 modelos, pesos por 1/MdAPE",
        "pros": "Reduz a variação de um modelo isolado.",
        "contras": "Mais difícil de interpretar que um modelo só.",
    },
}

# Por que cada modelo teve o resultado que teve, com os NOSSOS dados (versão 3:
# anúncios reais + histórico simulado; resultados em dados/resultados_modelos_v3.json).
# A interface junta a posição no ranking (calculada) com estes textos.
RESULTADOS = {
    "CatBoost": {
        "resultado": (
            "Fica perto dos melhores. O preço depende muito do bairro (até 70% de diferença no "
            "preço/m²), e o CatBoost trata categorias como o bairro melhor que os outros. Também "
            "capta efeitos que não são somas simples, como o preço/m² que não sobe com o número de "
            "quartos. Tem um dos maiores R² (erra pouco também nos imóveis caros)."
        ),
        "resultado_log": (
            "Com o alvo em log, foi bem em apartamentos, mas caiu em casas: o modelo em log tende a "
            "prever o valor típico e subestima as casas caras."
        ),
        "resultado_preco": "Com o alvo em reais, entra no ensemble de casas, que é a reserva do TabPFN.",
    },
    "XGBoost": {
        "resultado": (
            "Fica colado no CatBoost: é o mesmo tipo de modelo (árvores em sequência). Entra no "
            "ensemble dos dois tipos."
        ),
        "resultado_log": "Com o alvo em log, entra no ensemble de apartamentos e foi mal em casas.",
        "resultado_preco": "Com o alvo em reais, entra no ensemble de casas (a reserva do TabPFN).",
    },
    "Gradient Boosting": {
        "resultado": (
            "Mesma ideia do XGBoost, com árvores menores (profundidade 3). Fica perto dos melhores, "
            "o que mostra que, para estes dados, o tipo de modelo (árvores em sequência) importa "
            "mais que a implementação."
        ),
        "resultado_log": "Com o alvo em log, ficou atrás em casas (MdAPE 22,3%).",
        "resultado_preco": "Com o alvo em reais, ficou logo atrás do CatBoost e do XGBoost.",
    },
    "Random Forest": {
        "resultado": (
            "Pelo MdAPE, foi o melhor modelo único em apartamentos e um dos melhores em casas: "
            "acerta bem o imóvel típico. Tem R² menor que os modelos de boosting, porque erra mais "
            "nos imóveis caros (tira a média de árvores e não corrige os próprios erros). Entra no "
            "ensemble dos dois tipos (duas vezes em apartamentos)."
        ),
        "site": (
            "Usa 100 árvores em vez de 400: o MdAPE mudou menos de 0,2 ponto percentual e o modelo "
            "ficou 4 vezes menor (cerca de 20 MB por arquivo), o que cabe no site gratuito."
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
            "Erra de 2 a 8 pontos percentuais a mais (MdAPE) que os melhores: o preço não é uma "
            "soma de pesos fixos. A EDA mostrou que o preço/m² não sobe com o número de quartos e "
            "que o efeito da área muda conforme o bairro, e uma reta não capta isso."
        ),
        "fora_do_site": "Não entra no site: ficou atrás dos modelos de árvore.",
    },
    "Ridge": {
        "resultado": (
            "Praticamente igual à regressão linear: a penalização ajuda quando as variáveis andam "
            "juntas (área e quartos), mas não resolve o problema principal, que é a relação não linear."
        ),
        "fora_do_site": "Foi testado para o site, mas ficou entre os últimos.",
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
    "TabPFN 3.5 (API)": {
        "resultado": (
            "É o modelo do site para casas: o melhor pelo MdAPE entre os que podem ser usados no "
            "site (19,2% na validação cruzada e 18,7% no teste). Em apartamentos ficou no meio do "
            "ranking (13,9%). Usa todas as linhas de treino e cada parte da validação levou poucos "
            "segundos no servidor da Prior Labs."
        ),
        "site": (
            "No site, as características do imóvel e os anúncios de treino vão para o servidor da "
            "Prior Labs, que devolve a estimativa em segundos. Se a API não responder (sem internet "
            "ou sem cota), o site usa a reserva: o ensemble de Random Forest, CatBoost e XGBoost."
        ),
    },
    "TabPFN": {
        "resultado": (
            "Testado no computador em três versões (v2, 3.5 e 3.5 Fast), no mesmo protocolo dos "
            "outros. O 3.5 foi o 1º de todos em casas (MdAPE 18,6%) e o 4º em apartamentos, "
            "treinando com só 3 mil linhas sorteadas: com todas as linhas não terminou em 1 hora. O "
            "3.5 Fast conseguiu usar todas as linhas, mas errou um pouco mais."
        ),
        "fora_do_site": (
            "Esta versão, que roda no computador, não entra no site: precisa do PyTorch e leva "
            "minutos por previsão. O mesmo TabPFN 3.5, pela API da Prior Labs, entra no site."
        ),
    },
    "Ensemble": {
        "resultado": (
            "Média ponderada dos 3 melhores modelos que rodam no próprio site, concorrendo como mais "
            "um candidato. É o modelo do site para apartamentos (Random Forest com alvo em reais e em "
            "log e XGBoost em log; MdAPE 12,9% na validação cruzada e 13,1% no teste). Em casas, "
            "ficou logo atrás do TabPFN pela API e é a reserva dele. Os erros de modelos diferentes "
            "se compensam em parte, e a média não depende de um modelo só."
        ),
    },
}

POR_QUE_GERAL = """
**Por que os resultados são esses, com os dados que temos**

- **O melhor modelo é escolhido pelo MdAPE** (o erro percentual típico: metade dos imóveis
  erra menos que ele) na validação cruzada; em empate, ganha o menor MAE. O ensemble
  concorre como mais um modelo.
- **No site:** casas usam o TabPFN 3.5 pela API da Prior Labs; apartamentos usam o ensemble
  de Random Forest e XGBoost. Se a API não responder, as casas usam o ensemble de reserva.
- **Os modelos de árvores (Random Forest, CatBoost, XGBoost, Gradient Boosting) e o TabPFN
  são os melhores.** O preço não é uma soma simples: o bairro muda o efeito da área, e o
  preço/m² não sobe com o número de quartos. Modelos lineares não captam isso.
- **Os melhores ficam muito perto uns dos outros.** Quando algoritmos diferentes chegam ao
  mesmo erro, o limite está na informação disponível: os anúncios não têm endereço, idade
  nem padrão de acabamento.
- **MdAPE e R² medem coisas diferentes.** O MdAPE olha o imóvel típico; o R² pesa muito os
  imóveis caros. Os modelos escolhidos pelo MdAPE erram menos no imóvel comum, mas têm R²
  um pouco menor que o CatBoost, que acerta melhor os caros.
- **Apartamentos (R² ≈ 0,78) são bem mais previsíveis que casas (R² ≈ 0,54).** A
  correlação entre área e preço é 0,86 nos apartamentos e 0,69 nas casas, e a categoria
  "casa" mistura padrões muito diferentes.
- **O histórico simulado não muda quem é o melhor.** Ele ensina ao modelo o efeito do tempo
  (coerente com o IPCA+INCC), não qual imóvel vale mais.
- **Todos os erros são medidos só em preços reais**, de imóveis que nenhum modelo viu.
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
    # As variantes do TabPFN dividem o mesmo texto de resultado.
    resultado = RESULTADOS.get(base) or next((v for k, v in RESULTADOS.items() if base.startswith(k)), {})
    return {**MODELOS[base], **resultado}
