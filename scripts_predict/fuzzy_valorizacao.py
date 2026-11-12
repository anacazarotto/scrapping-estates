"""Lógica fuzzy da v4: ajuste da valorização de cada imóvel em torno do índice IPCA+INCC.

Na v3, todo imóvel tem o mesmo passado em %: o preço de cada ano é o preço de hoje
dividido pelo quanto o índice subiu. Na v4, cada imóvel ganha uma taxa própria:

    taxa do imóvel = taxa do índice IPCA+INCC + ajuste fuzzy (pontos percentuais ao ano)

O ajuste vem de regras escritas em linguagem comum ("se o bairro é barato e a tendência
é de alta, a valorização é muito acima do índice"). As regras e as faixas ficam todas
neste arquivo, logo abaixo, para a autora editar.

Como funciona (sistema de Mamdani, em numpy puro, sem dependência nova):
1. Fuzzificação: cada entrada vira graus de pertinência entre 0 e 1. Ex.: um bairro com
   preço/m² 0,85 vezes a mediana do tipo é "barato" com grau 0,33 e "médio" com grau 0,33.
2. Regras: cada regra liga com a força da sua condição (E = menor dos graus).
3. Agregação: as saídas das regras são cortadas na força de cada uma e somadas por
   sobreposição (máximo ponto a ponto).
4. Defuzzificação pelo centroide: o ajuste é o "centro de massa" da área resultante.
5. Centralização por tipo: do ajuste de cada imóvel se tira a média do tipo, então a
   média continua igual ao índice. A fuzzy só redistribui a valorização entre imóveis.

    python scripts_predict/fuzzy_valorizacao.py   # mostra as regras e um exemplo
"""

import numpy as np

INF = 1e9  # "infinito" para os trapézios abertos nas pontas

# =====================================================================================
# FAIXAS (funções de pertinência). Trapézio (a, b, c, d): grau 0 antes de a, sobe até 1
# entre a e b, fica em 1 de b a c e desce até 0 entre c e d. Triângulo = trapézio com b = c.
# =====================================================================================
ENTRADAS = {
    # Preço/m² mediano do BAIRRO dividido pelo preço/m² mediano do TIPO (só imóveis de
    # treino). 1,00 = bairro na média; 0,80 = 20% mais barato. Bairro com menos de
    # MIN_IMOVEIS_BAIRRO imóveis no treino fica com 1,00 (médio).
    "preco_bairro": {
        "barato": (-INF, -INF, 0.80, 0.95),
        "medio": (0.80, 0.95, 1.05, 1.20),
        "caro": (1.05, 1.20, INF, INF),
    },
    # Área do imóvel (privativa; se não houver, a total) dividida pela área mediana do tipo
    # no treino. Apartamento: mediana ~87 m²; casa: ~200 m².
    "area": {
        "pequeno": (-INF, -INF, 0.60, 0.85),
        "medio": (0.60, 0.85, 1.20, 1.60),
        "grande": (1.20, 1.60, INF, INF),
    },
    # Tendência do bairro nos anúncios entre as coletas (mar a ago/2026), em % ao ano,
    # medida nos MESMOS imóveis de treino anunciados em meses diferentes. Como 96% dos
    # preços não mudam de um mês para o outro, o valor é encolhido para zero quando há
    # poucos imóveis (peso n / (n + ENCOLHIMENTO_TENDENCIA)); bairro com menos de
    # MIN_IMOVEIS_TENDENCIA imóveis repetidos fica "estável" (0%).
    "tendencia": {
        "queda": (-INF, -INF, -2.0, -0.5),
        "estavel": (-2.0, -0.5, 0.5, 2.0),
        "alta": (0.5, 2.0, INF, INF),
    },
}

# Saída: ajuste em pontos percentuais ao ano em torno do índice (de -1,5 a +1,5).
SAIDA_MIN, SAIDA_MAX = -1.5, 1.5
SAIDA = {
    "muito_abaixo": (-INF, -INF, -1.5, -0.75),
    "abaixo": (-1.5, -0.75, -0.75, 0.0),
    "neutro": (-0.75, 0.0, 0.0, 0.75),
    "acima": (0.0, 0.75, 0.75, 1.5),
    "muito_acima": (0.75, 1.5, INF, INF),
}

# =====================================================================================
# REGRAS: ({entrada: termo, ...}, termo de saída). Várias condições = E (todas juntas).
# "tipo" (Apartamento/Casa) também pode entrar como condição, ex.: {"tipo": "Casa", ...}.
# =====================================================================================
REGRAS = [
    # Bairro x tendência: bairros mais baratos têm mais espaço para subir (infraestrutura
    # chegando, procura de quem não alcança os bairros caros); bairros caros já estão
    # consolidados. A tendência dos anúncios empurra para cima ou para baixo.
    ({"preco_bairro": "barato", "tendencia": "alta"}, "muito_acima"),
    ({"preco_bairro": "barato", "tendencia": "estavel"}, "acima"),
    ({"preco_bairro": "barato", "tendencia": "queda"}, "neutro"),
    ({"preco_bairro": "medio", "tendencia": "alta"}, "acima"),
    ({"preco_bairro": "medio", "tendencia": "estavel"}, "neutro"),
    ({"preco_bairro": "medio", "tendencia": "queda"}, "abaixo"),
    ({"preco_bairro": "caro", "tendencia": "alta"}, "neutro"),
    ({"preco_bairro": "caro", "tendencia": "estavel"}, "abaixo"),
    ({"preco_bairro": "caro", "tendencia": "queda"}, "muito_abaixo"),
    # Área: imóveis menores têm mais procura (primeiro imóvel, investidor, aluguel) e
    # vendem mais rápido; imóveis grandes têm menos compradores.
    ({"area": "pequeno"}, "acima"),
    ({"area": "medio"}, "neutro"),
    ({"area": "grande"}, "abaixo"),
]

MIN_IMOVEIS_BAIRRO = 10      # mínimo de imóveis de treino para o bairro ter preço/m² próprio
MIN_IMOVEIS_TENDENCIA = 10   # mínimo de imóveis repetidos para o bairro ter tendência própria
ENCOLHIMENTO_TENDENCIA = 20  # quanto maior, mais a tendência é puxada para zero
LIMITE_TENDENCIA = 50.0      # variação anual de um anúncio limitada a ±50% (erros de digitação)

PONTOS = 601  # resolução da saída para o centroide


# ------------------------------------------------------------------------- núcleo
def trapezio(x, a, b, c, d):
    """Grau de pertinência (0 a 1) de x num trapézio (a, b, c, d); vetorizado."""
    x = np.asarray(x, dtype=float)
    sobe = np.ones_like(x) if b <= a else np.clip((x - a) / (b - a), 0, 1)
    desce = np.ones_like(x) if d <= c else np.clip((d - x) / (d - c), 0, 1)
    return np.minimum(sobe, desce)


def fuzzificar(entradas):
    """{entrada: valores} -> {(entrada, termo): graus}."""
    return {
        (nome, termo): trapezio(entradas[nome], *abcd)
        for nome, termos in ENTRADAS.items() if nome in entradas
        for termo, abcd in termos.items()
    }


def forcas_regras(entradas, tipos=None, regras=REGRAS):
    """Força de cada regra para cada imóvel (matriz regras x imóveis)."""
    graus = fuzzificar(entradas)
    n = len(next(iter(entradas.values())))
    tipos = np.asarray(tipos if tipos is not None else [""] * n)
    forcas = []
    for condicao, _ in regras:
        f = np.ones(n)
        for nome, termo in condicao.items():
            f = np.minimum(f, (tipos == termo).astype(float) if nome == "tipo" else graus[(nome, termo)])
        forcas.append(f)
    return np.array(forcas)


def ajuste_bruto(entradas, tipos=None, regras=REGRAS):
    """Ajuste fuzzy (p.p. ao ano) de cada imóvel, antes da centralização."""
    forcas = forcas_regras(entradas, tipos, regras)
    y = np.linspace(SAIDA_MIN, SAIDA_MAX, PONTOS)
    saidas = np.array([trapezio(y, *SAIDA[termo]) for _, termo in regras])  # regras x pontos
    # Corte de cada regra na sua força e agregação pelo máximo: (imóveis x pontos).
    area = np.max(np.minimum(forcas[:, :, None], saidas[:, None, :]), axis=0)
    soma = area.sum(axis=1)
    return np.where(soma > 0, (area * y).sum(axis=1) / np.where(soma > 0, soma, 1), 0.0)


def centralizar(ajuste, tipos):
    """Tira a média do ajuste dentro de cada tipo: a média por tipo fica exatamente 0."""
    ajuste, tipos = np.asarray(ajuste, dtype=float), np.asarray(tipos)
    saida = ajuste.copy()
    for t in np.unique(tipos):
        m = tipos == t
        saida[m] -= ajuste[m].mean()
    return saida


# ----------------------------------------------------- referências tiradas do treino
def referencias(treino, tendencias=None):
    """Medianas do treino usadas como régua (por tipo): preço/m² do tipo e do bairro e área.

    `treino`: imóveis reais de treino (colunas tipo_imovel, bairro, preco, area_privada,
    area_total). `tendencias`: {(tipo, bairro): (variação % ao ano, imóveis)} já medidas.
    """
    t = treino.assign(area_ref=area_ref(treino))
    t = t.assign(pm2=t["preco"] / t["area_ref"])
    ref = {"tipos": {}}
    for tipo, g in t.groupby("tipo_imovel"):
        pm2_tipo = float(g["pm2"].median())
        bairros = g.groupby("bairro")["pm2"].agg(["median", "size"])
        bairros = bairros[bairros["size"] >= MIN_IMOVEIS_BAIRRO]
        ref["tipos"][tipo] = {
            "pm2_tipo": pm2_tipo,
            "area_mediana": float(g["area_ref"].median()),
            "preco_bairro": {b: float(r["median"] / pm2_tipo) for b, r in bairros.iterrows()},
            "tendencia": {
                b: float(v * n / (n + ENCOLHIMENTO_TENDENCIA))
                for (tp, b), (v, n) in (tendencias or {}).items()
                if tp == tipo and n >= MIN_IMOVEIS_TENDENCIA
            },
        }
    return ref


def area_ref(df):
    privada = df["area_privada"].astype(float)
    return privada.where(privada > 0, df["area_total"].astype(float))


def entradas_imoveis(df, ref):
    """Valores das entradas fuzzy para cada imóvel de `df` (usando a régua `ref`)."""
    tipos = df["tipo_imovel"].to_numpy()
    r = [ref["tipos"][t] for t in tipos]
    return {
        "preco_bairro": np.array([x["preco_bairro"].get(b, 1.0) for x, b in zip(r, df["bairro"])]),
        "area": area_ref(df).to_numpy() / np.array([x["area_mediana"] for x in r]),
        "tendencia": np.array([x["tendencia"].get(b, 0.0) for x, b in zip(r, df["bairro"])]),
    }


def ajustes(df, ref):
    """Ajuste final (p.p. ao ano, já centralizado por tipo) de cada imóvel de `df`.

    A centralização usa a média do próprio `df`; para o histórico do treino, `df` é o treino.
    """
    tipos = df["tipo_imovel"].to_numpy()
    return centralizar(ajuste_bruto(entradas_imoveis(df, ref), tipos), tipos)


# --------------------------------------------------------- tendência entre coletas
def tendencias_bairros(unified_db, codigos_validos=None):
    """Variação anual média dos MESMOS anúncios entre a 1ª e a última coleta, por bairro.

    `codigos_validos`: só esses anúncios entram (ex.: os de imóveis de treino), para a
    tendência não usar nada dos imóveis de teste. Retorna {(tipo, bairro): (% ao ano, n)}.
    """
    import sqlite3

    import pandas as pd
    from imoveis_ml import canonical_tipo, normalize_text

    with sqlite3.connect(unified_db) as conn:
        hist = pd.read_sql_query("SELECT codigo, data_coleta, preco FROM historico_precos WHERE preco > 0", conn)
        feats = pd.read_sql_query("SELECT codigo, bairro, tipo_imovel FROM imoveis", conn)
    if codigos_validos is not None:
        hist = hist[hist["codigo"].isin(set(codigos_validos))]
    hist["mes"] = pd.to_datetime(hist["data_coleta"]).dt.to_period("M")
    painel = hist.groupby(["codigo", "mes"], as_index=False)["preco"].median().sort_values(["codigo", "mes"])
    painel["mes_n"] = painel["mes"].dt.year * 12 + painel["mes"].dt.month
    ponta = painel.groupby("codigo").agg(p0=("preco", "first"), p1=("preco", "last"),
                                         m0=("mes_n", "first"), m1=("mes_n", "last"))
    ponta = ponta[ponta["m1"] > ponta["m0"]]
    anual = np.log(ponta["p1"] / ponta["p0"]) * 12 / (ponta["m1"] - ponta["m0"]) * 100
    ponta["anual"] = anual.clip(-LIMITE_TENDENCIA, LIMITE_TENDENCIA)
    feats["bairro"] = feats["bairro"].fillna("").map(normalize_text)
    feats["tipo_imovel"] = feats["tipo_imovel"].fillna("").map(canonical_tipo)
    ponta = ponta.join(feats.set_index("codigo"), how="inner")
    agg = ponta.groupby(["tipo_imovel", "bairro"])["anual"].agg(["mean", "size"])
    return {k: (float(r["mean"]), int(r["size"])) for k, r in agg.iterrows()}


# ------------------------------------------------------------------ explicação
def explicar(entrada, tipo=""):
    """Passo a passo de um imóvel: graus, força de cada regra e ajuste (antes de centralizar)."""
    e = {k: np.array([v]) for k, v in entrada.items()}
    graus = {k: float(v[0]) for k, v in fuzzificar(e).items()}
    forcas = forcas_regras(e, [tipo])[:, 0]
    return {"graus": graus, "regras": [(c, s, float(f)) for (c, s), f in zip(REGRAS, forcas)],
            "ajuste_bruto": float(ajuste_bruto(e, [tipo])[0])}


if __name__ == "__main__":
    ex = explicar({"preco_bairro": 0.85, "area": 0.70, "tendencia": 1.2}, "Apartamento")
    print("Graus:", {f"{k[0]}={k[1]}": round(v, 2) for k, v in ex["graus"].items() if v > 0})
    for cond, saida, f in ex["regras"]:
        if f > 0:
            print(f"  {cond} -> {saida}: força {f:.2f}")
    print(f"Ajuste (antes de centralizar): {ex['ajuste_bruto']:+.2f} p.p./ano")
