"""Modo avançado da interface: todos os modelos candidatos e o dataset de treino.

Lê dois arquivos gerados por `scripts_predict/resultados_modelos.py`:
- `dados/resultados_modelos.json`: métricas e previsões no teste de cada candidato;
- `dados/dataset_imoveis.csv`: casas e apartamentos usados no treino.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from modelos_info import ALVOS, info

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTADOS_JSON = PROJECT_ROOT / "dados" / "resultados_modelos.json"
DATASET_CSV = PROJECT_ROOT / "dados" / "dataset_imoveis.csv"
TIPOS = ("Apartamento", "Casa")

COR_DESTAQUE = "#2a78d6"
COR_NEUTRA = "rgba(128, 128, 128, 0.45)"
COR_REFERENCIA = "rgba(128, 128, 128, 0.8)"
SEM_BARRA = {"displayModeBar": False}


def reais(valor):
    return f"R$ {valor:,.0f}".replace(",", ".")


def num(valor, casas=1):
    return f"{valor:.{casas}f}".replace(".", ",")


@st.cache_data(show_spinner=False)
def carregar_resultados():
    if not RESULTADOS_JSON.exists():
        return None
    return json.loads(RESULTADOS_JSON.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def carregar_dataset():
    if not DATASET_CSV.exists():
        return None
    return pd.read_csv(DATASET_CSV)


def rotulo_modelo(m):
    if m["modelo"] == "Ensemble":
        return "Ensemble (o modelo usado no site)"
    return f"{m['modelo']} — alvo {m['alvo']} · {m['posicao_cv']}º na validação cruzada"


def marcas_log(lo, hi):
    """Marcas 1-2-5 em reais para um eixo em escala log (R$ 200 mil, R$ 1 mi...)."""
    vals = [m * 10**e for e in range(4, 9) for m in (1, 2, 5) if lo <= m * 10**e <= hi]

    def texto(v):
        if v >= 1e6:
            return f"R$ {v / 1e6:g} mi".replace(".", ",")
        return f"R$ {v / 1e3:g} mil"

    return vals, [texto(v) for v in vals]


def layout_base(fig, altura, titulo_x, titulo_y):
    fig.update_layout(
        height=altura,
        margin=dict(l=10, r=10, t=30, b=10),
        separators=",.",
        showlegend=False,
        xaxis=dict(title=titulo_x, gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(title=titulo_y, gridcolor="rgba(128,128,128,0.15)"),
    )
    return fig


def grafico_previsto_real(seg, modelo):
    real = np.asarray(seg["real"], dtype=float)
    prev = np.asarray(modelo["previsto"], dtype=float)
    erro = (prev - real) / real * 100
    lim = [max(1.0, min(real.min(), prev.min()) * 0.9), max(real.max(), prev.max()) * 1.1]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=lim, y=lim, mode="lines", hoverinfo="skip",
            line=dict(color=COR_REFERENCIA, width=1, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=real, y=prev, mode="markers",
            marker=dict(size=8, color=COR_DESTAQUE, opacity=0.55,
                        line=dict(width=1, color="rgba(255,255,255,0.8)")),
            customdata=np.column_stack([seg["bairro"], seg["area"], erro]),
            hovertemplate=(
                "Real: R$ %{x:,.0f}<br>Previsto: R$ %{y:,.0f}<br>Erro: %{customdata[2]:+.1f}%"
                "<br>Bairro: %{customdata[0]}<br>Área: %{customdata[1]} m²<extra></extra>"
            ),
        )
    )
    layout_base(fig, 440, "Preço anunciado real (escala log)", "Preço previsto (escala log)")
    vals, textos = marcas_log(*lim)
    eixo = dict(type="log", tickvals=vals, ticktext=textos, range=[np.log10(lim[0]), np.log10(lim[1])])
    fig.update_xaxes(**eixo)
    fig.update_yaxes(**eixo)
    return fig


def grafico_erros(seg, modelo):
    real = np.asarray(seg["real"], dtype=float)
    erro = (np.asarray(modelo["previsto"], dtype=float) - real) / real * 100
    erro = np.clip(erro, -100, 100)
    fig = go.Figure(
        go.Histogram(
            x=erro, xbins=dict(start=-100, end=100, size=5),
            marker=dict(color=COR_DESTAQUE, line=dict(width=2, color="rgba(255,255,255,0.9)")),
            hovertemplate="Erro de %{x}%: %{y} imóveis<extra></extra>",
        )
    )
    fig.add_vline(x=0, line=dict(color=COR_REFERENCIA, width=1, dash="dash"))
    layout_base(fig, 300, "Erro da previsão (%) — negativo = previu abaixo do real", "Imóveis no teste")
    fig.update_xaxes(ticksuffix="%")
    return fig


def grafico_comparacao(seg):
    modelos = sorted(seg["modelos"], key=lambda m: m["teste_tipico"]["mae"], reverse=True)
    # MLP e SVM chegam a milhões de erro: corta o eixo e mostra o valor no rótulo.
    teto = float(np.percentile([m["teste_tipico"]["mae"] for m in modelos], 85)) * 1.15
    nomes = [("★ " if m["modelo"] == "Ensemble" else "") + f"{m['modelo']} ({m['alvo'] if m['modelo'] != 'Ensemble' else 'site'})" for m in modelos]
    maes = [m["teste_tipico"]["mae"] for m in modelos]
    cores = [COR_DESTAQUE if m["modelo"] == "Ensemble" else COR_NEUTRA for m in modelos]
    fig = go.Figure(
        go.Bar(
            x=[min(v, teto) for v in maes], y=nomes, orientation="h",
            marker=dict(color=cores, cornerradius=4),
            text=[reais(v) + (" (fora da escala)" if v > teto else "") for v in maes],
            textposition="outside", cliponaxis=False,
            customdata=[m["teste_tipico"]["r2"] for m in modelos],
            hovertemplate="%{y}<br>Erro médio (MAE): %{text}<br>R²: %{customdata:.3f}<extra></extra>",
        )
    )
    layout_base(fig, 28 * len(modelos) + 60, "Erro médio no teste (MAE, R$) — menor é melhor", "")
    fig.update_xaxes(range=[0, teto * 1.35], tickprefix="R$ ", tickformat=",.0f")
    return fig


def tabela_modelos(seg):
    linhas = []
    for m in seg["modelos"]:
        t, c, f = m["teste_tipico"], m["cv"], m["teste_completo"]
        linhas.append({
            "Modelo": m["modelo"],
            "Alvo": "—" if m["modelo"] == "Ensemble" else m["alvo"],
            "Posição na validação cruzada": str(m["posicao_cv"] or "—"),
            "MAE validação cruzada": reais(c["mae"]) if c else "—",
            "MAE teste": reais(t["mae"]),
            "R² teste": num(t["r2"], 3),
            "Erro mediano": num(t["erro_mediano_pct"]) + "%",
            "Até ±20%": num(t["ate_20_pct"], 0) + "%",
            "R² teste completo": num(f["r2"], 3),
        })
    return pd.DataFrame(linhas)


def aba_modelos(resultados):
    c1, c2 = st.columns([1, 3])
    tipo = c1.radio("Tipo", TIPOS, horizontal=True, key="av_tipo")
    seg = resultados["segmentos"][tipo]
    modelos = seg["modelos"]
    escolha = c2.selectbox("Modelo", range(len(modelos)), format_func=lambda i: rotulo_modelo(modelos[i]), key="av_modelo")
    modelo = modelos[escolha]
    dados = info(modelo["modelo"]) or {}

    e, d = st.columns(2)
    with e:
        st.markdown(f"**{modelo['modelo']}** · {dados.get('familia', '')}")
        st.write(dados.get("ideia", ""))
        if modelo["modelo"] == "Ensemble":
            st.caption(f"Composição: {modelo['alvo']}")
        else:
            st.caption(f"Alvo: {ALVOS.get(modelo['alvo'], modelo['alvo'])} · Parâmetros: {dados.get('parametros', '—')}")
    with d:
        st.markdown(f"**Pontos positivos:** {dados.get('pros', '—')}")
        st.markdown(f"**Pontos negativos:** {dados.get('contras', '—')}")

    t = modelo["teste_tipico"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Erro médio (MAE)", reais(t["mae"]), help="Em média, quanto a previsão erra em reais, no teste.")
    k2.metric("R²", num(t["r2"], 3), help="Quanto da diferença de preço entre imóveis o modelo explica (1 = tudo).")
    k3.metric("Erro mediano", num(t["erro_mediano_pct"]) + "%", help="Metade das previsões erra menos que isso.")
    k4.metric("Previsões a até ±20%", num(t["ate_20_pct"], 0) + "%")
    cv = modelo["cv"]
    st.caption(
        (f"Validação cruzada no treino: MAE {reais(cv['mae'])}, R² {num(cv['r2'], 3)}. " if cv else "")
        + f"Teste completo (com os anúncios fora do padrão): R² {num(modelo['teste_completo']['r2'], 3)}. "
        + f"Teste com {seg['linhas_teste_tipico']} {tipo.lower()}s que o modelo nunca viu."
    )

    st.markdown("**Preço previsto x preço real no teste**")
    st.plotly_chart(grafico_previsto_real(seg, modelo), width="stretch", config=SEM_BARRA)
    st.caption(
        "Cada ponto é um imóvel do teste. A linha tracejada é o acerto perfeito (previsto = real): "
        "quanto mais perto dela, melhor; pontos abaixo dela foram subestimados."
    )
    st.markdown("**Distribuição do erro**")
    st.plotly_chart(grafico_erros(seg, modelo), width="stretch", config=SEM_BARRA)


def aba_comparacao(resultados):
    tipo = st.radio("Tipo", TIPOS, horizontal=True, key="av_tipo_comp")
    seg = resultados["segmentos"][tipo]
    st.plotly_chart(grafico_comparacao(seg), width="stretch", config=SEM_BARRA)
    st.caption("★ = ensemble usado no site. Todos medidos no mesmo teste, que nenhum modelo viu no treino.")
    st.dataframe(tabela_modelos(seg), hide_index=True, width="stretch")


def aba_dataset():
    df = carregar_dataset()
    if df is None:
        st.info("Dataset não encontrado. Gere com: `python scripts_predict/resultados_modelos.py`")
        return
    df = df.assign(bairro=df["bairro"].fillna("").map(lambda b: b.title() if b else "(sem bairro)"))
    c1, c2, c3 = st.columns([1, 2, 2])
    tipos = c1.multiselect("Tipo", list(TIPOS), default=list(TIPOS))
    bairros = c2.multiselect("Bairro", sorted(df["bairro"].unique()), placeholder="Todos")
    teto = int(df["preco"].quantile(0.99) / 1000 / 50 + 1) * 50  # em R$ mil
    faixa = c3.slider(
        "Preço (R$ mil)", 0, teto, (0, teto), step=50,
        help=f"No máximo ({teto}), inclui também os anúncios acima disso, como os com erro de cadastro.",
    )
    filtro = df["tipo_imovel"].isin(tipos) & (df["preco"] >= faixa[0] * 1000)
    if faixa[1] < teto:
        filtro &= df["preco"] <= faixa[1] * 1000
    if bairros:
        filtro &= df["bairro"].isin(bairros)
    sub = df[filtro]

    k1, k2, k3 = st.columns(3)
    k1.metric("Imóveis", f"{len(sub):,}".replace(",", "."))
    k2.metric("Preço mediano", reais(sub["preco"].median()) if len(sub) else "—")
    k3.metric("Preço/m² mediano", reais(sub["preco_m2"].median()) if len(sub) else "—")
    st.dataframe(
        sub,
        hide_index=True,
        width="stretch",
        column_config={
            "tipo_imovel": "Tipo",
            "bairro": "Bairro",
            "area_total": st.column_config.NumberColumn("Área total (m²)", format="%.0f"),
            "area_privada": st.column_config.NumberColumn("Área privativa (m²)", format="%.0f"),
            "quartos": "Quartos",
            "banheiros": "Banheiros",
            "vagas": "Vagas",
            "preco": st.column_config.NumberColumn("Preço (R$)", format="localized"),
            "preco_m2": st.column_config.NumberColumn("Preço/m² (R$)", format="%.0f"),
        },
    )
    st.download_button(
        "Baixar CSV", sub.to_csv(index=False).encode("utf-8"), "dataset_imoveis_chapeco.csv", "text/csv"
    )
    st.caption(
        "Casas e apartamentos após unificação, deduplicação e normalização (antes da limpeza "
        "estatística do treino, por isso ainda há anúncios com erro de cadastro). Sem código do "
        "anúncio, endereço, foto ou imobiliária. Preço anunciado, não de venda."
    )


def tela_avancada():
    st.subheader("Modo avançado")
    resultados = carregar_resultados()
    if resultados is None:
        st.info("Resultados dos modelos não encontrados. Gere com: `python scripts_predict/resultados_modelos.py`")
        aba_dataset()
        return
    st.caption(f"Resultados gerados em {resultados['gerado_em'][:10]}. Protocolo: {resultados['protocolo']}.")
    a1, a2, a3 = st.tabs(["Modelos", "Comparação", "Dataset"])
    with a1:
        aba_modelos(resultados)
    with a2:
        aba_comparacao(resultados)
    with a3:
        aba_dataset()
