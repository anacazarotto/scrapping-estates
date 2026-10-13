"""Interface do TCC (versão 3): valor de um imóvel hoje, no passado e ao longo dos anos.

Rodar a partir da raiz do projeto:
    streamlit run interface/app.py

Usa só a versão 3:
- modelo de preço v3 (`scripts_predict/modelo_v3.py`): anúncios reais + histórico
  simulado pelo índice IPCA+INCC;
- valorização v3 (`scripts_predict/imoveis_projecao.py`, versao="v3"): taxa medida nos
  anúncios combinada com o índice IPCA+INCC.

Precisa de:
    python scripts_predict/modelo_v3.py                           (modelo de preço v3)
    python scripts_predict/imoveis_projecao.py train-projecao     (taxas por bairro)
    python scripts_predict/resultados_v3.py                       (aba de modelos)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts_predict"))

from imoveis_ml_hibrido import format_currency, input_warnings  # noqa: E402
from imoveis_projecao import (  # noqa: E402
    MAX_ANOS,
    ajustar_referencia,
    annual_pct,
    bairros_por_tipo,
    load_projection_artifact,
    project,
    reference_rate,
)
from indice_ipca_incc import resumo as resumo_ipca_incc  # noqa: E402
from ivgr import load_ivgr  # noqa: E402
from modelo_v3 import MEIO_DO_ANO, load_v3, prever_preco_v3  # noqa: E402
from modo_avancado import tela_dataset, tela_modelos  # noqa: E402

MODELO_V3 = Path(os.getenv("MODELO_PRECO_V3", PROJECT_ROOT / "modelos" / "preco_imovel_modelo_v3.pkl"))
PROJECTION_MODEL = Path(os.getenv("MODELO_PROJECAO", PROJECT_ROOT / "modelos" / "projecao_modelo.pkl"))
BAIRROS_JSON = PROJECT_ROOT / "dados" / "bairros.json"
NORMALIZED_DB = PROJECT_ROOT / "imoveis_normalizados.db"
TIPOS = ("Apartamento", "Casa")
VERSAO = "v3"
ANOS_PASSADO = list(range(2016, 2026))  # junho de cada ano simulado; o último ponto é hoje

# Paleta de referência (skill de visualização): série 1 = azul, série 2 = laranja.
COR_CENTRAL = "#2a78d6"
COR_FAIXA = "rgba(42, 120, 214, 0.15)"
COR_REFERENCIA = "#eb6834"

st.set_page_config(page_title="Valor do imóvel ao longo dos anos", page_icon="🏠", layout="wide")


# ------------------------------------------------------------------ dados
@st.cache_resource(show_spinner="Carregando modelos...")
def carregar_modelos():
    faltando = [p.name for p in (MODELO_V3, PROJECTION_MODEL) if not p.exists()]
    if faltando:
        return None, None, faltando
    return load_v3(MODELO_V3), load_projection_artifact(PROJECTION_MODEL), []


@st.cache_data(show_spinner=False)
def carregar_bairros():
    """Lista de bairros por tipo: do JSON publicado ou, localmente, do banco normalizado."""
    if BAIRROS_JSON.exists():
        return json.loads(BAIRROS_JSON.read_text(encoding="utf-8"))
    if NORMALIZED_DB.exists():
        return bairros_por_tipo(NORMALIZED_DB)
    return {t: [] for t in TIPOS}


@st.cache_data(show_spinner=False)
def carregar_ivgr():
    return load_ivgr(atualizar=False)


@st.cache_data(show_spinner=False)
def carregar_ipca_incc():
    return resumo_ipca_incc()


def ano_hoje(proj_art):
    """Mês da última coleta na convenção do modelo v3 (ago/2026 -> 2026 + 7,5/12)."""
    mes = pd.Period(proj_art["ultimo_mes"], freq="M")
    return mes.year + (mes.month - 0.5) / 12


def pct(valor, casas=2):
    """Porcentagem com sinal no formato brasileiro (ex.: +0,65%)."""
    return f"{valor:+.{casas}f}%".replace(".", ",")


def rotulo_bairro(bairro):
    return bairro.title() if bairro else bairro


# --------------------------------------------------------------- gráficos
def grafico_projecao(tabela, taxa_referencia, rotulo_referencia):
    anos = tabela["ano"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=anos, y=tabela["otimista"], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False)
    )
    fig.add_trace(
        go.Scatter(
            x=anos,
            y=tabela["pessimista"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=COR_FAIXA,
            name="Faixa pessimista–otimista",
            customdata=tabela[["otimista"]],
            hovertemplate="Pessimista: R$ %{y:,.0f}<br>Otimista: R$ %{customdata[0]:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=anos,
            y=tabela["central"],
            mode="lines+markers",
            name="Cenário central",
            line=dict(color=COR_CENTRAL, width=2),
            marker=dict(size=8),
            hovertemplate="Central: R$ %{y:,.0f}<extra></extra>",
        )
    )
    if taxa_referencia is not None:
        fig.add_trace(
            go.Scatter(
                x=anos,
                y=tabela["referencia"],
                mode="lines",
                name=f"{rotulo_referencia} ({pct(taxa_referencia)}/ano)",
                line=dict(color=COR_REFERENCIA, width=2, dash="dash"),
                hovertemplate=rotulo_referencia + ": R$ %{y:,.0f}<extra></extra>",
            )
        )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode="x unified",
        separators=",.",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(title="Anos a partir de hoje", dtick=1 if len(anos) <= 16 else 2, showgrid=False),
        yaxis=dict(title="Valor estimado (R$)", tickprefix="R$ ", tickformat=",.0f", gridcolor="rgba(128,128,128,0.2)"),
    )
    return fig


def grafico_passado(passado):
    fig = go.Figure(
        go.Scatter(
            x=passado["Ano"],
            y=passado["Valor estimado"],
            mode="lines+markers",
            line=dict(color=COR_CENTRAL, width=2, dash="dot"),
            marker=dict(size=8),
            hovertemplate="%{x}: R$ %{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        separators=",.",
        showlegend=False,
        xaxis=dict(title="", showgrid=False, type="category"),
        yaxis=dict(title="Valor estimado (R$)", tickprefix="R$ ", tickformat=",.0f", gridcolor="rgba(128,128,128,0.2)"),
    )
    return fig


def tabela_formatada(tabela, rotulo_referencia):
    out = pd.DataFrame({"Ano": tabela["ano"]})
    for col, nome in (
        ("pessimista", "Pessimista"),
        ("central", "Central"),
        ("otimista", "Otimista"),
        ("referencia", rotulo_referencia or "Referência"),
    ):
        if col in tabela:
            out[nome] = tabela[col].map(format_currency)
    return out


# ---------------------------------------------------------------- entradas
def escolher_referencia(proj_art):
    """Linha de comparação no gráfico: IPCA+INCC (padrão), IPCA, IVG-R, outra taxa ou nenhuma."""
    ipca = proj_art.get("ipca")
    ivgr = carregar_ivgr()
    combinado = carregar_ipca_incc()
    opcoes = {}
    if combinado:
        opcoes[f"IPCA+INCC média 10 anos ({pct(combinado['media_anual_pct'])} ao ano)"] = (
            combinado["media_anual_pct"],
            "IPCA+INCC média 10 anos",
        )
    if ipca:
        for escolha in ("ipca-media", "ipca-12m"):
            taxa, rotulo = reference_rate(proj_art, escolha)
            opcoes[f"{rotulo} ({pct(taxa)} ao ano)"] = (taxa, rotulo)
    if ivgr:
        for chave, nome in (("media_anual_pct", "IVG-R média 10 anos"), ("acumulado_12m_pct", "IVG-R 12 meses")):
            opcoes[f"{nome} ({pct(ivgr[chave])} ao ano)"] = (ivgr[chave], nome)
    opcoes["Outra taxa"] = None
    opcoes["Nenhuma"] = (None, None)
    escolha = st.selectbox(
        "Comparar com",
        list(opcoes),
        help=(
            "Linha tracejada no gráfico: quanto o imóvel valeria se só acompanhasse o índice. "
            + (f"IPCA+INCC: {combinado['fonte']}. " if combinado else "")
            + (f"IPCA: {ipca['fonte']}. " if ipca else "")
            + (f"IVG-R: {ivgr['fonte']}." if ivgr else "")
        ),
    )
    if opcoes[escolha] is None:
        taxa = st.number_input("Taxa de referência (% ao ano)", value=5.0, step=0.5)
        return taxa, "Referência"
    return opcoes[escolha]


def escolher_contagem(coluna, rotulo, art, tipo, campo, padrao):
    """Lista suspensa de 0 até o máximo visto no treino para o tipo de imóvel."""
    limites = art["segments"][tipo]["clip_bounds"].get(campo)
    # floor: o máximo oferecido nunca passa do limite de treino (ex.: 4,64 vagas -> 4).
    maximo = max(1, int(np.floor(limites[1]))) if limites else 10
    return coluna.selectbox(
        rotulo,
        list(range(0, maximo + 1)),
        index=min(padrao, maximo),
        format_func=lambda v: "—" if v == 0 else str(v),
        key=f"{campo}_{tipo}",
        help=f"Até {maximo}: o máximo dos {tipo.lower()}s usados no treino. — = não informado.",
    )


# ------------------------------------------------------------------ telas
def tela_estimativa(art, proj_art):
    bairros = carregar_bairros()

    with st.sidebar:
        st.header("Características do imóvel")
        tipo = st.radio("Tipo", TIPOS, horizontal=True)
        opcoes = bairros[tipo] or ["centro"]
        padrao = opcoes.index("centro") if "centro" in opcoes else 0
        bairro = st.selectbox("Bairro", opcoes, index=padrao, format_func=rotulo_bairro)

        c1, c2 = st.columns(2)
        area_privada = c1.number_input(
            "Área privativa (m²)",
            min_value=0.0,
            value=80.0 if tipo == "Apartamento" else 150.0,
            step=5.0,
            format="%.0f",
        )
        area_total = c2.number_input(
            "Área total (m²)",
            min_value=0.0,
            value=100.0 if tipo == "Apartamento" else 300.0,
            step=5.0,
            format="%.0f",
        )
        c1, c2, c3 = st.columns(3)
        quartos = escolher_contagem(c1, "Quartos", art, tipo, "quartos", padrao=2 if tipo == "Apartamento" else 3)
        banheiros = escolher_contagem(c2, "Banheiros", art, tipo, "banheiros", padrao=2)
        vagas = escolher_contagem(c3, "Vagas", art, tipo, "vagas", padrao=1)

        st.header("Projeção")
        anos = st.slider("Horizonte (anos)", min_value=1, max_value=MAX_ANOS, value=10)
        taxa_ref, rotulo_ref = escolher_referencia(proj_art)
        with st.expander("Opções avançadas"):
            usar_preco = st.checkbox("Informar o preço atual em vez de estimar")
            preco_atual = st.number_input(
                "Preço atual (R$)", min_value=0.0, value=500_000.0, step=10_000.0, disabled=not usar_preco
            )

    if area_privada <= 0 and area_total <= 0:
        st.warning("Informe a área privativa ou a área total.")
        return

    dados = dict(
        area_total=area_total,
        area_privada=area_privada,
        bairro=bairro,
        tipo_imovel=tipo,
        quartos=quartos,
        banheiros=banheiros,
        vagas=vagas,
    )
    try:
        res = project(
            proj_art,
            art,
            anos=anos,
            preco_atual=preco_atual if usar_preco and preco_atual > 0 else None,
            taxa_referencia=taxa_ref,
            versao=VERSAO,
            valor_estimado=prever_preco_v3(art, ano=ano_hoje(proj_art), **dados),
            **dados,
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    tabela = res["projecao"]
    final = tabela.iloc[-1]

    st.subheader(f"{tipo} no bairro {rotulo_bairro(bairro)}")
    if not res["usou_preco_informado"]:
        avisos = input_warnings(art, **dados)
        if avisos:
            st.warning(
                "**Estimativa pouco confiável: este imóvel foge do padrão dos dados de treino.**\n\n"
                + "\n".join(f"- {a}" for a in avisos),
                icon="⚠️",
            )
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Valor estimado hoje",
        format_currency(res["valor_hoje"]),
        help=(
            "Modelo de preço v3: média ponderada dos 3 melhores modelos de cada tipo de imóvel (veja a aba Modelos)."
            if not res["usou_preco_informado"]
            else "Preço informado por você."
        ),
    )
    c2.metric(
        f"Em {anos} ano(s) — cenário central",
        format_currency(final["central"]),
        delta=pct(100 * (final["central"] / res["valor_hoje"] - 1), 1),
    )
    c3.metric(
        "Valorização anual (central)",
        pct(res["taxa_anual_central_pct"]),
        help=f"Faixa: {pct(res['taxa_anual_pessimista_pct'])} a {pct(res['taxa_anual_otimista_pct'])} ao ano.",
    )
    st.caption(
        f"Valorização: {res['peso_dados'] * 100:.0f}% vem da variação medida nos anúncios "
        f"({len(proj_art['meses'])} meses de coleta) e {(1 - res['peso_dados']) * 100:.0f}% do índice "
        "IPCA+INCC (inflação e custo da construção) dos últimos 10 anos. Com mais meses de coleta, "
        "o peso dos anúncios aumenta."
    )
    if taxa_ref is not None:
        st.caption(
            f"Se acompanhasse apenas a referência ({rotulo_ref}, {pct(taxa_ref)} ao ano), "
            f"valeria {format_currency(final['referencia'])} em {anos} ano(s)."
        )
    if not res["usou_preco_informado"]:
        st.caption(
            f"Erro típico do modelo de preço para {tipo.lower()}s: "
            f"± {format_currency(res['erro_tipico_modelo_preco'])} (MAE no teste com preços reais)."
        )
    else:
        st.caption(f"O modelo estimaria {format_currency(res['valor_modelo_hoje'])} para este imóvel.")
    if res["taxa_origem"] == "tipo":
        st.info(
            f"O bairro tem poucos imóveis acompanhados ao longo dos meses; foi usada a taxa geral de {tipo.lower()}s."
        )

    st.markdown("**Ao longo dos próximos anos**")
    st.plotly_chart(grafico_projecao(tabela, taxa_ref, rotulo_ref), width="stretch")
    with st.expander("Ver tabela ano a ano"):
        st.dataframe(tabela_formatada(tabela, rotulo_ref), hide_index=True, width="stretch")

    st.markdown("**Quanto este imóvel valia nos anos anteriores**")
    mes_hoje = pd.Period(proj_art["ultimo_mes"], freq="M")
    passado = pd.DataFrame(
        {
            "Ano": [f"jun/{a}" for a in ANOS_PASSADO] + [f"{mes_hoje.strftime('%m/%Y')} (hoje)"],
            "Valor estimado": [prever_preco_v3(art, ano=a + MEIO_DO_ANO, **dados) for a in ANOS_PASSADO]
            + [res["valor_modelo_hoje"]],
        }
    )
    st.plotly_chart(grafico_passado(passado), width="stretch")
    st.caption(
        "Estimativa do modelo v3, treinado com os anúncios reais e com um passado SIMULADO para cada "
        "imóvel: o preço anunciado levado a cada ano pelo índice IPCA+INCC realizado. É coerente com a "
        "inflação e o custo da construção, mas não é um preço observado. Valores de junho de cada ano; "
        "o último ponto é o valor de hoje."
    )

    with st.expander("Como a estimativa é feita e limitações"):
        st.markdown(f"""
1. **Valor hoje**: modelo de preço v3, treinado com os anúncios coletados de 8 imobiliárias de
   Chapecó (bairro, tipo, áreas, quartos, banheiros e vagas) e com um histórico simulado de
   2016 a 2025. Erro medido só em preços reais que o modelo nunca viu.
2. **Índice IPCA+INCC**: média das variações mensais da inflação (IPCA) e do custo da
   construção (INCC), séries do Banco Central. Lógica: o terreno e o bem acompanham a
   inflação; a construção acompanha o custo de construir.
3. **Valorização**: a variação do preço anunciado dos *mesmos* imóveis entre as coletas
   ({", ".join(proj_art["meses"])}) quase não mostra nada, porque 96% dos anúncios mantêm o
   preço de um mês para o outro. Por isso a taxa parte do índice IPCA+INCC (média de 10
   anos) e dá aos anúncios um peso de T / (T + 12), em que T são os meses de coleta. As
   diferenças entre bairros vêm dos anúncios.
4. **Faixa**: combina o intervalo de 80% da taxa medida (bootstrap) com a variação do índice
   em 12 meses (percentis 10 e 90 dos últimos 10 anos).
5. **Passado**: o histórico simulado é uma hipótese (o imóvel teria acompanhado o índice).

**Limitações**: preço anunciado não é preço de venda; poucos meses de coleta; sem endereço,
idade ou acabamento do imóvel. A projeção é um cenário, não uma garantia.
""")


def tela_bairros(proj_art):
    st.subheader("Valorização anual estimada por bairro")
    rows = []
    for key, r in proj_art["segmentos"].items():
        bairro, tipo = key.split("|")
        v3, _ = ajustar_referencia(r, proj_art, "ipca_incc")
        rows.append(
            {
                "Bairro": rotulo_bairro(bairro) or "(sem bairro)",
                "Tipo": tipo,
                "Imóveis acompanhados": r["imoveis"],
                "Central (%/ano)": annual_pct(v3["mensal"]),
                "Pessimista (%/ano)": annual_pct(v3["mensal_baixa"]),
                "Otimista (%/ano)": annual_pct(v3["mensal_alta"]),
            }
        )
    df = pd.DataFrame(rows)
    tipo = st.radio("Tipo", TIPOS, horizontal=True, key="tipo_bairros")
    df = df[df["Tipo"] == tipo].sort_values("Central (%/ano)", ascending=False)
    for col in ("Central (%/ano)", "Pessimista (%/ano)", "Otimista (%/ano)"):
        df[col] = df[col].map(pct)
    geral, _ = ajustar_referencia(proj_art["tipos"][tipo], proj_art, "ipca_incc")
    st.caption(
        f"Taxa geral de {tipo.lower()}s: {pct(annual_pct(geral['mensal']))} ao ano "
        f"({proj_art['tipos'][tipo]['imoveis']} imóveis acompanhados). Cada taxa combina a variação "
        "medida nos anúncios do bairro com o índice IPCA+INCC (ver \"Como a estimativa é feita\")."
    )
    st.dataframe(df.drop(columns="Tipo"), hide_index=True, width="stretch")


def main():
    st.title("🏠 Quanto o imóvel pode valer ao longo dos anos")
    st.caption(
        "Imóveis à venda em Chapecó/SC — estimativa por aprendizado de máquina. "
        "Projeto acadêmico (TCC): os valores são estimativas a partir de preços anunciados "
        "e não substituem uma avaliação profissional do imóvel."
    )

    art, proj_art, faltando = carregar_modelos()
    if faltando:
        st.error(
            "Modelos não encontrados: "
            + ", ".join(faltando)
            + ". Treine antes com:\n\n"
            "`python scripts_predict/modelo_v3.py`\n\n"
            "`python scripts_predict/imoveis_projecao.py train-projecao`"
        )
        return

    aba1, aba2, aba3, aba4 = st.tabs(["Estimativa", "Valorização por bairro", "Modelos", "Dataset"])
    with aba1:
        tela_estimativa(art, proj_art)
    with aba2:
        tela_bairros(proj_art)
    with aba3:
        tela_modelos()
    with aba4:
        tela_dataset()


main()
