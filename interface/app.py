"""Interface do TCC: estimativa do valor de um imóvel hoje e ao longo dos anos.

Rodar a partir da raiz do projeto:
    streamlit run interface/app.py

Precisa dos modelos treinados:
    python scripts_predict/imoveis_ml_hibrido.py train-hibrido   (valor hoje)
    python scripts_predict/imoveis_projecao.py train-projecao     (valorização anual)
"""

import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts_predict"))

from imoveis_ml import canonical_tipo, normalize_text  # noqa: E402
from imoveis_ml_hibrido import format_currency  # noqa: E402
from imoveis_ml_hibrido import load_artifact as load_price_artifact  # noqa: E402
from imoveis_projecao import (  # noqa: E402
    MAX_ANOS,
    annual_pct,
    load_projection_artifact,
    project,
    reference_rate,
)

PRICE_MODEL = Path(
    os.getenv(
        "MODELO_PRECO",
        PROJECT_ROOT / "modelos" / "preco_imovel_modelo_hibrido_rapido.pkl",
    )
)
PROJECTION_MODEL = Path(
    os.getenv("MODELO_PROJECAO", PROJECT_ROOT / "modelos" / "projecao_modelo.pkl")
)
NORMALIZED_DB = PROJECT_ROOT / "imoveis_normalizados.db"
TIPOS = ("Apartamento", "Casa")
MIN_IMOVEIS_BAIRRO = 3

# Paleta de referência (skill de visualização): série 1 = azul, série 2 = laranja.
COR_CENTRAL = "#2a78d6"
COR_FAIXA = "rgba(42, 120, 214, 0.15)"
COR_REFERENCIA = "#eb6834"

st.set_page_config(
    page_title="Valor do imóvel ao longo dos anos", page_icon="🏠", layout="wide"
)


# ------------------------------------------------------------------ dados
@st.cache_resource(show_spinner="Carregando modelos...")
def carregar_modelos():
    faltando = [p.name for p in (PRICE_MODEL, PROJECTION_MODEL) if not p.exists()]
    if faltando:
        return None, None, faltando
    return (
        load_price_artifact(PRICE_MODEL),
        load_projection_artifact(PROJECTION_MODEL),
        [],
    )


@st.cache_data(show_spinner=False)
def bairros_por_tipo():
    """Bairros com anúncios suficientes em cada tipo, para a lista de seleção."""
    if not NORMALIZED_DB.exists():
        return {t: [] for t in TIPOS}
    with sqlite3.connect(NORMALIZED_DB) as conn:
        df = pd.read_sql_query(
            "SELECT bairro, tipo_imovel FROM imoveis_normalizados", conn
        )
    df["bairro"] = df["bairro"].fillna("").map(normalize_text)
    df["tipo_imovel"] = df["tipo_imovel"].fillna("").map(canonical_tipo)
    df = df[df["bairro"] != ""]
    out = {}
    for tipo in TIPOS:
        counts = df.loc[df["tipo_imovel"] == tipo, "bairro"].value_counts()
        out[tipo] = sorted(counts[counts >= MIN_IMOVEIS_BAIRRO].index)
    return out


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
        go.Scatter(
            x=anos,
            y=tabela["otimista"],
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        )
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
            hovertemplate=(
                "Pessimista: R$ %{y:,.0f}<br>Otimista: R$ %{customdata[0]:,.0f}<extra></extra>"
            ),
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
        xaxis=dict(
            title="Anos a partir de hoje",
            dtick=1 if len(anos) <= 16 else 2,
            showgrid=False,
        ),
        yaxis=dict(
            title="Valor estimado (R$)",
            tickprefix="R$ ",
            tickformat=",.0f",
            gridcolor="rgba(128,128,128,0.2)",
        ),
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


def escolher_referencia(proj_art):
    """Seletor do cenário de referência: IPCA (padrão), outra taxa ou nenhum."""
    ipca = proj_art.get("ipca")
    opcoes = {}
    if ipca:
        for escolha in ("ipca-media", "ipca-12m"):
            taxa, rotulo = reference_rate(proj_art, escolha)
            opcoes[f"{rotulo} ({pct(taxa)} ao ano)"] = (taxa, rotulo)
    opcoes["Outra taxa"] = None
    opcoes["Nenhuma"] = (None, None)
    escolha = st.selectbox(
        "Comparar com",
        list(opcoes),
        help=(
            f"IPCA: {ipca['fonte']}, {ipca['inicio']} a {ipca['fim']}."
            if ipca
            else "IPCA indisponível: rode train-projecao com internet."
        ),
    )
    if opcoes[escolha] is None:
        taxa = st.number_input("Taxa de referência (% ao ano)", value=5.0, step=0.5)
        return taxa, "Referência"
    return opcoes[escolha]


# ------------------------------------------------------------------ telas
def tela_estimativa(price_art, proj_art):
    bairros = bairros_por_tipo()

    with st.sidebar:
        st.header("Características do imóvel")
        tipo = st.radio("Tipo", TIPOS, horizontal=True)
        opcoes = bairros[tipo] or ["centro"]
        padrao = opcoes.index("centro") if "centro" in opcoes else 0
        bairro = st.selectbox("Bairro", opcoes, index=padrao, format_func=rotulo_bairro)

        c1, c2 = st.columns(2)
        area_privada = c1.number_input(
            "Área privativa (m²)", min_value=0.0, value=80.0, step=5.0
        )
        area_total = c2.number_input(
            "Área total (m²)",
            min_value=0.0,
            value=100.0 if tipo == "Apartamento" else 300.0,
            step=5.0,
        )
        c1, c2, c3 = st.columns(3)
        quartos = c1.number_input("Quartos", min_value=0, max_value=10, value=2)
        banheiros = c2.number_input("Banheiros", min_value=0, max_value=10, value=2)
        vagas = c3.number_input("Vagas", min_value=0, max_value=10, value=1)

        st.header("Projeção")
        anos = st.slider("Horizonte (anos)", min_value=1, max_value=MAX_ANOS, value=10)
        taxa_ref, rotulo_ref = escolher_referencia(proj_art)
        with st.expander("Opções avançadas"):
            usar_preco = st.checkbox("Informar o preço atual em vez de estimar")
            preco_atual = st.number_input(
                "Preço atual (R$)",
                min_value=0.0,
                value=500_000.0,
                step=10_000.0,
                disabled=not usar_preco,
            )

    if area_privada <= 0 and area_total <= 0:
        st.warning("Informe a área privativa ou a área total.")
        return

    try:
        res = project(
            proj_art,
            price_art,
            bairro=bairro,
            tipo_imovel=tipo,
            area_total=area_total,
            area_privada=area_privada,
            quartos=quartos,
            banheiros=banheiros,
            vagas=vagas,
            anos=anos,
            preco_atual=preco_atual if usar_preco and preco_atual > 0 else None,
            taxa_referencia=taxa_ref,
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    tabela = res["projecao"]
    final = tabela.iloc[-1]

    st.subheader(f"{tipo} no bairro {rotulo_bairro(bairro)}")
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Valor estimado hoje",
        format_currency(res["valor_hoje"]),
        help=(
            "Modelo de preço híbrido (ensemble por tipo de imóvel)."
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

    if taxa_ref is not None:
        final_ref = final["referencia"]
        st.caption(
            f"Se acompanhasse apenas a inflação ({rotulo_ref}, {pct(taxa_ref)} ao ano), "
            f"valeria {format_currency(final_ref)} em {anos} ano(s)."
        )
    if not res["usou_preco_informado"]:
        st.caption(
            f"Erro típico do modelo de preço para {tipo.lower()}s: "
            f"± {format_currency(res['erro_tipico_modelo_preco'])} (MAE no teste)."
        )
    else:
        st.caption(
            f"O modelo estimaria {format_currency(res['valor_modelo_hoje'])} para este imóvel."
        )
    if res["taxa_origem"] == "tipo":
        st.info(
            f"O bairro tem poucos imóveis acompanhados ao longo dos meses; foi usada a taxa geral "
            f"de {tipo.lower()}s."
        )

    st.plotly_chart(grafico_projecao(tabela, taxa_ref, rotulo_ref), width="stretch")

    with st.expander("Ver tabela ano a ano"):
        st.dataframe(
            tabela_formatada(tabela, rotulo_ref), hide_index=True, width="stretch"
        )

    with st.expander("Como a estimativa é feita e limitações"):
        st.markdown(f"""
1. **Valor hoje**: modelo de preço treinado com anúncios coletados de imobiliárias de Chapecó
   (bairro, tipo, áreas, quartos, banheiros e vagas).
2. **Valorização**: variação do preço anunciado dos *mesmos* imóveis entre as coletas
   ({", ".join(proj_art["meses"])}), anualizada. Bairros com poucos imóveis são aproximados da
   taxa do tipo de imóvel.
3. **Faixa**: intervalo de 80% da taxa estimada por bootstrap (pessimista e otimista).
4. **Referência**: IPCA oficial (IBGE, via Banco Central), que mostra quanto o imóvel valeria
   se apenas acompanhasse a inflação. Não há índice FipeZap para Chapecó.

**Limitações**: poucos meses de coleta extrapolados para anos; preço anunciado não é preço de
venda; imóveis vendidos saem da base; anúncios raramente mudam de preço, o que tende a
subestimar a valorização. A projeção é um cenário, não uma garantia.
""")


def tela_bairros(proj_art):
    st.subheader("Valorização anual estimada por bairro")
    rows = []
    for key, r in proj_art["segmentos"].items():
        bairro, tipo = key.split("|")
        rows.append(
            {
                "Bairro": rotulo_bairro(bairro) or "(sem bairro)",
                "Tipo": tipo,
                "Imóveis": r["imoveis"],
                "Central (%/ano)": round(annual_pct(r["mensal"]), 2),
                "Pessimista (%/ano)": round(annual_pct(r["mensal_baixa"]), 2),
                "Otimista (%/ano)": round(annual_pct(r["mensal_alta"]), 2),
            }
        )
    df = pd.DataFrame(rows)
    tipo = st.radio("Tipo", TIPOS, horizontal=True, key="tipo_bairros")
    df = df[df["Tipo"] == tipo].sort_values("Central (%/ano)", ascending=False)
    geral = proj_art["tipos"][tipo]
    st.caption(
        f"Taxa geral de {tipo.lower()}s: {pct(annual_pct(geral['mensal']))} ao ano "
        f"({geral['imoveis']} imóveis acompanhados)."
    )
    st.dataframe(df.drop(columns="Tipo"), hide_index=True, width="stretch")


def tela_modelos(price_art, proj_art):
    st.subheader("Desempenho do modelo de preço (conjunto de teste)")
    rows = []
    for tipo, seg in price_art["segments"].items():
        m = seg["metrics"]
        rows.append(
            {
                "Tipo": tipo,
                "Imóveis no treino": seg["rows"],
                "MAE": format_currency(m["mae"]),
                "RMSE": format_currency(m["rmse"]),
                "R²": round(m["r2"], 3),
                "Modelos no ensemble": ", ".join(
                    f"{s['model_name']} ({s['target']})" for s in seg["selected"]
                ),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        f"Modelo de preço treinado em {price_art['trained_at'][:10]}; "
        f"taxas de valorização calculadas em {proj_art['trained_at'][:10]} "
        f"com coletas até {proj_art['ultimo_mes']}."
    )


def main():
    st.title("🏠 Quanto o imóvel pode valer ao longo dos anos")
    st.caption("Imóveis à venda em Chapecó/SC — estimativa por aprendizado de máquina")

    price_art, proj_art, faltando = carregar_modelos()
    if faltando:
        st.error(
            "Modelos não encontrados: "
            + ", ".join(faltando)
            + ". Treine antes com:\n\n"
            "`python scripts_predict/imoveis_ml_hibrido.py train-hibrido`\n\n"
            "`python scripts_predict/imoveis_projecao.py train-projecao`"
        )
        return

    aba1, aba2, aba3 = st.tabs(
        ["Estimativa", "Valorização por bairro", "Sobre os modelos"]
    )
    with aba1:
        tela_estimativa(price_art, proj_art)
    with aba2:
        tela_bairros(proj_art)
    with aba3:
        tela_modelos(price_art, proj_art)


main()
