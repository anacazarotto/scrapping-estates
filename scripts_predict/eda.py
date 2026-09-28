"""Análise exploratória de dados (EDA) para o TCC.

Gera as figuras `docs/figures/eda_*.png` e imprime as estatísticas usadas em
`docs/02_analise_exploratoria.md`.

    python scripts_predict/eda.py
"""

import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from imoveis_ml import normalize_text  # noqa: E402
from imoveis_projecao import repeat_pairs  # noqa: E402
from imoveis_valorizacao import load_panel  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

UNIFIED_DB = Path("imoveis_unificado.db")
NORMALIZED_DB = Path("imoveis_normalizados.db")
OUT = Path("docs/figures")

AZUL, LARANJA, AQUA, CINZA, TEXTO = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#9a9993",
    "#52514e",
)
COR_TIPO = {"Apartamento": AZUL, "Casa": LARANJA}
FONTES = {
    "N": "Nostra Casa",
    "PL": "Plaza",
    "C": "Casa Imóveis",
    "SI": "SIM",
    "SM": "Santa Maria",
    "S": "Santa Maria",
    "P": "Padrá",
    "M": "Markize",
    "K": "Katedral",
}
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": CINZA,
        "axes.labelcolor": TEXTO,
        "xtick.color": TEXTO,
        "ytick.color": TEXTO,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "figure.dpi": 130,
        "savefig.bbox": "tight",
    }
)


def br(v, d=0):
    return f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_reais_mil(v, _):
    return f"R$ {br(v / 1000)} mil" if v < 1e6 else f"R$ {br(v / 1e6, 1)} mi"


def salvar(fig, nome, titulo):
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=11)
    fig.savefig(OUT / nome)
    plt.close(fig)


def carregar():
    with sqlite3.connect(NORMALIZED_DB) as conn:
        norm = pd.read_sql_query("SELECT * FROM imoveis_normalizados", conn)
    with sqlite3.connect(UNIFIED_DB) as conn:
        unif = pd.read_sql_query("SELECT codigo, tipo_imovel FROM imoveis", conn)
        hist = pd.read_sql_query(
            "SELECT codigo, data_coleta FROM historico_precos", conn
        )
    for c in ("preco", "area_total", "area_privada", "quartos", "banheiros", "vagas"):
        norm[c] = pd.to_numeric(norm[c], errors="coerce").fillna(0)
    norm["area_ref"] = norm["area_privada"].where(
        norm["area_privada"] > 0, norm["area_total"]
    )
    # Mesmo tratamento dos modelos: sem acento e minúsculas ("Médici" = "Medici").
    norm["bairro"] = norm["bairro"].fillna("").map(normalize_text)
    return norm, unif, hist


def casas_aptos(norm):
    """Casas e apartamentos com preço e área plausíveis (visão 'típica' para gráficos)."""
    ca = norm[norm["tipo_imovel"].isin(COR_TIPO)].copy()
    ca = ca[(ca["preco"] >= 50_000) & (ca["preco"] <= 20_000_000)]
    ca = ca[(ca["area_ref"] >= 20) & (ca["area_ref"] <= 2_000)]
    ca["preco_m2"] = ca["preco"] / ca["area_ref"]
    return ca


def fig_funil(norm, unif, hist, ca):
    etapas = [
        ("Registros coletados (12 coletas)", len(hist)),
        ("Imóveis únicos (unificação por código)", len(unif)),
        ("Após normalização e deduplicação", len(norm)),
        ("Casas e apartamentos", int(norm["tipo_imovel"].isin(COR_TIPO).sum())),
        ("Com preço e área plausíveis", len(ca)),
    ]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    nomes = [e[0] for e in etapas][::-1]
    vals = [e[1] for e in etapas][::-1]
    ax.barh(nomes, vals, color=[AZUL] + [CINZA] * 4, height=0.6)
    for y, v in enumerate(vals):
        ax.text(v, y, f"  {br(v)}", va="center", color=TEXTO)
    ax.set_xlim(0, max(vals) * 1.15)
    ax.set_xlabel("Quantidade de registros / imóveis")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: br(v)))
    salvar(
        fig, "eda_01_funil_dados.png", "Figura E1 — Da coleta bruta à base de modelagem"
    )


def fig_fontes_tipos(unif, norm):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), gridspec_kw={"wspace": 0.45})
    fontes = (
        unif["codigo"].str.split("-").str[0].map(FONTES).value_counts().sort_values()
    )
    axes[0].barh(fontes.index, fontes.values, color=AZUL, height=0.6)
    for y, v in enumerate(fontes.values):
        axes[0].text(v, y, f"  {br(v)}", va="center", color=TEXTO, fontsize=9)
    axes[0].set_title("Imóveis únicos por imobiliária")
    axes[0].set_xlim(0, fontes.max() * 1.18)
    tipos = norm["tipo_imovel"].value_counts().sort_values()
    cores = [AZUL if t in COR_TIPO else CINZA for t in tipos.index]
    axes[1].barh(tipos.index, tipos.values, color=cores, height=0.6)
    for y, v in enumerate(tipos.values):
        axes[1].text(
            v,
            y,
            f"  {br(v)} ({br(100 * v / tipos.sum(), 1)}%)",
            va="center",
            color=TEXTO,
            fontsize=9,
        )
    axes[1].set_title("Tipos de imóvel (base normalizada)")
    axes[1].set_xlim(0, tipos.max() * 1.35)
    salvar(
        fig,
        "eda_02_fontes_tipos.png",
        "Figura E2 — Origem dos anúncios e tipos de imóvel (azul: tipos modelados)",
    )


def fig_completude(norm):
    ca = norm[norm["tipo_imovel"].isin(COR_TIPO)]
    campos = {
        "Endereço": ca["endereco"].fillna("").eq(""),
        "Vagas": ca["vagas"].eq(0),
        "Banheiros": ca["banheiros"].eq(0),
        "Quartos": ca["quartos"].eq(0),
        "Área privativa": ca["area_privada"].eq(0),
        "Área total": ca["area_total"].eq(0),
        "Bairro": ca["bairro"].fillna("").eq(""),
        "Preço": ca["preco"].le(0),
    }
    pct = pd.Series({k: 100 * v.mean() for k, v in campos.items()}).sort_values()
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    cores = [LARANJA if v >= 20 else AZUL for v in pct.values]
    ax.barh(pct.index, pct.values, color=cores, height=0.6)
    for y, v in enumerate(pct.values):
        ax.text(v, y, f"  {br(v, 1)}%", va="center", color=TEXTO, fontsize=9)
    ax.set_xlim(0, 115)
    ax.set_xlabel("% de casas e apartamentos com o campo vazio ou zero")
    salvar(
        fig,
        "eda_03_campos_faltantes.png",
        "Figura E3 — Campos faltantes (laranja: 20% ou mais)",
    )
    return pct


def fig_distribuicoes(ca):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4), gridspec_kw={"wspace": 0.3})
    bins = np.logspace(np.log10(5e4), np.log10(2e7), 45)
    for tipo, cor in COR_TIPO.items():
        g = ca[ca["tipo_imovel"] == tipo]
        axes[0].hist(
            g["preco"],
            bins=bins,
            color=cor,
            alpha=0.55,
            label=f"{tipo} (mediana R$ {br(g['preco'].median() / 1000)} mil)",
        )
        axes[1].hist(
            g["preco_m2"].clip(upper=25_000),
            bins=45,
            color=cor,
            alpha=0.55,
            label=f"{tipo} (mediana R$ {br(g['preco_m2'].median())}/m²)",
        )
    axes[0].set_xscale("log")
    axes[0].xaxis.set_major_formatter(FuncFormatter(fmt_reais_mil))
    axes[0].set_title("Preço anunciado (escala logarítmica)")
    axes[1].set_title("Preço por m² (valores acima de R$ 25 mil agrupados)")
    axes[1].xaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"R$ {br(v / 1000)} mil")
    )
    for ax in axes:
        ax.set_ylabel("Quantidade de imóveis")
        ax.legend(frameon=False, fontsize=8.5)
    salvar(
        fig,
        "eda_04_distribuicao_precos.png",
        "Figura E4 — Distribuição de preços: assimétrica, com cauda longa de imóveis caros",
    )


def fig_preco_area(ca):
    fig, axes = plt.subplots(
        1, 2, figsize=(11.5, 4.3), gridspec_kw={"wspace": 0.3}, sharey=True
    )
    for ax, (tipo, cor) in zip(axes, COR_TIPO.items()):
        g = ca[ca["tipo_imovel"] == tipo]
        ax.scatter(
            g["area_ref"], g["preco"], s=8, alpha=0.35, color=cor, edgecolor="none"
        )
        rho = g[["area_ref", "preco"]].corr("spearman").iloc[0, 1]
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{tipo}: correlação de Spearman = {br(rho, 2)}")
        ax.set_xlabel("Área (m², escala log)")
        ax.set_xticks([30, 50, 100, 200, 500, 1000])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: br(v)))
        ax.xaxis.set_minor_formatter(NullFormatter())
    axes[0].set_yticks([2e5, 5e5, 1e6, 2e6, 5e6, 1e7])
    axes[0].yaxis.set_major_formatter(FuncFormatter(fmt_reais_mil))
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylabel("Preço anunciado (escala log)")
    salvar(
        fig,
        "eda_05_preco_vs_area.png",
        "Figura E5 — Área é a variável mais associada ao preço",
    )


def fig_quartos(ca):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"wspace": 0.3})
    for ax, (tipo, cor) in zip(axes, COR_TIPO.items()):
        g = ca[(ca["tipo_imovel"] == tipo) & ca["quartos"].between(1, 5)]
        grupos = [
            g.loc[g["quartos"] == q, "preco_m2"].clip(upper=25_000) for q in range(1, 6)
        ]
        bp = ax.boxplot(grupos, patch_artist=True, showfliers=False, widths=0.55)
        for patch in bp["boxes"]:
            patch.set(facecolor=cor, alpha=0.35, edgecolor=cor)
        for med in bp["medians"]:
            med.set(color=TEXTO, linewidth=1.5)
        ax.set_xticks(
            range(1, 6), [f"{q}\n(n={len(x)})" for q, x in zip(range(1, 6), grupos)]
        )
        ax.set_xlabel("Quartos")
        ax.set_title(tipo)
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _: f"R$ {br(v / 1000)} mil")
        )
    axes[0].set_ylabel("Preço por m²")
    salvar(
        fig,
        "eda_06_preco_m2_por_quartos.png",
        "Figura E6 — Preço por m² por número de quartos (caixa: 50% centrais)",
    )


def fig_correlacao(ca):
    cols = {
        "preco": "Preço",
        "area_ref": "Área",
        "quartos": "Quartos",
        "banheiros": "Banheiros",
        "vagas": "Vagas",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={"wspace": 0.35})
    for ax, tipo in zip(axes, COR_TIPO):
        g = ca[ca["tipo_imovel"] == tipo][list(cols)].replace(0, np.nan)
        corr = g.corr("spearman").rename(index=cols, columns=cols)
        im = ax.imshow(corr.values, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(cols)), corr.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(cols)), corr.index)
        for i in range(len(cols)):
            for j in range(len(cols)):
                v = corr.values[i, j]
                ax.text(
                    j,
                    i,
                    br(v, 2),
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="white" if v > 0.6 else TEXTO,
                )
        ax.set_title(tipo)
        ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=axes, shrink=0.8, label="Correlação de Spearman")
    salvar(
        fig,
        "eda_07_correlacoes.png",
        "Figura E7 — Correlações entre as variáveis (campos vazios ignorados)",
    )


def fig_bairros(ca):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={"wspace": 0.55})
    tabelas = {}
    for ax, (tipo, cor) in zip(axes, COR_TIPO.items()):
        g = ca[(ca["tipo_imovel"] == tipo) & (ca["bairro"].fillna("") != "")]
        stats = g.groupby("bairro")["preco_m2"].agg(["median", "size"])
        stats = stats[stats["size"] >= 20].sort_values("median").tail(15)
        tabelas[tipo] = stats
        ax.barh(
            [b.title() for b in stats.index], stats["median"], color=cor, height=0.6
        )
        for y, (v, n) in enumerate(zip(stats["median"], stats["size"])):
            ax.text(v, y, f"  R$ {br(v)} (n={n})", va="center", fontsize=8, color=TEXTO)
        ax.set_xlim(0, stats["median"].max() * 1.45)
        ax.set_title(f"{tipo}: mediana do preço/m²")
        ax.set_xticks([0, 4000, 8000])
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda v, _: f"R$ {br(v / 1000)} mil")
        )
    salvar(
        fig,
        "eda_08_preco_m2_bairros.png",
        "Figura E8 — Bairros mais caros por m² (bairros com 20+ anúncios)",
    )
    return tabelas


def fig_temporal(hist):
    panel = load_panel(UNIFIED_DB)
    pares = repeat_pairs(panel)
    var = 100 * np.expm1(pares["dlog"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw={"wspace": 0.35})
    por_mes = panel.groupby(panel["mes"].astype(str))["codigo"].nunique()
    axes[0].bar(por_mes.index, por_mes.values, color=AZUL, width=0.6)
    for x, v in enumerate(por_mes.values):
        axes[0].text(x, v, br(v), ha="center", va="bottom", fontsize=8.5, color=TEXTO)
    axes[0].set_title("Imóveis anunciados por mês")
    axes[0].tick_params(axis="x", rotation=30)
    meses = panel.groupby("codigo")["mes"].nunique().value_counts().sort_index()
    axes[1].bar(meses.index.astype(str), meses.values, color=AZUL, width=0.6)
    for x, v in enumerate(meses.values):
        axes[1].text(x, v, br(v), ha="center", va="bottom", fontsize=8.5, color=TEXTO)
    axes[1].set_title("Em quantos meses cada imóvel apareceu")
    axes[1].set_xlabel("Meses (de 5 possíveis)")
    mudou = var[var.abs() > 1e-9]
    axes[2].hist(mudou.clip(-30, 30), bins=40, color=LARANJA, alpha=0.8)
    axes[2].axvline(0, color=TEXTO, lw=0.8)
    axes[2].set_title(
        f"Variação quando o preço muda\n({br(100 * len(mudou) / len(var), 1)}% das comparações)"
    )
    axes[2].set_xlabel("Variação entre coletas (%, limitada a ±30%)")
    salvar(
        fig,
        "eda_09_dinamica_temporal.png",
        "Figura E9 — Dinâmica no tempo: a maioria dos anúncios permanece e raramente muda de preço",
    )
    return pares, var, por_mes, meses


def main():
    norm, unif, hist = carregar()
    ca = casas_aptos(norm)
    fig_funil(norm, unif, hist, ca)
    fig_fontes_tipos(unif, norm)
    pct = fig_completude(norm)
    fig_distribuicoes(ca)
    fig_preco_area(ca)
    fig_quartos(ca)
    fig_correlacao(ca)
    tabelas = fig_bairros(ca)
    pares, var, por_mes, meses = fig_temporal(hist)

    print("coletas:", sorted(hist["data_coleta"].str[:10].unique()))
    print(
        "registros brutos",
        len(hist),
        "| únicos",
        len(unif),
        "| normalizados",
        len(norm),
        "| casas+aptos típicos",
        len(ca),
    )
    print("campos vazios (%):", pct.round(1).to_dict())
    for tipo in COR_TIPO:
        g = ca[ca["tipo_imovel"] == tipo]
        print(
            tipo,
            g[["preco", "area_ref", "preco_m2"]]
            .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
            .round(0)
            .to_dict(),
        )
        print(
            " skew log-preço %.2f | skew preço %.2f"
            % (np.log(g["preco"]).skew(), g["preco"].skew())
        )
        print(
            " top bairros preço/m²:", tabelas[tipo]["median"].round(0).tail(5).to_dict()
        )
    bruto = norm[norm["tipo_imovel"].isin(COR_TIPO)]
    print(
        "preço < 50 mil:",
        int((bruto["preco"] < 50_000).sum()),
        "| > 20 mi:",
        int((bruto["preco"] > 20_000_000).sum()),
        "| área < 20:",
        int((bruto["area_ref"] < 20).sum()),
        "| área > 2000:",
        int((bruto["area_ref"] > 2000).sum()),
    )
    print("cidades:", norm["cidade"].fillna("").str.lower().value_counts().to_dict())
    print(
        "pares",
        len(var),
        "sem mudança %.1f%% | subiu %.1f%% | caiu %.1f%%"
        % (
            100 * (var.abs() < 1e-9).mean(),
            100 * (var > 1e-9).mean(),
            100 * (var < -1e-9).mean(),
        ),
    )
    print("imóveis por mês", por_mes.to_dict(), "| meses por imóvel", meses.to_dict())


if __name__ == "__main__":
    main()
