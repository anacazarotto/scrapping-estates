"""Gera as figuras do artigo a partir dos resultados da v3 e dos bancos locais (análise exploratória).

Uso: python artigo_tcc/gerar_figuras.py
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
SAIDA = Path(__file__).resolve().parent / "Figuras"
AZUL = "#2a6fb0"
CINZA = "#9aa3ad"
CLARO = "#d5dae0"
TINTA = "#222222"

LARANJA = "#e07b39"
COR_TIPO = {"Apartamento": AZUL, "Casa": LARANJA}

ESTILO = {
    "font.family": "serif",
    "font.size": 10,
    "axes.edgecolor": "#888888",
    "axes.labelcolor": TINTA,
    "xtick.color": TINTA,
    "ytick.color": TINTA,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.formatter.use_locale": False,
    "axes.titleweight": "normal",
    "figure.dpi": 100,
    "savefig.bbox": "standard",
}
plt.rcParams.update(ESTILO)
VIRGULA = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))

NOMES = {
    "Redes Neurais Artificiais (MLP)": "Rede Neural (MLP)",
    "Máquinas de Vetores de Suporte (SVM)": "SVM",
    "Regressão Linear Múltipla": "Regressão Linear",
}

# Os bairros vêm sem acento da base (mesmo tratamento dos modelos); na figura, com a grafia correta
BAIRROS = {
    "jardim italia": "Jardim Itália", "lider": "Líder", "paraiso": "Paraíso",
    "passo dos fortes": "Passo dos Fortes", "presidente medici": "Presidente Médici",
    "santo antonio": "Santo Antônio", "sao cristovao": "São Cristóvão", "sao lucas": "São Lucas",
    "seminario": "Seminário", "universitario": "Universitário",
}

# Variação anual do índice IPCA+INCC (docs/09, seção 9.2)
INDICE = {2016: 6.20, 2017: 3.60, 2018: 3.79, 2019: 4.23, 2020: 6.65,
          2021: 11.95, 2022: 7.53, 2023: 4.06, 2024: 5.68, 2025: 5.10}


def rotulo(m):
    if m["modelo"] == "Ensemble":
        return "Ensemble"
    if m.get("api"):
        return f"{m['modelo'].replace(' (API)', '')} pela API ({m['linhas_treino']:,} linhas)".replace(",", ".")
    if m["modelo"].startswith("TabPFN"):
        return f"{m['modelo'].replace('v3.5', '3.5')} local ({m['linhas_treino']:,} linhas)".replace(",", ".")
    alvo = "log" if m["alvo"].startswith("log") else "preço"
    return f"{NOMES.get(m['modelo'], m['modelo'])} ({alvo})"


def ranking_mdape(dados):
    corte = 30
    fig, eixos = plt.subplots(1, 2, figsize=(9, 6.2))
    for ax, seg in zip(eixos, ["Apartamento", "Casa"]):
        modelos = sorted(dados["segmentos"][seg]["modelos"], key=lambda m: m["posicao_cv"])
        nomes = [rotulo(m) for m in modelos]
        mdape = [m["cv"]["mdape"] for m in modelos]
        cores = [AZUL if m.get("escolhido") else (CINZA if m.get("no_site", True) else CLARO) for m in modelos]
        y = range(len(modelos))
        ax.barh(y, [min(v, corte) for v in mdape], color=cores, height=0.7)
        for i, v in enumerate(mdape):
            texto = f"{v:.1f}%".replace(".", ",")
            ax.text(min(v, corte) + 0.3, i, texto, va="center", fontsize=7.5, color=TINTA)
        ax.set_yticks(list(y), nomes, fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlim(0, corte + 6)
        ax.xaxis.set_major_formatter(VIRGULA)
        ax.set_title(seg, fontsize=10)
        ax.set_xlabel("MdAPE na validação cruzada (%)")
        ax.grid(axis="x", color="#e5e5e5", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.text(0.01, 0.005, "Ordem: MdAPE na validação cruzada (empate decidido pelo MAE). Barras acima de 30% foram cortadas.\n"
             "Azul: modelo do site; cinza-escuro: candidatos do site; cinza-claro: fora do site.", fontsize=7.5, color="#555555")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(SAIDA / "v3_ranking_mdape.png", dpi=200)
    plt.close(fig)


def previsto_real(dados):
    fig, eixos = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, seg in zip(eixos, ["Apartamento", "Casa"]):
        segmento = dados["segmentos"][seg]
        # Modelo publicado no site (o ensemble em apartamentos, o TabPFN pela API em casas)
        escolhido = next(m for m in segmento["modelos"] if m.get("escolhido"))
        real = [v / 1e6 for v in segmento["real"]]
        prev = [v / 1e6 for v in escolhido["previsto"]]
        pares = [(r, p) for r, p in zip(real, prev) if r > 0.05]
        ax.scatter([r for r, _ in pares], [p for _, p in pares], s=10, color=AZUL, alpha=0.55,
                   edgecolors="none")
        lim = max(max(r for r, _ in pares), max(p for _, p in pares)) * 1.05
        ax.plot([0, lim], [0, lim], color="#555555", linewidth=1, linestyle="--")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_title(f"{seg}: {rotulo(escolhido)}", fontsize=10)
        ax.xaxis.set_major_formatter(VIRGULA)
        ax.yaxis.set_major_formatter(VIRGULA)
        ax.set_xlabel("Preço anunciado (R$ milhões)")
        ax.set_ylabel("Preço estimado (R$ milhões)")
        ax.grid(color="#eeeeee", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(SAIDA / "v3_previsto_vs_real.png", dpi=200)
    plt.close(fig)


def indice():
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    anos = list(INDICE)
    valores = list(INDICE.values())
    ax.bar(anos, valores, color=AZUL, width=0.65)
    for a, v in zip(anos, valores):
        ax.text(a, v + 0.2, f"{v:.2f}%".replace(".", ","), ha="center", fontsize=8, color=TINTA)
    ax.axhline(5.80, color="#555555", linewidth=1, linestyle="--")
    ax.text(2017.5, 6.1, "média de 10 anos: 5,80%", fontsize=8, ha="center", color="#555555")
    ax.set_xticks(anos)
    ax.set_ylabel("Variação no ano (%)")
    ax.set_ylim(0, 13.5)
    ax.grid(axis="y", color="#eeeeee", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(SAIDA / "v3_indice_ipca_incc.png", dpi=200)
    plt.close(fig)


def imoveis_tipicos():
    """Casas e apartamentos com preço e área plausíveis (mesma base da análise exploratória)."""
    sys.path.insert(0, str(RAIZ / "scripts_predict"))
    import eda  # noqa: E402 - precisa dos bancos locais (imoveis_*.db)

    eda.UNIFIED_DB = RAIZ / "imoveis_unificado.db"
    eda.NORMALIZED_DB = RAIZ / "imoveis_normalizados.db"
    plt.rcParams.update(ESTILO)  # o eda.py troca o estilo ao ser importado
    norm, _, _ = eda.carregar()
    return eda.casas_aptos(norm)


def reais(v):
    if v >= 1e6:
        return f"R$ {v / 1e6:.2f}".replace(".", ",") + (" milhão" if v < 2e6 else " milhões")
    return f"R$ {v / 1e3:.0f} mil"


def distribuicao_precos(ca):
    fig, eixos = plt.subplots(1, 2, figsize=(9, 3.6))
    faixas = np.logspace(np.log10(5e4), np.log10(2e7), 45)
    for tipo, cor in COR_TIPO.items():
        g = ca[ca["tipo_imovel"] == tipo]
        eixos[0].hist(g["preco"], bins=faixas, color=cor, alpha=0.55,
                      label=f"{tipo} (mediana {reais(g['preco'].median())})")
        m2 = g["preco_m2"].median()
        eixos[1].hist(g["preco_m2"].clip(upper=25_000), bins=45, color=cor, alpha=0.55,
                      label=f"{tipo} (mediana R$ {m2:,.0f}/m²)".replace(",", "."))
    eixos[0].set_xscale("log")
    eixos[0].set_xticks([1e5, 1e6, 1e7], ["R$ 100 mil", "R$ 1 milhão", "R$ 10 milhões"])
    eixos[0].xaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
    eixos[0].set_title("Preço anunciado (escala logarítmica)", fontsize=10)
    eixos[1].set_title("Preço por m² (acima de R$ 25 mil agrupados)", fontsize=10)
    eixos[1].set_xticks(range(0, 25_001, 5000), ["0", "5", "10", "15", "20", "25+"])
    eixos[0].set_xlabel("Preço anunciado")
    eixos[1].set_xlabel("Preço por m² (R$ mil)")
    eixos[0].legend(frameon=False, fontsize=8, loc="upper left")
    eixos[1].legend(frameon=False, fontsize=8, loc="upper right")
    for ax in eixos:
        ax.set_ylabel("Quantidade de imóveis")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.25)  # espaço para a legenda acima das barras
    fig.tight_layout()
    fig.savefig(SAIDA / "eda_04_distribuicao_precos.png", dpi=200)
    plt.close(fig)


def bairros(ca):
    fig, eixos = plt.subplots(1, 2, figsize=(9, 4.6))
    for ax, (tipo, cor) in zip(eixos, COR_TIPO.items()):
        g = ca[(ca["tipo_imovel"] == tipo) & (ca["bairro"] != "")]
        stats = g.groupby("bairro")["preco_m2"].agg(["median", "size"])
        stats = stats[stats["size"] >= 20].sort_values("median").tail(15)
        nomes = [BAIRROS.get(b, b.title()) for b in stats.index]
        ax.barh(nomes, stats["median"], color=cor, height=0.65)
        for y, (v, n) in enumerate(zip(stats["median"], stats["size"])):
            ax.text(v, y, f"  {v:,.0f} (n={n})".replace(",", "."), va="center", fontsize=8, color=TINTA)
        ax.set_xlim(0, stats["median"].max() * 1.4)
        ax.set_xticks([0, 4000, 8000], ["0", "4.000", "8.000"])
        ax.tick_params(axis="y", labelsize=8.5)
        ax.set_title(tipo, fontsize=10)
        ax.set_xlabel("Preço mediano por m² (R$)")
    fig.tight_layout()
    fig.savefig(SAIDA / "eda_08_preco_m2_bairros.png", dpi=200)
    plt.close(fig)


def correlacoes(ca):
    """Correlação de Spearman entre as variáveis numéricas (campos vazios ignorados)."""
    cols = {"preco": "Preço", "area_ref": "Área", "quartos": "Quartos", "banheiros": "Banheiros", "vagas": "Vagas"}
    fig, eixos = plt.subplots(1, 2, figsize=(9, 3.9), gridspec_kw={"wspace": 0.4})
    for ax, tipo in zip(eixos, COR_TIPO):
        g = ca[ca["tipo_imovel"] == tipo][list(cols)].replace(0, np.nan)
        corr = g.corr("spearman").rename(index=cols, columns=cols)
        im = ax.imshow(corr.values, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(cols)), corr.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(cols)), corr.index)
        for i in range(len(cols)):
            for j in range(len(cols)):
                v = corr.values[i, j]
                ax.text(j, i, f"{v:.2f}".replace(".", ","), ha="center", va="center", fontsize=8.5,
                        color="white" if v > 0.6 else TINTA)
        ax.set_title(tipo, fontsize=10)
        ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=eixos, shrink=0.85, label="Correlação de Spearman", format=VIRGULA)
    fig.savefig(SAIDA / "eda_07_correlacoes.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    dados = json.loads((RAIZ / "dados" / "resultados_modelos_v3.json").read_text(encoding="utf-8"))
    SAIDA.mkdir(exist_ok=True)
    ranking_mdape(dados)
    previsto_real(dados)
    indice()
    ca = imoveis_tipicos()
    distribuicao_precos(ca)
    bairros(ca)
    correlacoes(ca)


if __name__ == "__main__":
    main()
