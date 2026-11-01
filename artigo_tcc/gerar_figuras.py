"""Gera as figuras do artigo a partir dos resultados da v3.

Uso: python artigo_tcc/gerar_figuras.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
SAIDA = Path(__file__).resolve().parent / "Figuras"
AZUL = "#2a6fb0"
CINZA = "#9aa3ad"
CLARO = "#d5dae0"
TINTA = "#222222"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.edgecolor": "#888888",
    "axes.labelcolor": TINTA,
    "xtick.color": TINTA,
    "ytick.color": TINTA,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.formatter.use_locale": False,
})
VIRGULA = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))

NOMES = {
    "Redes Neurais Artificiais (MLP)": "Rede Neural (MLP)",
    "Máquinas de Vetores de Suporte (SVM)": "SVM",
    "Regressão Linear Múltipla": "Regressão Linear",
}

# Variação anual do índice IPCA+INCC (docs/09, seção 9.2)
INDICE = {2016: 6.20, 2017: 3.60, 2018: 3.79, 2019: 4.23, 2020: 6.65,
          2021: 11.95, 2022: 7.53, 2023: 4.06, 2024: 5.68, 2025: 5.10}


def rotulo(m):
    if m["modelo"] == "Ensemble":
        return "Ensemble"
    # Linhas de treino em mil, curto para caber no gráfico: 3000 -> "3 mil", 11429 -> "11,4 mil"
    mil = f"{m.get('linhas_treino', 0) / 1000:.1f}".replace(".0", "").replace(".", ",") + " mil"
    if m.get("api"):
        return f"{m['modelo'].replace(' (API)', '')} pela API ({mil})"
    if m["modelo"].startswith("TabPFN"):
        return f"{m['modelo'].replace('v3.5', '3.5')} local ({mil})"
    alvo = "log" if m["alvo"].startswith("log") else "preço"
    return f"{NOMES.get(m['modelo'], m['modelo'])} ({alvo})"


def ranking_mdape(dados):
    corte = 30
    fig, eixos = plt.subplots(1, 2, figsize=(8, 7.2))
    for ax, seg in zip(eixos, ["Apartamento", "Casa"]):
        modelos = sorted(dados["segmentos"][seg]["modelos"], key=lambda m: m["posicao_cv"])
        nomes = [rotulo(m) for m in modelos]
        mdape = [m["cv"]["mdape"] for m in modelos]
        cores = [AZUL if m.get("escolhido") else (CINZA if m.get("no_site", True) else CLARO) for m in modelos]
        y = range(len(modelos))
        ax.barh(y, [min(v, corte) for v in mdape], color=cores, height=0.7)
        for i, v in enumerate(mdape):
            texto = f"{v:.1f}%".replace(".", ",")
            ax.text(min(v, corte) + 0.4, i, texto, va="center", fontsize=8.5, color=TINTA)
        ax.set_yticks(list(y), nomes, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlim(0, corte + 9)
        ax.set_xticks([0, 10, 20, 30])
        ax.xaxis.set_major_formatter(VIRGULA)
        ax.set_title(seg, fontsize=10)
        ax.set_xlabel("MdAPE na validação cruzada (%)", fontsize=9)
        ax.grid(axis="x", color="#e5e5e5", linewidth=0.6)
        ax.set_axisbelow(True)
    # A legenda das cores fica no LaTeX, abaixo da figura, em letra legível.
    fig.tight_layout()
    fig.savefig(SAIDA / "v3_ranking_mdape.png", dpi=200, bbox_inches="tight")
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


# Nomes dos bairros com acento (a base guarda sem acento, em minúsculas)
BAIRROS = {"jardim italia": "Jardim Itália", "lider": "Líder", "paraiso": "Paraíso",
           "presidente medici": "Presidente Médici", "santo antonio": "Santo Antônio",
           "sao cristovao": "São Cristóvão", "sao lucas": "São Lucas", "seminario": "Seminário",
           "universitario": "Universitário"}


def bairros():
    """Preço mediano por m² nos 15 bairros mais caros com 20 ou mais imóveis (mesma regra de eda.py)."""
    import sys

    sys.path.insert(0, str(RAIZ / "scripts_predict"))
    import eda

    norm, _, _ = eda.carregar()
    ca = eda.casas_aptos(norm)
    fig, eixos = plt.subplots(1, 2, figsize=(8, 4.6))
    for ax, tipo in zip(eixos, ["Apartamento", "Casa"]):
        g = ca[(ca["tipo_imovel"] == tipo) & (ca["bairro"].fillna("") != "")]
        stats = g.groupby("bairro")["preco_m2"].agg(["median", "size"])
        stats = stats[stats["size"] >= 20].sort_values("median").tail(15)
        nomes = [BAIRROS.get(b, b.title().replace(" Dos ", " dos ")) for b in stats.index]
        ax.barh(nomes, stats["median"] / 1000, color=AZUL, height=0.65)
        for y, (v, n) in enumerate(zip(stats["median"], stats["size"])):
            texto = f"{v:,.0f}".replace(",", ".")
            ax.text(v / 1000 + 0.15, y, f"R$ {texto} (n={n})", va="center", fontsize=7.5, color=TINTA)
        ax.set_xlim(0, stats["median"].max() / 1000 * 1.5)
        ax.tick_params(axis="y", labelsize=8.5)
        ax.xaxis.set_major_formatter(VIRGULA)
        ax.set_title(tipo, fontsize=10)
        ax.set_xlabel("Preço mediano por m² (R$ mil)")
        ax.grid(axis="x", color="#eeeeee", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(SAIDA / "eda_08_preco_m2_bairros.png", dpi=200)
    plt.close(fig)


def indice():
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    anos = list(INDICE)
    valores = list(INDICE.values())
    ax.bar(anos, valores, color=AZUL, width=0.65)
    for a, v in zip(anos, valores):
        ax.text(a, v + 0.25, f"{v:.2f}%".replace(".", ","), ha="center", fontsize=8, color=TINTA,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8})
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


def main():
    dados = json.loads((RAIZ / "dados" / "resultados_modelos_v3.json").read_text(encoding="utf-8"))
    SAIDA.mkdir(exist_ok=True)
    ranking_mdape(dados)
    previsto_real(dados)
    indice()
    bairros()


if __name__ == "__main__":
    main()
