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
        return "Ensemble (site)"
    alvo = "log" if m["alvo"].startswith("log") else "preço"
    return f"{NOMES.get(m['modelo'], m['modelo'])} ({alvo})"


def ranking_r2(dados):
    fig, eixos = plt.subplots(1, 2, figsize=(9, 5.6))
    for ax, seg in zip(eixos, ["Apartamento", "Casa"]):
        modelos = sorted(dados["segmentos"][seg]["modelos"], key=lambda m: m["posicao_cv"])
        nomes = [rotulo(m) for m in modelos]
        r2 = [m["teste_tipico"]["r2"] for m in modelos]
        cores = [AZUL if m["posicao_cv"] <= 3 else CINZA for m in modelos]
        y = range(len(modelos))
        ax.barh(y, [max(v, -0.5) for v in r2], color=cores, height=0.7)
        for i, v in enumerate(r2):
            texto = f"{v:.3f}".replace(".", ",")
            x = max(v, -0.5)
            ax.text(x + 0.02 if x >= 0 else 0.02, i, texto, va="center", fontsize=7.5, color=TINTA)
        ax.set_yticks(list(y), nomes, fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlim(-0.5, 1.0)
        ax.axvline(0, color="#888888", linewidth=0.8)
        ax.xaxis.set_major_formatter(VIRGULA)
        ax.set_title(seg, fontsize=10)
        ax.set_xlabel("R² no teste (preços reais)")
        ax.grid(axis="x", color="#e5e5e5", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.text(0.01, 0.005, "Ordem: posição na validação cruzada. Barras azuis: o ensemble do site e os três modelos que o compõem. "
             "Valores abaixo de −0,5 foram cortados no gráfico.", fontsize=7.5, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(SAIDA / "v3_ranking_r2.png", dpi=200)
    plt.close(fig)


def previsto_real(dados):
    fig, eixos = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, seg in zip(eixos, ["Apartamento", "Casa"]):
        segmento = dados["segmentos"][seg]
        ens = next(m for m in segmento["modelos"] if m["modelo"] == "Ensemble")
        real = [v / 1e6 for v in segmento["real"]]
        prev = [v / 1e6 for v in ens["previsto"]]
        pares = [(r, p) for r, p in zip(real, prev) if r > 0.05]
        ax.scatter([r for r, _ in pares], [p for _, p in pares], s=10, color=AZUL, alpha=0.55,
                   edgecolors="none")
        lim = max(max(r for r, _ in pares), max(p for _, p in pares)) * 1.05
        ax.plot([0, lim], [0, lim], color="#555555", linewidth=1, linestyle="--")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_title(seg, fontsize=10)
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


def main():
    dados = json.loads((RAIZ / "dados" / "resultados_modelos_v3.json").read_text(encoding="utf-8"))
    SAIDA.mkdir(exist_ok=True)
    ranking_r2(dados)
    previsto_real(dados)
    indice()


if __name__ == "__main__":
    main()
