"""Figuras 11 a 14 dos capítulos 4 e 5 da documentação, e erro por faixa de preço.

Lê os relatórios em docs/reports e os modelos em modelos/. A figura 12 e a tabela de
erro por faixa refazem a avaliação do modelo da interface (mesmo split e mesmos modelos
escolhidos), o que leva cerca de 1 minuto.

    DISABLE_TABPFN=1 python scripts_predict/figuras_resultados.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import imoveis_ml_hibrido as hib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from imoveis_projecao import annual_pct, load_projection_artifact  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

OUT = Path("docs/figures")
PRICE_MODEL = Path("modelos/preco_imovel_modelo_hibrido_rapido.pkl")
AZUL, LARANJA, AQUA, CINZA, TEXTO = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#9a9993",
    "#52514e",
)
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


def reais(s):
    return float(s.replace("R$", "").replace(".", "").replace(",", ".").strip())


def linhas_tabela(texto):
    return [
        [c.strip() for c in linha.strip("|").split("|")]
        for linha in texto.splitlines()
        if linha.startswith("| ")
        and "---" not in linha
        and not linha.startswith("| Modelo")
    ]


def fig_ranking_preco():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"wspace": 0.75})
    txt = Path("docs/reports/modelo_hibrido.md").read_text(encoding="utf-8")
    for ax, seg in zip(axes, ("Apartamento", "Casa")):
        parte = txt.split(f"## Segmento: {seg}", 1)[1].split("### Top 10", 1)[1]
        rows = linhas_tabela(parte)[:8]
        nomes = [f"{r[0]} ({r[1]})" for r in rows][::-1]
        maes = [reais(r[2]) / 1000 for r in rows][::-1]
        ax.barh(
            nomes,
            maes,
            color=[AZUL if "TabPFN" in n else CINZA for n in nomes],
            height=0.6,
        )
        for y, v in enumerate(maes):
            ax.text(v, y, f"  R$ {br(v)} mil", va="center", fontsize=8.5, color=TEXTO)
        ax.set_title(f"{seg}: erro médio (MAE) na validação cruzada")
        ax.set_xlim(min(maes) * 0.85, max(maes) * 1.12)
        ax.set_xlabel("MAE (R$ mil) — menor é melhor")
        ax.grid(axis="x", color="#e4e3df")
        ax.set_axisbelow(True)
    fig.suptitle(
        "Figura 11 — Modelos de preço: 8 melhores por tipo (TabPFN em azul)",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.savefig(OUT / "11_preco_ranking_modelos.png")
    plt.close(fig)


def previsoes_teste():
    """Refaz a avaliação do modelo da interface: previsões no teste típico por segmento."""
    art = hib.load_artifact(PRICE_MODEL)
    df = hib.load_training_frame(Path("imoveis_normalizados.db"))
    out = {}
    for tipo, seg in art["segments"].items():
        split = hib.split_and_prepare_segment(
            df, tipo, seed=art["seed"], test_size=art["test_size"]
        )
        train, test = split["train"], split["test_typical"]
        escolhidos = [
            {"model": s["model_name"], "target": s["target"], "mae": s["mae_validacao"]}
            for s in seg["selected"]
        ]
        ens = hib.fit_weighted_ensemble(
            train[hib.FEATURE_COLUMNS],
            train["preco"].to_numpy(float),
            escolhidos,
            seed=art["seed"],
        )
        y = test["preco"].to_numpy(float)
        out[tipo] = (y, hib.predict_weighted_ensemble(ens, test[hib.FEATURE_COLUMNS]))
    return out


def resumo_erros(pred):
    for tipo, (y, p) in pred.items():
        ape = np.abs(p - y) / y
        print(
            f"{tipo}: n={len(y)} MAE={np.mean(np.abs(p - y)):.0f} erro%mediano={100 * np.median(ape):.1f} "
            f"<=10%: {100 * np.mean(ape <= 0.1):.0f}% <=20%: {100 * np.mean(ape <= 0.2):.0f}%"
        )
        faixas = pd.qcut(y, 4)
        tab = pd.DataFrame({"ae": np.abs(p - y), "ape": ape, "faixa": faixas})
        print(
            tab.groupby("faixa", observed=True)
            .agg(n=("ae", "size"), mae=("ae", "mean"), mdape=("ape", "median"))
            .round(3)
        )


def fig_previsto_real(pred):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, seg in zip(axes, ("Apartamento", "Casa")):
        y, p = pred[seg][0] / 1e6, pred[seg][1] / 1e6
        m = (
            y > 0.05
        )  # anúncios com preço em escala errada (ex.: "R$ 1,35") ficam fora do gráfico
        ax.scatter(
            y[m], p[m], s=14, color=AZUL, alpha=0.55, edgecolor="white", linewidth=0.4
        )
        lim = [0, max(y[m].max(), p[m].max()) * 1.05]
        ax.plot(lim, lim, color=TEXTO, lw=1, ls="--", label="previsão perfeita")
        ax.fill_between(
            lim,
            [v * 0.8 for v in lim],
            [v * 1.2 for v in lim],
            color=AZUL,
            alpha=0.08,
            label="faixa de ±20%",
        )
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("Preço anunciado (R$ milhões)")
        ax.set_ylabel("Preço previsto (R$ milhões)")
        dentro = np.mean(np.abs(p - y) / y <= 0.2)
        ax.set_title(f"{seg} — {br(100 * dentro)}% dos imóveis dentro de ±20%")
        ax.legend(frameon=False, loc="upper left", fontsize=8.5)
        fmt = FuncFormatter(lambda v, _, d=(2 if lim[1] < 2 else 1): br(v, d))
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
    fig.suptitle(
        "Figura 12 — Preço previsto x preço anunciado no conjunto de teste (modelo da interface)",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.savefig(OUT / "12_preco_previsto_vs_real.png")
    plt.close(fig)


def fig_valorizacao():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"wspace": 0.7})
    for ax, h in zip(axes, (1, 3)):
        txt = Path(f"docs/reports/modelo_valorizacao_h{h}.md").read_text(
            encoding="utf-8"
        )
        parte = txt.split("## Resultados no teste temporal", 1)[1].split(
            "## Chance", 1
        )[0]
        rows = sorted(linhas_tabela(parte), key=lambda r: float(r[2]), reverse=True)
        nomes = [r[0].replace("Baseline: ", "Ref.: ") for r in rows]
        vals = [float(r[2]) for r in rows]
        base = float([r for r in rows if r[0] == "Baseline: preço não muda"][0][2])
        cores = [
            CINZA if n.startswith("Ref.") else (AQUA if v < base else AZUL)
            for n, v in zip(nomes, vals)
        ]
        ax.barh(nomes, vals, color=cores, height=0.6)
        ax.axvline(base, color=LARANJA, lw=1.5, ls="--")
        ax.text(
            base,
            -1.0,
            ' referência "preço não muda"',
            color=LARANJA,
            fontsize=8,
            va="center",
        )
        for y, v in enumerate(vals):
            ax.text(v, y, f"  {br(v, 2)}", va="center", fontsize=8.5, color=TEXTO)
        ax.set_xlim(min(vals) * 0.9, max(vals) * 1.09)
        ax.set_ylim(-1.5, len(nomes) - 0.5)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: br(v, 1)))
        ax.set_title(
            f"Variação do preço em {h} {'mês' if h == 1 else 'meses'}: RMSE no teste"
        )
        ax.set_xlabel("RMSE (pontos percentuais) — menor é melhor")
    fig.suptitle(
        "Figura 13 — Prever a variação de preço: só o horizonte de 3 meses supera a referência (verde)",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.savefig(OUT / "13_valorizacao_rmse_horizontes.png")
    plt.close(fig)


def fig_taxas_bairros():
    art = load_projection_artifact("modelos/projecao_modelo.pkl")
    segs = [
        (k.split("|")[0], v)
        for k, v in art["segmentos"].items()
        if k.endswith("|Apartamento") and v["imoveis"] >= 15 and k.split("|")[0]
    ]
    segs.sort(key=lambda kv: annual_pct(kv[1]["mensal"]))
    fig, ax = plt.subplots(figsize=(9, 7.5))
    for i, (_, v) in enumerate(segs):
        lo, c, hi = (
            annual_pct(v["mensal_baixa"]),
            annual_pct(v["mensal"]),
            annual_pct(v["mensal_alta"]),
        )
        ax.plot([lo, hi], [i, i], color=AZUL, alpha=0.35, lw=4, solid_capstyle="round")
        ax.plot(c, i, "o", color=AZUL, ms=6)
    ipca = art["ipca"]["media_anual_pct"]
    geral = annual_pct(art["tipos"]["Apartamento"]["mensal"])
    ax.axvline(ipca, color=LARANJA, ls="--", lw=1.5)
    ax.text(
        ipca, len(segs) - 0.5, f" IPCA {br(ipca, 2)}%/ano", color=LARANJA, fontsize=9
    )
    ax.axvline(geral, color=TEXTO, ls=":", lw=1.2)
    ax.text(
        geral, -1.3, f" média apartamentos {br(geral, 2)}%", color=TEXTO, fontsize=8.5
    )
    ax.axvline(0, color=CINZA, lw=0.8)
    ax.set_yticks(range(len(segs)), [b.title() for b, _ in segs])
    ax.set_ylim(-1.8, len(segs))
    ax.set_xlabel(
        "Valorização anual do preço anunciado (%) — ponto: central; barra: faixa de 80%"
    )
    ax.set_title(
        "Figura 14 — Apartamentos: valorização anual por bairro\n(bairros com 15+ imóveis) x IPCA",
        loc="left",
    )
    fig.savefig(OUT / "14_valorizacao_anual_bairros_ipca.png")
    plt.close(fig)


def main():
    fig_ranking_preco()
    pred = previsoes_teste()
    resumo_erros(pred)
    fig_previsto_real(pred)
    fig_valorizacao()
    fig_taxas_bairros()
    print("Figuras 11 a 14 salvas em", OUT)


if __name__ == "__main__":
    main()
