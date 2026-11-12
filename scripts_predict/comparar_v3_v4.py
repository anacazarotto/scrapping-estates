"""Comparação v3 x v4 (para o docs/10): taxas, métricas, valorização projetada e figura.

Lê o que os scripts de cada versão já gravaram (não treina nada):
- dados/dataset_simulado_v3.csv e dados/dataset_simulado_v4.csv (histórico simulado);
- dados/resultados_modelos_v3.json e dados/resultados_modelos_v4.json (ranking e teste);
- modelos/projecao_modelo.pkl (taxa medida nos anúncios, usada na projeção v3).

Projeção v4: a v3 combina a taxa dos anúncios (peso w = 29%) com o índice IPCA+INCC
(peso 1 - w = 71%). Na v4, o índice de cada imóvel é o índice + ajuste fuzzy, então

    taxa v4 do imóvel = taxa v3 do bairro + (1 - w) x ajuste fuzzy do imóvel

e a faixa pessimista-otimista se desloca do mesmo tanto. Na média de cada tipo, o ajuste é 0.

    python scripts_predict/comparar_v3_v4.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from imoveis_projecao import ajustar_referencia, annual_pct, lookup_rate  # noqa: E402
from model_io import load_model  # noqa: E402

SAIDA = Path("dados/comparacao_v3_v4.json")
FIGURA = Path("docs/figures/15_v4_passado_exemplos.png")
CORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # paleta categórica padrão, ordem fixa
TINTA, TINTA2, GRADE, FUNDO = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def ler(caminho):
    return pd.read_csv(caminho, keep_default_na=False, na_values=[""])


def metricas(resultados):
    """Ranking completo e métricas do escolhido, por tipo."""
    saida = {}
    for tipo, s in resultados["segmentos"].items():
        linhas = []
        for m in s["modelos"]:
            t = m["teste_tipico"]
            linhas.append({"posicao": m["posicao_cv"], "modelo": m["modelo"], "alvo": m["alvo"],
                           "no_site": m["no_site"], "escolhido": m["escolhido"],
                           "mdape_cv": m["cv"]["mdape"], "mdape_teste": t["erro_mediano_pct"],
                           "mae": t["mae"], "rmse": t["rmse"], "r2": t["r2"]})
        saida[tipo] = linhas
    return saida


def projecao(dataset_v4, proj):
    """Taxa projetada (central, pessimista, otimista) na v3 e distribuição por imóvel na v4."""
    reais = dataset_v4[dataset_v4["simulado"] == 0]
    saida = {}
    for tipo, g in reais.groupby("tipo_imovel"):
        tipo_rate, w = ajustar_referencia(proj["tipos"][tipo], proj, base="ipca_incc")
        v3 = {k: annual_pct(tipo_rate[c]) for k, c in
              (("central", "mensal"), ("pessimista", "mensal_baixa"), ("otimista", "mensal_alta"))}
        por_imovel = {"central": [], "pessimista": [], "otimista": []}
        for bairro, aj in zip(g["bairro"].fillna(""), g["ajuste_pp"]):
            rate, _ = ajustar_referencia(lookup_rate(proj, bairro, tipo)[0], proj, base="ipca_incc")
            for k, c in (("central", "mensal"), ("pessimista", "mensal_baixa"), ("otimista", "mensal_alta")):
                por_imovel[k].append(annual_pct(rate[c]) + (1 - w) * aj)
        bairro_v3 = [annual_pct(ajustar_referencia(lookup_rate(proj, b, tipo)[0], proj, base="ipca_incc")[0]["mensal"])
                     for b in g["bairro"].fillna("")]
        saida[tipo] = {
            "peso_anuncios": w,
            "v3_tipo": v3,
            "v3_por_imovel_central": quantis(bairro_v3),
            "v4_por_imovel": {k: quantis(v) for k, v in por_imovel.items()},
        }
    return saida


def quantis(v):
    v = np.asarray(v, dtype=float)
    return {"min": float(v.min()), "p10": float(np.percentile(v, 10)), "mediana": float(np.median(v)),
            "media": float(v.mean()), "p90": float(np.percentile(v, 90)), "max": float(v.max())}


def exemplos(v4, tipo="Apartamento"):
    """4 imóveis do tipo: o de menor ajuste, dois do meio e o de maior ajuste."""
    reais = v4[(v4["simulado"] == 0) & (v4["tipo_imovel"] == tipo)].sort_values("ajuste_pp")
    pos = [0, len(reais) // 3, 2 * len(reais) // 3, len(reais) - 1]
    return reais.iloc[pos]


def figura(v3, v4):
    escolha = exemplos(v4)
    fig, eixos = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, facecolor=FUNDO)
    for eixo, (titulo, base) in zip(eixos, (("v3: todos com o mesmo passado", v3),
                                            ("v4: cada imóvel com o seu passado", v4))):
        eixo.set_facecolor(FUNDO)
        inicios = []
        for i, (cor, im) in enumerate(zip(CORES, escolha.itertuples())):
            linhas = base[base["imovel"] == im.imovel].sort_values("ano")
            hoje = float(linhas.loc[linhas["simulado"] == 0, "preco"].iloc[0])
            x, y = linhas["ano"].to_numpy(), linhas["preco"].to_numpy() / hoje * 100
            eixo.plot(x, y, color=cor, lw=2, marker="o", ms=4, zorder=3 + i,
                      label=f"{im.bairro.title() if isinstance(im.bairro, str) and im.bairro else 'sem bairro'}, "
                            f"{im.area_privada or im.area_total:.0f} m² (ajuste {im.ajuste_pp:+.2f} p.p.)")
            if base is v4:
                inicios.append((y[0], x[0]))
        # Valor de 2016 à esquerda de cada linha, afastando rótulos muito próximos.
        ultimo = -np.inf
        for valor, x0 in sorted(inicios):
            pos = max(valor, ultimo + 2.2)
            ultimo = pos
            eixo.annotate(f"{valor:.0f}", (x0, pos), xytext=(-6, 0), textcoords="offset points",
                          ha="right", va="center", fontsize=8, color=TINTA2)
        if base is v3:
            eixo.annotate(f"{y[0]:.0f} (as 4 linhas coincidem)", (x[0], y[0]), xytext=(4, -12),
                          textcoords="offset points", fontsize=8, color=TINTA2)
        eixo.set_title(titulo, fontsize=11, color=TINTA, loc="left")
        eixo.grid(axis="y", color=GRADE, lw=0.8)
        eixo.set_xlim(2015.3, 2027)
        for lado in ("top", "right"):
            eixo.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            eixo.spines[lado].set_color(GRADE)
        eixo.tick_params(colors=TINTA2, labelsize=9)
        eixo.set_xlabel("ano", color=TINTA2, fontsize=9)
    eixos[0].set_ylabel("valor do imóvel (ago/2026 = 100)", color=TINTA2, fontsize=9)
    alcas, rotulos = eixos[1].get_legend_handles_labels()
    fig.legend(alcas, rotulos, loc="lower center", ncol=2, fontsize=8, frameon=False, labelcolor=TINTA)
    fig.suptitle("Passado simulado de 4 apartamentos de exemplo", fontsize=12, color=TINTA, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    FIGURA.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURA, dpi=150, facecolor=FUNDO)
    return escolha[["imovel", "bairro", "area_privada", "area_total", "ajuste_pp"]].to_dict("records")


def passado_aprendido(v3):
    """Taxa anual 2016->2026 que o modelo FINAL de cada versão atribui a cada imóvel real.

    Usa os modelos que rodam no site (ensembles locais; na casa, a reserva). Mostra se o
    modelo aprendeu passados diferentes para imóveis diferentes. Não é medida de acerto.
    """
    from imoveis_ml_hibrido import apply_numeric_clip_bounds, apply_rare_bairros
    from modelo_v3 import MEIO_DO_ANO, load_v3, prever

    reais = v3[v3["simulado"] == 0].copy()
    reais["bairro"] = reais["bairro"].fillna("")
    saida = {}
    for versao, caminho, pasta in (("v3", "modelos/preco_imovel_modelo_v3.pkl", "v3"),
                                    ("v4", "modelos/preco_imovel_modelo_v4.pkl", "v4")):
        art = load_model(caminho)
        for tipo, seg in art["segments"].items():
            for e in seg["ensemble"]:
                e["estimator"] = load_model(Path("modelos") / pasta / e["arquivo"])
            x = reais[reais["tipo_imovel"] == tipo]
            x = apply_rare_bairros(x, set(seg["rare_bairros"]))
            x = apply_numeric_clip_bounds(x, {k: tuple(v) for k, v in seg["clip_bounds"].items()})
            hoje = prever(seg["ensemble"], x)
            antes = prever(seg["ensemble"], x.assign(ano=2016 + MEIO_DO_ANO))
            anos = x["ano"].to_numpy() - (2016 + MEIO_DO_ANO)
            # Anúncios com preço absurdo (R$ 1 ou R$ 2) podem ter previsão negativa em 2016.
            ok = (hoje > 0) & (antes > 0)
            taxa = ((hoje[ok] / antes[ok]) ** (1 / anos[ok]) - 1) * 100
            saida.setdefault(tipo, {})[versao] = {**quantis(taxa), "desvio": float(np.std(taxa)),
                                                  "fora": int((~ok).sum()), "modelo": seg["reserva"]["model"]}
        del art
    return saida


def main():
    v3, v4 = ler("dados/dataset_simulado_v3.csv"), ler("dados/dataset_simulado_v4.csv")
    comparacao = {
        "passado_aprendido_2016": passado_aprendido(v3),
        "taxas_v4": json.loads(Path("dados/taxas_fuzzy_v4.json").read_text(encoding="utf-8")),
        "exemplos_figura": figura(v3, v4),
        "projecao": projecao(v4, load_model("modelos/projecao_modelo.pkl")),
    }
    for versao in ("v3", "v4"):
        caminho = Path(f"dados/resultados_modelos_{versao}.json")
        if caminho.exists():
            comparacao[f"metricas_{versao}"] = metricas(json.loads(caminho.read_text(encoding="utf-8")))
    SAIDA.write_text(json.dumps(comparacao, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"Comparação salva em {SAIDA}; figura em {FIGURA}")


if __name__ == "__main__":
    main()
