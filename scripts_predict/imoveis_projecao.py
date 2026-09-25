"""PROJEÇÃO do valor de um imóvel ao longo dos anos.

Junta os dois modelos do projeto:

1. Valor hoje: modelo de preço híbrido (`imoveis_ml_hibrido.py train-hibrido`), a
   partir das características do imóvel (bairro, tipo, área, quartos...).
2. Taxa de valorização: medida nos próprios dados coletados, pelos MESMOS imóveis
   anunciados em meses diferentes ("repeat listings", tabela `historico_precos`).

Taxa por segmento (bairro + tipo)
---------------------------------
- Taxa mensal = soma das variações log / soma dos meses decorridos, somando todos os
  pares de coletas consecutivas de cada imóvel (aceita meses faltando no meio).
- Segmentos com poucos imóveis são puxados para a taxa do tipo (Casa/Apartamento)
  por credibilidade: peso = n / (n + K). Bairro sem dados usa a taxa do tipo.
- Incerteza: bootstrap por imóvel (reamostra imóveis, não observações), dando um
  intervalo de 80% (percentis 10 e 90) para a taxa.

Projeção: valor(ano) = valor_hoje * exp(12 * taxa_mensal * ano), com cenários
pessimista / central / otimista vindos do intervalo da taxa, e um cenário de
referência com o IPCA (inflação oficial; não há FipeZap para Chapecó).

Limitações que devem constar no TCC: a taxa vem de poucos meses de coleta e é
extrapolada para anos; preço ANUNCIADO não é preço de venda; imóveis vendidos saem
do painel (viés de sobrevivência). A projeção é um cenário, não uma garantia.
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from imoveis_ml import canonical_tipo, normalize_text, unify_bairro
from imoveis_ml_hibrido import format_currency
from imoveis_ml_hibrido import load_artifact as load_price_artifact
from imoveis_ml_hibrido import predict_price
from imoveis_valorizacao import ALLOWED_TYPES, MAX_ABS_MONTHLY_LOG_CHANGE, load_panel
from ipca import load_ipca
from model_io import load_model, save_model

DEFAULT_UNIFIED_DB = Path("imoveis_unificado.db")
DEFAULT_PRICE_MODEL = Path("modelos/preco_imovel_modelo_hibrido_rapido.pkl")
DEFAULT_MODEL_PATH = Path("modelos/projecao_modelo.pkl")
DEFAULT_REPORT_PATH = Path("docs/reports/taxas_valorizacao_anual.md")

ARTIFACT_KIND = "projecao_anual_v1"
CREDIBILITY_K = 30  # imóveis: com 30 imóveis repetidos o bairro pesa 50%
MIN_SEGMENT_IMOVEIS = 5
BOOTSTRAP_SAMPLES = 1000
INTERVAL = (10, 90)  # percentis do cenário pessimista / otimista
MAX_ANOS = 30
REFERENCIAS = ("ipca-media", "ipca-12m", "nenhuma")


# --------------------------------------------------------------------- taxas
def repeat_pairs(panel):
    """Pares de coletas consecutivas do mesmo imóvel: variação log e meses decorridos."""
    work = panel.sort_values(["codigo", "mes_n"]).copy()
    work["gap"] = work["mes_n"] - work.groupby("codigo")["mes_n"].shift(1)
    work["dlog"] = np.log(work["preco"] / work.groupby("codigo")["preco"].shift(1))
    work = work.dropna(subset=["gap", "dlog"])
    work = work[work["gap"] > 0]
    # Saltos > 50% por mês são erro de cadastro, não valorização.
    work = work[work["dlog"].abs() <= MAX_ABS_MONTHLY_LOG_CHANGE * work["gap"]]
    return work[["codigo", "bairro", "tipo_imovel", "gap", "dlog"]]


def per_imovel_sums(pairs):
    return pairs.groupby("codigo").agg(
        bairro=("bairro", "first"),
        tipo_imovel=("tipo_imovel", "first"),
        dlog=("dlog", "sum"),
        gap=("gap", "sum"),
    )


def bootstrap_rate(dlog, gap, rng, samples=BOOTSTRAP_SAMPLES):
    """Taxa mensal (soma dlog / soma meses) e sua distribuição por bootstrap de imóveis."""
    dlog = np.asarray(dlog, dtype=float)
    gap = np.asarray(gap, dtype=float)
    point = float(dlog.sum() / gap.sum())
    idx = rng.integers(0, len(dlog), size=(samples, len(dlog)))
    boots = dlog[idx].sum(axis=1) / gap[idx].sum(axis=1)
    return point, boots


def estimate_rates(panel, seed=42, k=CREDIBILITY_K):
    """Taxas mensais por tipo e por (bairro, tipo), com intervalo por bootstrap."""
    rng = np.random.default_rng(seed)
    sums = per_imovel_sums(repeat_pairs(panel))

    tipos = {}
    for tipo, grp in sums.groupby("tipo_imovel"):
        point, boots = bootstrap_rate(grp["dlog"], grp["gap"], rng)
        tipos[tipo] = {
            "point": point,
            "boots": boots,
            "imoveis": int(len(grp)),
            "meses": float(grp["gap"].sum()),
        }

    segmentos = {}
    for (bairro, tipo), grp in sums.groupby(["bairro", "tipo_imovel"]):
        if tipo not in tipos or len(grp) < MIN_SEGMENT_IMOVEIS:
            continue
        point, boots = bootstrap_rate(grp["dlog"], grp["gap"], rng)
        w = len(grp) / (len(grp) + k)
        base = tipos[tipo]
        segmentos[(bairro, tipo)] = {
            "point": w * point + (1 - w) * base["point"],
            "boots": w * boots + (1 - w) * base["boots"],
            "bruta": point,
            "peso_bairro": w,
            "imoveis": int(len(grp)),
        }
    return tipos, segmentos


def summarize(entry):
    lo, hi = np.percentile(entry["boots"], INTERVAL)
    return {
        "mensal": float(entry["point"]),
        "mensal_baixa": float(lo),
        "mensal_alta": float(hi),
    }


def annual_pct(monthly_log):
    return 100.0 * float(np.expm1(12.0 * monthly_log))


# ----------------------------------------------------------------- projeção
def reference_rate(artifact, escolha):
    """Taxa anual de referência (%) e rótulo: IPCA média anual, IPCA 12 meses ou nenhuma."""
    ipca = artifact.get("ipca")
    if escolha == "nenhuma" or not ipca:
        return None, None
    if escolha == "ipca-12m":
        return ipca["acumulado_12m_pct"], "IPCA 12 meses"
    return ipca["media_anual_pct"], f"IPCA média {ipca['anos_media']} anos"


def lookup_rate(artifact, bairro, tipo):
    key = f"{normalize_text(bairro)}|{tipo}"
    if key in artifact["segmentos"]:
        return artifact["segmentos"][key], "bairro"
    return artifact["tipos"][tipo], "tipo"


def project(
    artifact,
    price_artifact,
    *,
    bairro,
    tipo_imovel,
    area_total=0,
    area_privada=0,
    quartos=0,
    banheiros=0,
    vagas=0,
    anos=10,
    preco_atual=None,
    taxa_referencia=None,
):
    """Valor estimado hoje e ano a ano. Função pensada para ser usada pela interface.

    `preco_atual` (opcional) substitui o valor estimado pelo modelo de preço.
    `taxa_referencia` (opcional, % ao ano) adiciona um cenário de referência (IPCA).
    """
    bairro = unify_bairro(bairro)
    tipo = canonical_tipo(tipo_imovel)
    if tipo not in ALLOWED_TYPES:
        raise ValueError("Tipo de imóvel deve ser Casa ou Apartamento.")
    anos = int(anos)
    if not 1 <= anos <= MAX_ANOS:
        raise ValueError(f"Horizonte deve estar entre 1 e {MAX_ANOS} anos.")

    valor_modelo = predict_price(
        price_artifact,
        area_total=area_total,
        area_privada=area_privada,
        bairro=bairro,
        tipo_imovel=tipo,
        quartos=quartos,
        banheiros=banheiros,
        vagas=vagas,
    )
    valor_hoje = float(preco_atual) if preco_atual else float(valor_modelo)
    rate, origem = lookup_rate(artifact, bairro, tipo)

    rows = []
    for ano in range(0, anos + 1):
        row = {
            "ano": ano,
            "pessimista": valor_hoje * float(np.exp(12 * rate["mensal_baixa"] * ano)),
            "central": valor_hoje * float(np.exp(12 * rate["mensal"] * ano)),
            "otimista": valor_hoje * float(np.exp(12 * rate["mensal_alta"] * ano)),
        }
        if taxa_referencia is not None:
            row["referencia"] = valor_hoje * (1 + float(taxa_referencia) / 100.0) ** ano
        rows.append(row)

    segment_metrics = price_artifact["segments"][tipo]["metrics"]
    return {
        "tipo_imovel": tipo,
        "bairro": normalize_text(bairro),
        "valor_modelo_hoje": float(valor_modelo),
        "valor_hoje": valor_hoje,
        "usou_preco_informado": bool(preco_atual),
        "erro_tipico_modelo_preco": float(segment_metrics["mae"]),
        "taxa_origem": origem,  # "bairro" ou "tipo" (bairro sem dados suficientes)
        "taxa_anual_central_pct": annual_pct(rate["mensal"]),
        "taxa_anual_pessimista_pct": annual_pct(rate["mensal_baixa"]),
        "taxa_anual_otimista_pct": annual_pct(rate["mensal_alta"]),
        "taxa_referencia_pct": taxa_referencia,
        "dados_ate": artifact["ultimo_mes"],
        "projecao": pd.DataFrame(rows),
    }


# ------------------------------------------------------------------ comandos
def build_artifact(unified_db, seed=42):
    panel = load_panel(unified_db)
    tipos, segmentos = estimate_rates(panel, seed=seed)
    months = sorted(panel["mes"].unique())
    return {
        "kind": ARTIFACT_KIND,
        "tipos": {
            t: {**summarize(v), "imoveis": v["imoveis"]} for t, v in tipos.items()
        },
        "segmentos": {
            f"{b}|{t}": {
                **summarize(v),
                "imoveis": v["imoveis"],
                "peso_bairro": v["peso_bairro"],
                "mensal_bruta": v["bruta"],
            }
            for (b, t), v in segmentos.items()
        },
        "meses": [str(m) for m in months],
        "ultimo_mes": str(months[-1]),
        "credibilidade_k": CREDIBILITY_K,
        "intervalo_percentis": list(INTERVAL),
        "ipca": load_ipca(),
        "seed": seed,
        "trained_at": datetime.now().isoformat(),
    }


def build_report(artifact):
    p_lo, p_hi = artifact["intervalo_percentis"]
    lines = [
        "# Taxas de valorização anual (projeção de longo prazo)",
        "",
        f"- Meses de coleta: {', '.join(artifact['meses'])}",
        "- Método: variação do preço anunciado dos mesmos imóveis entre coletas "
        "(repeat listings), anualizada.",
        f"- Bairros puxados para a taxa do tipo por credibilidade: peso do bairro = n / (n + "
        f"{artifact['credibilidade_k']}).",
        f"- Intervalo: percentis {p_lo} e {p_hi} de bootstrap por imóvel "
        "(cenários pessimista e otimista).",
        "",
        "## Por tipo",
        "",
        "| Tipo | Imóveis | Pessimista (%/ano) | Central (%/ano) | Otimista (%/ano) |",
        "|---|---:|---:|---:|---:|",
    ]
    for tipo, r in artifact["tipos"].items():
        lines.append(
            f"| {tipo} | {r['imoveis']} | {annual_pct(r['mensal_baixa']):+.2f} | "
            f"{annual_pct(r['mensal']):+.2f} | {annual_pct(r['mensal_alta']):+.2f} |"
        )
    lines += [
        "",
        "## Por bairro",
        "",
        "| Bairro | Tipo | Imóveis | Peso do bairro | Bruta (%/ano) | Pessimista | Central | Otimista |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    segs = sorted(
        artifact["segmentos"].items(),
        key=lambda kv: (kv[0].split("|")[1], -kv[1]["mensal"]),
    )
    for key, r in segs:
        bairro, tipo = key.split("|")
        lines.append(
            f"| {bairro or '(sem bairro)'} | {tipo} | {r['imoveis']} | {r['peso_bairro']:.2f} | "
            f"{annual_pct(r['mensal_bruta']):+.2f} | {annual_pct(r['mensal_baixa']):+.2f} | "
            f"{annual_pct(r['mensal']):+.2f} | {annual_pct(r['mensal_alta']):+.2f} |"
        )
    lines += [
        "",
        "## Limitações",
        "",
        "- A taxa vem de poucos meses de coleta e é extrapolada para anos.",
        "- Preço anunciado não é preço de venda; imóveis vendidos saem do painel.",
        "- Anúncios raramente mudam de preço, o que tende a subestimar a valorização real.",
        "  Por isso a projeção mostra também o cenário de referência com o IPCA.",
    ]
    ipca = artifact.get("ipca")
    if ipca:
        lines[lines.index("## Por tipo") : lines.index("## Por tipo")] = [
            "## Referência: IPCA",
            "",
            f"- Fonte: {ipca['fonte']} ({ipca['inicio']} a {ipca['fim']}).",
            f"- Média anual dos últimos {ipca['anos_media']} anos: "
            f"{ipca['media_anual_pct']:+.2f}% (padrão da projeção).",
            f"- Acumulado dos últimos 12 meses: {ipca['acumulado_12m_pct']:+.2f}%.",
            "- Não há índice de preços de imóveis (FipeZap) para Chapecó; o IPCA indica",
            "  quanto o imóvel valeria se apenas acompanhasse a inflação.",
            "",
        ]
    return "\n".join(lines) + "\n"


def cmd_train(args):
    artifact = build_artifact(args.unified_db, seed=args.seed)
    save_model(artifact, args.model_path)
    report = Path(args.report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(build_report(artifact), encoding="utf-8")
    print(f"Taxas de projeção salvas em {args.model_path} | Relatório: {report}")
    for tipo, r in artifact["tipos"].items():
        print(
            f"{tipo:<12} central {annual_pct(r['mensal']):+.2f}%/ano  "
            f"(pessimista {annual_pct(r['mensal_baixa']):+.2f}%, otimista {annual_pct(r['mensal_alta']):+.2f}%)  "
            f"imóveis={r['imoveis']}"
        )
    print(f"Segmentos (bairro + tipo) com taxa própria: {len(artifact['segmentos'])}")
    ipca = artifact["ipca"]
    if ipca:
        print(
            f"IPCA: média {ipca['anos_media']} anos {ipca['media_anual_pct']:+.2f}%/ano | "
            f"12 meses {ipca['acumulado_12m_pct']:+.2f}% (até {ipca['fim']})"
        )
    else:
        print("[aviso] IPCA indisponível: projeção sem cenário de referência.")


def load_projection_artifact(model_path):
    art = load_model(model_path)
    if not isinstance(art, dict) or art.get("kind") != ARTIFACT_KIND:
        raise SystemExit("Arquivo de projeção inválido. Rode: train-projecao")
    return art


def cmd_predict(args):
    art = load_projection_artifact(args.model_path)
    price_art = load_price_artifact(Path(args.price_model_path))
    if args.taxa_referencia is not None:
        taxa_ref, rotulo_ref = args.taxa_referencia, "Referência"
    else:
        taxa_ref, rotulo_ref = reference_rate(art, args.referencia)
    try:
        res = project(
            art,
            price_art,
            bairro=args.bairro,
            tipo_imovel=args.tipo_imovel,
            area_total=args.area_total,
            area_privada=args.area_privada,
            quartos=args.quartos,
            banheiros=args.banheiros,
            vagas=args.vagas,
            anos=args.anos,
            preco_atual=args.preco_atual,
            taxa_referencia=taxa_ref,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(
        f"Valor estimado hoje ({res['dados_ate']}): {format_currency(res['valor_modelo_hoje'])} "
        f"(erro típico do modelo: ± {format_currency(res['erro_tipico_modelo_preco'])})"
    )
    if res["usou_preco_informado"]:
        print(
            f"Projeção a partir do preço informado: {format_currency(res['valor_hoje'])}"
        )
    origem = (
        "do bairro"
        if res["taxa_origem"] == "bairro"
        else "do tipo (bairro com poucos dados)"
    )
    print(
        f"Valorização {origem}: {res['taxa_anual_central_pct']:+.2f}%/ano "
        f"(pessimista {res['taxa_anual_pessimista_pct']:+.2f}%, otimista {res['taxa_anual_otimista_pct']:+.2f}%)"
    )
    header = f"{'Ano':>4} {'Pessimista':>18} {'Central':>18} {'Otimista':>18}"
    if taxa_ref is not None:
        header += f" {rotulo_ref + ' ' + format(taxa_ref, '+.2f') + '%':>26}"
    print(header)
    for r in res["projecao"].itertuples(index=False):
        line = (
            f"{r.ano:>4} {format_currency(r.pessimista):>18} {format_currency(r.central):>18} "
            f"{format_currency(r.otimista):>18}"
        )
        if taxa_ref is not None:
            line += f" {format_currency(r.referencia):>26}"
        print(line)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Projeção do valor de imóveis ao longo dos anos."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser(
        "train-projecao", help="Calcula e salva as taxas de valorização anual."
    )
    t.add_argument("--unified-db", default=str(DEFAULT_UNIFIED_DB))
    t.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    t.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    t.add_argument("--seed", type=int, default=42)
    t.set_defaults(func=cmd_train)

    p = sub.add_parser("projetar", help="Projeta o valor de um imóvel ano a ano.")
    p.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--price-model-path", default=str(DEFAULT_PRICE_MODEL))
    p.add_argument("--bairro", required=True)
    p.add_argument("--tipo-imovel", required=True)
    p.add_argument("--area-total", type=float, default=0)
    p.add_argument("--area-privada", type=float, default=0)
    p.add_argument("--quartos", type=int, default=0)
    p.add_argument("--banheiros", type=int, default=0)
    p.add_argument("--vagas", type=int, default=0)
    p.add_argument("--anos", type=int, default=10)
    p.add_argument(
        "--preco-atual",
        type=float,
        default=None,
        help="Opcional: projeta a partir deste preço em vez do estimado pelo modelo.",
    )
    p.add_argument(
        "--taxa-referencia",
        type=float,
        default=None,
        help="Opcional: taxa anual em %% no lugar do IPCA.",
    )
    p.add_argument(
        "--referencia",
        choices=REFERENCIAS,
        default="ipca-media",
        help="Cenário de referência (padrão: IPCA média anual de 10 anos).",
    )
    p.set_defaults(func=cmd_predict)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
