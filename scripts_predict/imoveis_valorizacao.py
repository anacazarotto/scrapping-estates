"""Modelo de VALORIZAÇÃO de imóveis ao longo dos meses.

Enquanto os outros scripts preveem o preço de um imóvel hoje, este prevê quanto o
preço anunciado deve variar em `h` meses (padrão h=1), a partir do histórico de
coletas mensais (tabela `historico_precos` gerada pelo `imoveis_ml.py unify`).

Fluxo
-----
1. Painel mensal: um preço por imóvel e mês (mediana das coletas do mês).
2. Índice de valorização por segmento (bairro + tipo): mediana da variação de preço
   dos MESMOS imóveis entre meses consecutivos ("repeat listings"). Segmentos com
   poucos imóveis caem para o índice do tipo e depois para o geral.
3. Pares supervisionados (imóvel, mês t) -> alvo log(preço[t+h] / preço[t]).
   Features conhecidas no mês t: características do imóvel, preço/m² relativo ao
   segmento em t, meses anunciado, variação anterior do próprio anúncio e o
   "momentum" do segmento até t. Nada do futuro entra nas features.
4. Validação temporal (walk-forward): o teste é sempre o(s) último(s) mês(es);
   o ranking dos modelos usa validação cruzada agrupada por imóvel só no treino.
5. Comparação obrigatória com baselines ingênuos: "preço não muda" e
   "variação média do segmento".

Limitações que devem constar no TCC: preço ANUNCIADO não é preço de venda; imóveis
vendidos saem do painel (viés de sobrevivência); com poucos meses o sinal é fraco,
então o modelo só é útil se superar os baselines no teste temporal.
"""

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from imoveis_ml import canonical_tipo, format_currency, normalize_text
from model_io import load_model, save_model
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tabpfn_model import TABPFN_NAME, make_tabpfn, tabpfn_enabled

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - dependência opcional
    XGBRegressor = None

try:
    from sklearn_compat import CatBoostRegressor
except ImportError:  # pragma: no cover - dependência opcional
    CatBoostRegressor = None

DEFAULT_UNIFIED_DB = Path("imoveis_unificado.db")
DEFAULT_MODEL_PATH = Path("modelos/valorizacao_modelo.pkl")
DEFAULT_REPORT_PATH = Path("docs/reports/modelo_valorizacao.md")
DEFAULT_INDEX_CSV = Path("docs/reports/indice_valorizacao_bairros.csv")

ALLOWED_TYPES = ("Casa", "Apartamento")
MIN_SEGMENT_PAIRS = 8  # mínimo de imóveis repetidos para usar o índice do segmento
RARE_BAIRRO_MIN_COUNT = 10
MAX_ABS_MONTHLY_LOG_CHANGE = np.log(1.5)  # |variação| > 50%/mês = erro de cadastro
STABLE_BAND = 0.005  # ±0,5% conta como "estável" na acurácia de direção
CV_FOLDS = 5

NUMERIC_FEATURES = [
    "area_ref",
    "quartos",
    "banheiros",
    "vagas",
    "log_preco_m2",
    "preco_rel_segmento",
    "meses_anunciado",
    "variacao_anterior",
    "momentum_segmento",
    "momentum_tipo",
]
CATEGORICAL_FEATURES = ["bairro", "tipo_imovel"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# --------------------------------------------------------------------------- dados
def load_panel(unified_db):
    """Painel mensal (codigo, mes, preco) + características do imóvel."""
    unified_db = Path(unified_db)
    if not unified_db.exists():
        raise SystemExit(
            f"{unified_db} não encontrado. Rode antes: make unify-db "
            "(com os bancos imoveis_DD_MM_AAAA.db na raiz do projeto)."
        )
    conn = sqlite3.connect(unified_db)
    try:
        hist = pd.read_sql_query(
            "SELECT codigo, data_coleta, preco FROM historico_precos WHERE preco > 0", conn
        )
        feats = pd.read_sql_query(
            """
            SELECT codigo, bairro, tipo_imovel, area_total, area_privada,
                   quartos, banheiros, vagas
            FROM imoveis
            """,
            conn,
        )
    except pd.errors.DatabaseError as exc:
        raise SystemExit(
            "Tabela historico_precos ausente. Refaça o unify com a versão atual: "
            "make unify-db"
        ) from exc
    finally:
        conn.close()

    if hist.empty:
        raise SystemExit("historico_precos está vazio.")

    hist["mes"] = pd.to_datetime(hist["data_coleta"]).dt.to_period("M")
    panel = hist.groupby(["codigo", "mes"], as_index=False)["preco"].median()
    panel["mes_n"] = panel["mes"].dt.year * 12 + panel["mes"].dt.month  # índice inteiro

    for col in ("area_total", "area_privada", "quartos", "banheiros", "vagas"):
        feats[col] = pd.to_numeric(feats[col], errors="coerce").fillna(0.0)
    feats["area_ref"] = feats["area_privada"].where(feats["area_privada"] > 0, feats["area_total"])
    feats["bairro"] = feats["bairro"].fillna("").map(normalize_text)
    feats["tipo_imovel"] = feats["tipo_imovel"].fillna("").map(canonical_tipo)
    feats = feats[feats["tipo_imovel"].isin(ALLOWED_TYPES) & (feats["area_ref"] > 0)]

    panel = panel.merge(
        feats[["codigo", "bairro", "tipo_imovel", "area_ref", "quartos", "banheiros", "vagas"]],
        on="codigo",
        how="inner",
    )
    panel["preco_m2"] = panel["preco"] / panel["area_ref"]
    panel = panel.sort_values(["codigo", "mes"]).reset_index(drop=True)
    return panel


def monthly_changes(panel):
    """Variação log de cada imóvel entre meses CONSECUTIVOS (m-1 -> m)."""
    work = panel.sort_values(["codigo", "mes"]).copy()
    work["preco_ant"] = work.groupby("codigo")["preco"].shift(1)
    consecutive = (work["mes_n"] - work.groupby("codigo")["mes_n"].shift(1)) == 1
    work = work[consecutive].copy()
    work["dlog"] = np.log(work["preco"] / work["preco_ant"])
    return work[work["dlog"].abs() <= MAX_ABS_MONTHLY_LOG_CHANGE]


# ----------------------------------------------------------------- índice de mercado
def trimmed_mean(values):
    """Média das variações mensais (já limitadas a ±50% em monthly_changes).

    Não usamos mediana nem média aparada: com preços "grudados" (nos dados reais ~96%
    dos anúncios não mudam de um mês para o outro) ambas dão 0 e o índice fica plano.
    """
    return float(np.mean(np.asarray(values, dtype=float))) if len(values) else 0.0


def build_indices(panel):
    """Variação mensal média (aparada) por segmento, por tipo e geral (repeat listings)."""
    changes = monthly_changes(panel)
    agg = {"media": ("dlog", trimmed_mean), "count": ("dlog", "size")}
    seg = changes.groupby(["bairro", "tipo_imovel", "mes"]).agg(**agg).reset_index()
    seg = seg[seg["count"] >= MIN_SEGMENT_PAIRS]
    tipo = changes.groupby(["tipo_imovel", "mes"]).agg(**agg).reset_index()
    geral = changes.groupby("mes").agg(**agg).reset_index()
    return {
        "segmento": {(r.bairro, r.tipo_imovel, str(r.mes)): r.media for r in seg.itertuples()},
        "tipo": {(r.tipo_imovel, str(r.mes)): r.media for r in tipo.itertuples()},
        "geral": {str(r.mes): r.media for r in geral.itertuples()},
        "tabela_tipo": tipo,
        "tabela_geral": geral,
        "tabela_segmento": seg,
    }


def segment_momentum(indices, bairro, tipo, mes):
    """Variação do segmento no mês `mes` (m-1 -> m), com fallback tipo -> geral."""
    key = str(mes)
    if (bairro, tipo, key) in indices["segmento"]:
        return indices["segmento"][(bairro, tipo, key)]
    if (tipo, key) in indices["tipo"]:
        return indices["tipo"][(tipo, key)]
    return indices["geral"].get(key, 0.0)


def type_momentum(indices, tipo, mes):
    key = str(mes)
    return indices["tipo"].get((tipo, key), indices["geral"].get(key, 0.0))


def cumulative_index_table(indices, panel):
    """Tabela (bairro, tipo) x mês com o índice acumulado (base 100 no 1º mês)."""
    months = sorted(panel["mes"].unique())
    rows = []
    segments = panel.groupby(["bairro", "tipo_imovel"]).size().reset_index(name="n")
    for seg in segments.itertuples():
        level, row = 100.0, {"bairro": seg.bairro, "tipo_imovel": seg.tipo_imovel, "imoveis": seg.n}
        for i, mes in enumerate(months):
            if i > 0:
                level *= float(np.exp(segment_momentum(indices, seg.bairro, seg.tipo_imovel, mes)))
            row[str(mes)] = round(level, 2)
        row["variacao_total_%"] = round(level - 100.0, 2)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["tipo_imovel", "variacao_total_%"], ascending=[True, False])


# ------------------------------------------------------------------ pares (t, t+h)
def segment_cross_section(panel):
    """Mediana de preço/m² por (bairro, tipo, mês) e por (tipo, mês)."""
    seg = panel.groupby(["bairro", "tipo_imovel", "mes"])["preco_m2"].agg(["median", "count"])
    tipo = panel.groupby(["tipo_imovel", "mes"])["preco_m2"].median()
    return seg, tipo


def build_features(panel, indices):
    """Features de cada (imóvel, mês t), usando só informação disponível em t."""
    work = panel.sort_values(["codigo", "mes"]).copy()
    work["meses_anunciado"] = work["mes_n"] - work.groupby("codigo")["mes_n"].transform("min")

    prev_preco = work.groupby("codigo")["preco"].shift(1)
    gap_ok = (work["mes_n"] - work.groupby("codigo")["mes_n"].shift(1)) == 1
    var_ant = np.log(work["preco"] / prev_preco)
    work["variacao_anterior"] = np.where(gap_ok, var_ant, 0.0)
    work["variacao_anterior"] = work["variacao_anterior"].clip(
        -MAX_ABS_MONTHLY_LOG_CHANGE, MAX_ABS_MONTHLY_LOG_CHANGE
    )

    seg_cs, tipo_cs = segment_cross_section(panel)
    ref = []
    for r in work.itertuples():
        key = (r.bairro, r.tipo_imovel, r.mes)
        if key in seg_cs.index and seg_cs.loc[key, "count"] >= 5:
            ref.append(seg_cs.loc[key, "median"])
        else:
            ref.append(tipo_cs.get((r.tipo_imovel, r.mes), np.nan))
    work["preco_rel_segmento"] = work["preco_m2"] / np.asarray(ref, dtype=float)
    work["preco_rel_segmento"] = work["preco_rel_segmento"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    work["log_preco_m2"] = np.log(work["preco_m2"].clip(lower=1.0))
    work["momentum_segmento"] = [
        segment_momentum(indices, r.bairro, r.tipo_imovel, r.mes) for r in work.itertuples()
    ]
    work["momentum_tipo"] = [type_momentum(indices, r.tipo_imovel, r.mes) for r in work.itertuples()]
    return work


def build_pairs(features, horizon=1):
    """Liga cada (imóvel, t) ao preço do MESMO imóvel em t+h."""
    fut = features[["codigo", "mes_n", "preco"]].copy()
    fut["mes_n"] = fut["mes_n"] - horizon
    fut = fut.rename(columns={"preco": "preco_futuro"})
    pairs = features.merge(fut, on=["codigo", "mes_n"], how="inner")
    pairs["alvo"] = np.log(pairs["preco_futuro"] / pairs["preco"])
    limit = MAX_ABS_MONTHLY_LOG_CHANGE * max(1, horizon)
    pairs = pairs[pairs["alvo"].abs() <= limit].copy()
    pairs["mes_alvo"] = pairs["mes"] + horizon
    return pairs.reset_index(drop=True)


# ----------------------------------------------------------------------- modelos
def make_preprocessor():
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=0)), ("sc", StandardScaler())]),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="constant", fill_value="")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def model_factories(seed=42):
    """Regressores de erro quadrático: preveem a valorização ESPERADA (média).

    A maioria dos anúncios não muda de preço num mês; o que é previsível é a chance
    e o tamanho das mudanças. O valor esperado incorpora isso (ex.: 30% de chance de
    cair 6% => -1,8%), por isso o critério principal é o RMSE, que é minimizado pela
    média. O MAE também é reportado, mas nele o "preço não muda" (mediana) é difícil
    de bater — ver o classificador de redução para o sinal de direção.
    """
    factories = {
        "Ridge": lambda: Ridge(alpha=10.0),
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=300, min_samples_leaf=10, random_state=seed, n_jobs=-1
        ),
        "Gradient Boosting": lambda: HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
            min_samples_leaf=30, l2_regularization=1.0, random_state=seed,
        ),
    }
    if XGBRegressor is not None:
        factories["XGBoost"] = lambda: XGBRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=10, random_state=seed, n_jobs=-1,
        )
    if CatBoostRegressor is not None:
        factories["CatBoost"] = lambda: CatBoostRegressor(
            iterations=500, depth=5, learning_rate=0.03, random_seed=seed, verbose=False
        )
    if tabpfn_enabled():
        factories[TABPFN_NAME] = lambda: make_tabpfn(NUMERIC_FEATURES, CATEGORICAL_FEATURES, seed=seed)
    return factories


def classifier_factories(seed=42):
    """Classificadores para P(redução de preço em h meses)."""
    return {
        "Regressão Logística": lambda: LogisticRegression(C=1.0, max_iter=2000),
        "Gradient Boosting": lambda: HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
            min_samples_leaf=30, l2_regularization=1.0, random_state=seed,
        ),
    }


def build_classifier(name, seed=42):
    return Pipeline([("prep", make_preprocessor()), ("model", classifier_factories(seed)[name]())])


def build_model(name, seed=42):
    estimator = model_factories(seed)[name]()
    if name == TABPFN_NAME:
        return Pipeline([("model", estimator)])
    return Pipeline([("prep", make_preprocessor()), ("model", estimator)])


def compute_rare_bairros(df):
    counts = df["bairro"].value_counts()
    return set(counts[counts < RARE_BAIRRO_MIN_COUNT].index)


def apply_rare_bairros(df, rare):
    out = df.copy()
    out.loc[out["bairro"].isin(rare), "bairro"] = "outros"
    return out


def growth_metrics(y_true_log, y_pred_log, preco_atual):
    """Métricas em pontos percentuais de valorização e em R$."""
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    true_pct = 100 * np.expm1(y_true_log)
    pred_pct = 100 * np.expm1(y_pred_log)
    preco_atual = np.asarray(preco_atual, dtype=float)

    def direction(x):
        return np.where(x > STABLE_BAND, 1, np.where(x < -STABLE_BAND, -1, 0))

    return {
        "mae_pp": float(mean_absolute_error(true_pct, pred_pct)),
        "rmse_pp": float(np.sqrt(np.mean((true_pct - pred_pct) ** 2))),
        "r2": float(r2_score(y_true_log, y_pred_log)) if len(y_true_log) > 1 else 0.0,
        "acuracia_direcao": float(np.mean(direction(y_true_log) == direction(y_pred_log))),
        "mae_reais": float(np.mean(np.abs(preco_atual * np.expm1(y_true_log) - preco_atual * np.expm1(y_pred_log)))),
        "n": int(len(y_true_log)),
    }


def temporal_split(pairs, test_months=1):
    """Treino = pares cujo mês-alvo é anterior aos últimos `test_months` meses."""
    target_months = sorted(pairs["mes_alvo"].unique())
    if len(target_months) < 2:
        raise SystemExit(
            "São necessários ao menos 2 meses-alvo para validar no tempo "
            f"(há {len(target_months)}). Com horizonte h, precisa de h+2 meses de coleta."
        )
    test_months = min(test_months, len(target_months) - 1)
    cut = target_months[-test_months]
    return pairs[pairs["mes_alvo"] < cut].copy(), pairs[pairs["mes_alvo"] >= cut].copy()


def run_benchmark(unified_db, horizon=1, seed=42, test_months=1):
    panel = load_panel(unified_db)
    indices = build_indices(panel)
    features = build_features(panel, indices)
    pairs = build_pairs(features, horizon=horizon)
    train, test = temporal_split(pairs, test_months=test_months)

    rare = compute_rare_bairros(train)
    train = apply_rare_bairros(train, rare)
    test = apply_rare_bairros(test, rare)
    X_tr, y_tr = train[FEATURES], train["alvo"].to_numpy()
    X_te, y_te = test[FEATURES], test["alvo"].to_numpy()

    folds = int(max(2, min(CV_FOLDS, train["codigo"].nunique() // 20)))
    cv = GroupKFold(n_splits=folds)

    rows = []
    # Baselines ingênuos
    rows.append({"modelo": "Baseline: preço não muda", "cv_rmse_pp": np.nan,
                 **growth_metrics(y_te, np.zeros_like(y_te), test["preco"])})
    seg_mean = train.groupby(["bairro", "tipo_imovel"])["alvo"].mean()
    tipo_mean = train.groupby("tipo_imovel")["alvo"].mean()
    base_seg = [
        seg_mean.get((b, t), tipo_mean.get(t, float(y_tr.mean())))
        for b, t in zip(test["bairro"], test["tipo_imovel"])
    ]
    rows.append({"modelo": "Baseline: média do segmento", "cv_rmse_pp": np.nan,
                 **growth_metrics(y_te, base_seg, test["preco"])})

    for name in model_factories(seed):
        try:
            cv_pred = cross_val_predict(build_model(name, seed), X_tr, y_tr, cv=cv, groups=train["codigo"])
            model = build_model(name, seed).fit(X_tr, y_tr)
            pred = model.predict(X_te)
        except (RuntimeError, ValueError) as exc:
            print(f"[aviso] {name} ignorado: {exc}")
            continue
        cv_rmse = float(np.sqrt(np.mean((100 * np.expm1(y_tr) - 100 * np.expm1(cv_pred)) ** 2)))
        rows.append({"modelo": name, "cv_rmse_pp": cv_rmse, **growth_metrics(y_te, pred, test["preco"])})

    # Classificação: o imóvel terá redução de preço (> STABLE_BAND) em h meses?
    yc_tr = (y_tr < -STABLE_BAND).astype(int)
    yc_te = (y_te < -STABLE_BAND).astype(int)
    base_rate = float(yc_tr.mean())
    clf_rows = [{
        "modelo": "Baseline: taxa histórica",
        "auc": 0.5,
        "brier": float(brier_score_loss(yc_te, np.full(len(yc_te), base_rate))) if len(yc_te) else 0.0,
    }]
    if 0 < yc_tr.sum() < len(yc_tr) and 0 < yc_te.sum() < len(yc_te):
        for name in classifier_factories(seed):
            clf = build_classifier(name, seed).fit(X_tr, yc_tr)
            proba = clf.predict_proba(X_te)[:, 1]
            clf_rows.append({
                "modelo": name,
                "auc": float(roc_auc_score(yc_te, proba)),
                "brier": float(brier_score_loss(yc_te, proba)),
            })

    info = {
        "horizonte_meses": horizon,
        "meses": [str(m) for m in sorted(panel["mes"].unique())],
        "imoveis_no_painel": int(panel["codigo"].nunique()),
        "pares_treino": int(len(train)),
        "pares_teste": int(len(test)),
        "meses_alvo_teste": sorted({str(m) for m in test["mes_alvo"]}),
        "cv_folds": folds,
        "fracao_sem_mudanca": float(np.mean(np.abs(pairs["alvo"]) < 1e-9)),
        "taxa_reducao_treino": base_rate,
        "taxa_reducao_teste": float(yc_te.mean()) if len(yc_te) else 0.0,
        "classificacao": clf_rows,
    }
    return rows, info, panel, indices, pairs


# ------------------------------------------------------------------- relatório
def build_report(rows, info, index_table):
    models = [r for r in rows if not r["modelo"].startswith("Baseline")]
    baseline = min((r for r in rows if r["modelo"].startswith("Baseline")), key=lambda r: r["rmse_pp"])
    best = min(models, key=lambda r: r["cv_rmse_pp"]) if models else None
    lines = [
        "# Modelo de valorização mensal",
        "",
        f"- Meses de coleta: {', '.join(info['meses'])}",
        f"- Horizonte previsto: {info['horizonte_meses']} mês(es) à frente",
        f"- Imóveis no painel: {info['imoveis_no_painel']}",
        f"- Pares treino / teste: {info['pares_treino']} / {info['pares_teste']}",
        f"- Mês(es)-alvo do teste: {', '.join(info['meses_alvo_teste'])} (nunca vistos no treino)",
        f"- Pares em que o preço anunciado não mudou: {100 * info['fracao_sem_mudanca']:.1f}%",
        "",
        "## Como foi avaliado",
        "",
        "- Alvo: variação do preço anunciado do mesmo imóvel entre t e t+h (em log).",
        "- Features só com informação disponível em t (sem olhar o futuro).",
        "- Teste = último(s) mês(es); ranking por validação cruzada agrupada por imóvel no treino.",
        "- Erros em **pontos percentuais** (p.p.): errar 1,0 p.p. = prever +2% quando foi +1%.",
        "- Critério principal: **RMSE** (o modelo prevê a valorização *esperada*). O MAE",
        "  favorece prever 0% porque a maioria dos anúncios não muda num mês.",
        f"- Acurácia de direção: sobe / estável (±{100 * STABLE_BAND:.1f}%) / cai.",
        "",
        "## Resultados no teste temporal",
        "",
        "| Modelo | RMSE CV treino (p.p.) | RMSE teste (p.p.) | MAE teste (p.p.) | R² (log) | Acurácia direção | MAE (R$) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda r: r["rmse_pp"]):
        cv = "-" if pd.isna(r["cv_rmse_pp"]) else f"{r['cv_rmse_pp']:.3f}"
        lines.append(
            f"| {r['modelo']} | {cv} | {r['rmse_pp']:.3f} | {r['mae_pp']:.3f} | {r['r2']:.4f} | "
            f"{100 * r['acuracia_direcao']:.1f}% | {format_currency(r['mae_reais'])} |"
        )
    lines += [
        "",
        f"## Chance de redução de preço (> {100 * STABLE_BAND:.1f}%) em {info['horizonte_meses']} mês(es)",
        "",
        f"Taxa de reduções: {100 * info['taxa_reducao_treino']:.1f}% no treino, "
        f"{100 * info['taxa_reducao_teste']:.1f}% no teste. AUC 0,5 = sorteio; Brier menor é melhor.",
        "",
        "| Classificador | AUC | Brier |",
        "|---|---:|---:|",
    ]
    for r in info["classificacao"]:
        lines.append(f"| {r['modelo']} | {r['auc']:.3f} | {r['brier']:.4f} |")
    lines += ["", "## Leitura", ""]
    if best:
        ganho = baseline["rmse_pp"] - best["rmse_pp"]
        lines.append(
            f"- Modelo escolhido (menor RMSE de CV): **{best['modelo']}** — RMSE no teste "
            f"{best['rmse_pp']:.3f} p.p. contra {baseline['rmse_pp']:.3f} p.p. do melhor baseline "
            f"({baseline['modelo']})."
        )
        if ganho > 0:
            lines.append(f"- Ganho sobre o baseline: {ganho:.3f} p.p.")
        else:
            lines.append(
                "- O modelo **não** superou o baseline no teste: com os meses disponíveis, a "
                "variação futura ainda não é previsível além da tendência do segmento. "
                "Isso é um resultado válido para o TCC; reavalie com mais meses de coleta."
            )
    lines += [
        "- Limitações: preço anunciado ≠ preço de venda; imóveis vendidos saem do painel "
        "(viés de sobrevivência); poucos meses = pouca variação observada.",
        "",
        "## Índice de valorização acumulada (base 100 no 1º mês)",
        "",
        "Segmentos com poucos imóveis repetidos usam o índice do tipo (Casa/Apartamento).",
        "",
    ]
    top = index_table[index_table["imoveis"] >= 20].head(30)
    if not top.empty:
        cols = list(top.columns)
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join("---" for _ in cols) + "|")
        for r in top.itertuples(index=False):
            lines.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------- comandos
def cmd_benchmark(args):
    rows, info, panel, indices, _ = run_benchmark(
        args.unified_db, horizon=args.horizonte, seed=args.seed, test_months=args.meses_teste
    )
    index_table = cumulative_index_table(indices, panel)
    report = Path(args.report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(build_report(rows, info, index_table), encoding="utf-8")
    index_table.to_csv(args.index_csv, index=False, encoding="utf-8")
    print(f"Relatório: {report} | Índice por bairro: {args.index_csv}")
    for r in sorted(rows, key=lambda r: r["rmse_pp"]):
        print(
            f"{r['modelo']:<30} RMSE={r['rmse_pp']:.3f} p.p.  MAE={r['mae_pp']:.3f} p.p.  "
            f"direção={100 * r['acuracia_direcao']:.1f}%"
        )
    for r in info["classificacao"]:
        print(f"[P(redução)] {r['modelo']:<26} AUC={r['auc']:.3f} Brier={r['brier']:.4f}")


def cmd_train(args):
    rows, info, panel, indices, pairs = run_benchmark(
        args.unified_db, horizon=args.horizonte, seed=args.seed, test_months=args.meses_teste
    )
    models = [r for r in rows if not r["modelo"].startswith("Baseline")]
    best = min(models, key=lambda r: r["cv_rmse_pp"])

    # Modelo final: todos os pares (treino + teste).
    rare = compute_rare_bairros(pairs)
    all_pairs = apply_rare_bairros(pairs, rare)
    model = build_model(best["modelo"], args.seed).fit(all_pairs[FEATURES], all_pairs["alvo"].to_numpy())

    last = panel["mes"].max()
    seg_cs, tipo_cs = segment_cross_section(panel[panel["mes"] == last])
    artifact = {
        "kind": "valorizacao_mensal_v1",
        "modelo_nome": best["modelo"],
        "model": model,
        "horizonte_meses": args.horizonte,
        "ultimo_mes": str(last),
        "rare_bairros": sorted(rare),
        "ref_preco_m2_segmento": {f"{k[0]}|{k[1]}": float(v) for k, v in seg_cs["median"].items()},
        "ref_preco_m2_tipo": {f"{k[0]}": float(v) for k, v in tipo_cs.items()},
        "momentum_segmento": {
            f"{b}|{t}": segment_momentum(indices, b, t, last)
            for b, t in panel[["bairro", "tipo_imovel"]].drop_duplicates().itertuples(index=False)
        },
        "momentum_tipo": {t: type_momentum(indices, t, last) for t in ALLOWED_TYPES},
        "metricas_teste": best,
        "info": info,
        "trained_at": datetime.now().isoformat(),
    }
    save_model(artifact, args.model_path)
    print(f"Modelo de valorização salvo em {args.model_path} ({best['modelo']})")
    print(f"RMSE no teste temporal: {best['rmse_pp']:.3f} p.p. | MAE: {best['mae_pp']:.3f} p.p.")


def cmd_predict(args):
    art = load_model(args.model_path)
    if not isinstance(art, dict) or art.get("kind") != "valorizacao_mensal_v1":
        raise SystemExit("Arquivo de modelo de valorização inválido.")
    tipo = canonical_tipo(args.tipo_imovel)
    if tipo not in ALLOWED_TYPES:
        raise SystemExit("Use Casa ou Apartamento.")
    bairro = normalize_text(args.bairro)
    area = float(args.area_privada or args.area_total or 0)
    if area <= 0 or args.preco_atual <= 0:
        raise SystemExit("Informe área e preço atual maiores que zero.")
    preco_m2 = args.preco_atual / area
    ref = art["ref_preco_m2_segmento"].get(f"{bairro}|{tipo}") or art["ref_preco_m2_tipo"].get(tipo)
    row = pd.DataFrame([{
        "area_ref": area,
        "quartos": args.quartos,
        "banheiros": args.banheiros,
        "vagas": args.vagas,
        "log_preco_m2": float(np.log(max(preco_m2, 1.0))),
        "preco_rel_segmento": preco_m2 / ref if ref else 1.0,
        "meses_anunciado": args.meses_anunciado,
        "variacao_anterior": float(np.log1p(args.variacao_anterior / 100.0)),
        "momentum_segmento": art["momentum_segmento"].get(f"{bairro}|{tipo}", art["momentum_tipo"].get(tipo, 0.0)),
        "momentum_tipo": art["momentum_tipo"].get(tipo, 0.0),
        "bairro": "outros" if bairro in set(art["rare_bairros"]) else bairro,
        "tipo_imovel": tipo,
    }])
    pred = float(art["model"].predict(row[FEATURES])[0])
    pct = 100 * np.expm1(pred)
    h = art["horizonte_meses"]
    print(f"Valorização prevista em {h} mês(es): {pct:+.2f}%")
    print(f"Preço estimado: {format_currency(args.preco_atual * np.exp(pred))} "
          f"(atual {format_currency(args.preco_atual)}; referência {art['ultimo_mes']})")
    m = art["metricas_teste"]
    print(f"Erro do modelo no teste temporal: RMSE {m['rmse_pp']:.2f} p.p. | MAE {m['mae_pp']:.2f} p.p.")


def build_parser():
    parser = argparse.ArgumentParser(description="Modelo de valorização mensal de imóveis.")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--unified-db", default=str(DEFAULT_UNIFIED_DB))
        p.add_argument("--horizonte", type=int, default=1, help="Meses à frente (padrão 1).")
        p.add_argument("--meses-teste", type=int, default=1, help="Últimos meses usados como teste.")
        p.add_argument("--seed", type=int, default=42)

    b = sub.add_parser("benchmark-valorizacao", help="Compara modelos no teste temporal.")
    common(b)
    b.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    b.add_argument("--index-csv", default=str(DEFAULT_INDEX_CSV))
    b.set_defaults(func=cmd_benchmark)

    t = sub.add_parser("train-valorizacao", help="Treina e salva o melhor modelo.")
    common(t)
    t.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    t.set_defaults(func=cmd_train)

    p = sub.add_parser("predict-valorizacao", help="Prevê a valorização de um imóvel.")
    p.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--bairro", required=True)
    p.add_argument("--tipo-imovel", required=True)
    p.add_argument("--area-total", type=float, default=0)
    p.add_argument("--area-privada", type=float, default=0)
    p.add_argument("--quartos", type=int, default=0)
    p.add_argument("--banheiros", type=int, default=0)
    p.add_argument("--vagas", type=int, default=0)
    p.add_argument("--preco-atual", type=float, required=True, help="Preço anunciado hoje (R$).")
    p.add_argument("--meses-anunciado", type=int, default=0, help="Há quantos meses está anunciado.")
    p.add_argument("--variacao-anterior", type=float, default=0.0, help="Variação do último mês, em %%.")
    p.set_defaults(func=cmd_predict)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
