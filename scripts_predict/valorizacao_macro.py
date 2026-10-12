"""Valorização v3: preço de imóveis explicado por juros, inflação e custo de construção.

Ideia (sugestão do orientador): a valorização de imóveis depende dos juros. Com só 5
meses de anúncios não dá para medir isso em Chapecó, mas o Banco Central tem 25 anos de
dados REAIS, mês a mês (nada é simulado):

- IVG-R (SGS 21340): índice de valores de garantia de imóveis residenciais financiados;
- Selic (SGS 4189): taxa básica de juros, acumulada no mês e anualizada;
- IPCA (SGS 433): inflação;
- INCC (SGS 192): custo da construção civil.

O teste: prever a valorização do IVG-R nos 12 meses seguintes a partir das condições de
hoje, com validação no tempo em três cortes (treino até 2016, 2018 e 2020, teste nos
meses seguintes), contra três referências simples: "repete os últimos 12 meses", "média
dos últimos 10 anos" e "IPCA dos últimos 12 meses".

Resultado (docs/reports/valorizacao_macro.md): os modelos com juros só vencem no corte
de 2016 (Random Forest); em 2018 e em 2020 a referência "média dos últimos 10 anos"
é a melhor. Nenhum modelo supera as referências de forma consistente. A projeção do site continua usando a
média de longo prazo do IVG-R (valorização v2).

    python scripts_predict/valorizacao_macro.py          # baixa as séries, treina, salva
"""

import argparse
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados"
SERIES = {"ivgr": 21340, "selic": 4189, "ipca": 433, "incc": 192}
DEFAULT_CACHE = PROJECT_ROOT / "dados" / "series_macro.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "reports" / "valorizacao_macro.json"
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "reports" / "valorizacao_macro.md"
FIM_TREINO = "2018-12"
FEATURES = ["selic", "selic_var12", "ipca12", "incc12", "ivgr12"]


# ------------------------------------------------------------------ dados
def baixar_series(cache=DEFAULT_CACHE, atualizar=True, timeout=30):
    """Séries mensais do BCB desde 2000; cópia em dados/series_macro.json."""
    cache = Path(cache)
    if atualizar:
        try:
            dados = {}
            for nome, codigo in SERIES.items():
                params = {
                    "formato": "json",
                    "dataInicial": "01/01/2000",
                    "dataFinal": date.today().strftime("%d/%m/%Y"),
                }
                resp = requests.get(SGS_URL.format(codigo=codigo), params=params, timeout=timeout)
                resp.raise_for_status()
                dados[nome] = [{"data": i["data"], "valor": float(i["valor"])} for i in resp.json()]
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")
        except (requests.RequestException, ValueError, KeyError) as exc:
            print(f"[aviso] séries não baixadas do BCB ({exc}); usando {cache}.")
    return json.loads(cache.read_text(encoding="utf-8"))


def montar_base(series):
    """Tabela mensal com as variáveis de hoje e a valorização dos 12 meses seguintes."""
    frames = []
    for nome, itens in series.items():
        s = pd.Series(
            [i["valor"] for i in itens],
            index=pd.to_datetime([i["data"] for i in itens], dayfirst=True).to_period("M"),
            name=nome,
        )
        frames.append(s[~s.index.duplicated()])
    df = pd.concat(frames, axis=1).sort_index()

    def acum12(pct_mensal):
        return 100 * (np.exp(np.log1p(pct_mensal / 100).rolling(12).sum()) - 1)

    base = pd.DataFrame(index=df.index)
    base["selic"] = df["selic"]  # % ao ano
    base["selic_var12"] = df["selic"] - df["selic"].shift(12)  # p.p. em 12 meses
    base["ipca12"] = acum12(df["ipca"])
    base["incc12"] = acum12(df["incc"])
    base["ivgr12"] = 100 * (df["ivgr"] / df["ivgr"].shift(12) - 1)  # passado
    base["alvo"] = 100 * (df["ivgr"].shift(-12) / df["ivgr"] - 1)  # próximos 12 meses
    base["ivgr_media10a"] = 100 * ((df["ivgr"] / df["ivgr"].shift(120)) ** (1 / 10) - 1)
    return base, df


# ------------------------------------------------------------ modelos
def candidatos(seed=42):
    return {
        "Regressão Linear": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=seed),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=2, learning_rate=0.05, random_state=seed
        ),
    }


def avaliar(base, fim_treino=FIM_TREINO):
    """Testa nos meses depois de `fim_treino`, treinando só com alvos já conhecidos nele.

    O alvo de cada mês é a valorização dos 12 meses SEGUINTES. Por isso o treino termina
    12 meses antes do corte: assim nenhum alvo de treino usa preços do período de teste.
    """
    completo = base.dropna(subset=FEATURES + ["alvo"])
    corte = pd.Period(fim_treino, freq="M")
    treino = completo[completo.index <= corte - 12]
    teste = completo[completo.index > corte]
    y_te = teste["alvo"].to_numpy()

    def metricas(pred):
        erro = np.asarray(pred) - y_te
        return {
            "rmse_pp": float(np.sqrt(np.mean(erro**2))),
            "mae_pp": float(np.mean(np.abs(erro))),
            "acerto_direcao_pct": float(np.mean(np.sign(pred) == np.sign(y_te)) * 100),
        }

    linhas = [
        {"modelo": "Referência: repete os últimos 12 meses", **metricas(teste["ivgr12"])},
        {"modelo": "Referência: média dos últimos 10 anos", **metricas(teste["ivgr_media10a"].fillna(teste["ivgr12"]))},
        {"modelo": "Referência: IPCA dos últimos 12 meses", **metricas(teste["ipca12"])},
    ]
    previsoes = {}
    for nome, modelo in candidatos().items():
        modelo.fit(treino[FEATURES], treino["alvo"])
        pred = modelo.predict(teste[FEATURES])
        previsoes[nome] = pred
        linhas.append({"modelo": nome, **metricas(pred)})
    info = {
        "treino": f"{treino.index.min()} a {treino.index.max()}",
        "teste": f"{teste.index.min()} a {teste.index.max()}",
        "meses_treino": int(len(treino)),
        "meses_teste": int(len(teste)),
    }
    return linhas, previsoes, teste, info


def coeficientes(base):
    """Efeito de cada variável (regressão linear sem padronizar, base inteira)."""
    completo = base.dropna(subset=FEATURES + ["alvo"])
    lr = LinearRegression().fit(completo[FEATURES], completo["alvo"])
    return {f: float(c) for f, c in zip(FEATURES, lr.coef_)}, float(lr.intercept_)


CORTES = ("2016-12", "2018-12", "2020-12")


def avaliar_cortes(base, cortes=CORTES):
    """Repete a avaliação com três datas de corte, para não depender de uma só."""
    resultados = {}
    for corte in cortes:
        linhas, _, _, info = avaliar(base, fim_treino=corte)
        resultados[corte] = {"info": info, "linhas": linhas}
    return resultados


def relatorio_cortes(resultados, coefs):
    out = [
        "# Valorização v3: os juros ajudam a prever a valorização de imóveis?",
        "",
        "Gerado por `scripts_predict/valorizacao_macro.py`. Séries mensais reais do Banco",
        "Central: IVG-R (21340), Selic (4189), IPCA (433) e INCC (192). Alvo: valorização do",
        "IVG-R nos 12 meses seguintes (%). Variáveis: Selic, variação da Selic em 12 meses, IPCA",
        "e INCC dos últimos 12 meses e valorização dos últimos 12 meses. Cada corte treina com",
        "os meses até a data e testa nos meses seguintes, que o modelo nunca viu.",
        "",
    ]
    for corte, r in resultados.items():
        info = r["info"]
        out += [
            f"## Treino até {corte}",
            "",
            f"Treino: {info['treino']} ({info['meses_treino']} meses). Teste: {info['teste']} "
            f"({info['meses_teste']} meses).",
            "",
            "| Modelo | RMSE (p.p.) | MAE (p.p.) | Acerto da direção |",
            "|---|---:|---:|---:|",
        ]
        for linha in sorted(r["linhas"], key=lambda x: x["rmse_pp"]):
            out.append(
                f"| {linha['modelo']} | {linha['rmse_pp']:.2f} | {linha['mae_pp']:.2f} | "
                f"{linha['acerto_direcao_pct']:.0f}% |"
            )
        out.append("")
    nomes = {
        "selic": "Selic (+1 p.p.)",
        "selic_var12": "Variação da Selic em 12 meses (+1 p.p.)",
        "ipca12": "IPCA dos últimos 12 meses (+1 p.p.)",
        "incc12": "INCC dos últimos 12 meses (+1 p.p.)",
        "ivgr12": "Valorização dos últimos 12 meses (+1 p.p.)",
    }
    out += [
        "## Efeito de cada variável (regressão linear, todos os meses)",
        "",
        "Descritivo apenas: mostra a associação nos dados, não uma relação que preveja bem.",
        "",
        "| Variável | Efeito na valorização dos 12 meses seguintes |",
        "|---|---:|",
    ]
    out += [f"| {nomes[f]} | {c:+.2f} p.p. |" for f, c in coefs.items()]
    return chr(10).join(out) + chr(10)


def main():
    parser = argparse.ArgumentParser(description="Valorização v3 (modelo macroeconômico).")
    parser.add_argument("--sem-download", action="store_true", help="Usa só a cópia local das séries.")
    args = parser.parse_args()

    base, _ = montar_base(baixar_series(atualizar=not args.sem_download))
    resultados = avaliar_cortes(base)
    coefs, _ = coeficientes(base)
    saida = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "dados_ate": str(base.dropna(subset=FEATURES).index.max()),
        "cortes": resultados,
        "coeficientes": coefs,
    }
    DEFAULT_OUTPUT.write_text(json.dumps(saida, ensure_ascii=False), encoding="utf-8")
    DEFAULT_REPORT.write_text(relatorio_cortes(resultados, coefs), encoding="utf-8")
    for corte, r in resultados.items():
        melhor = min(r["linhas"], key=lambda x: x["rmse_pp"])
        print(f"Treino até {corte}: melhor = {melhor['modelo']} (RMSE {melhor['rmse_pp']:.2f} p.p.)")
        for linha in sorted(r["linhas"], key=lambda x: x["rmse_pp"]):
            print(f"    {linha['modelo']:<45} {linha['rmse_pp']:.2f}")
    print(f"Relatório: {DEFAULT_REPORT}")


if __name__ == "__main__":
    main()
