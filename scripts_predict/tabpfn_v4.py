"""Teste do TabPFN no protocolo da versão 4 (histórico com ajuste fuzzy), com corte de tempo.

O TabPFN foi pré-treinado com até ~10 mil linhas de treino. Com o histórico simulado, o
treino da v4 tem cerca de 11 mil (casas) e 15 mil (apartamentos) linhas. Para mostrar à
banca que ele foi testado, cada variante roda com o mesmo protocolo dos outros modelos
(validação cruzada de 5 partes agrupada por imóvel, erro só nas linhas reais, e depois o
teste com preços reais), em tentativas cada vez menores:

    1. todas as linhas de treino;
    2. se não terminar dentro do corte, 10 mil linhas sorteadas;
    3. se ainda não terminar, 3 mil linhas sorteadas.

Cada tentativa roda num processo separado e é interrompida se passar do corte (padrão:
60 minutos). O resultado de cada tentativa (terminou, cortada por tempo ou erro) fica em
`dados/tabpfn_v4/resumo.json`; as previsões da tentativa que terminou ficam em
`dados/tabpfn_v4/<tipo>_<versao>.json` e entram no ranking de `modelo_v4.py` e
`resultados_v4.py`.

    venv/Scripts/python scripts_predict/tabpfn_v4.py               # v2, v3.5 e v3.5-fast
    venv/Scripts/python scripts_predict/tabpfn_v4.py --corte 60 --tipos Casa
    TABPFN_VERSIONS=v3.5,v3.5-fast venv/Scripts/python scripts_predict/tabpfn_v4.py  # só as 3.5

O TabPFN 3.5 e o 3.5 Fast precisam do aceite da licença (token da Prior Labs); sem ele,
ficam registrados como pulados. O alvo é sempre o preço em reais: o TabPFN já transforma
o alvo internamente.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from imoveis_ml_hibrido import CATEGORICAL_COLUMNS, SEGMENT_TYPES
from modelo_v3 import NUM_V3, avaliar
from modelo_v4 import DEFAULT_NORMALIZED_DB, carregar_reais, dados_segmento
from sklearn.model_selection import GroupKFold
from tabpfn_model import TABPFN_VARIANTS, _variantes_ativas, make_tabpfn, requested_versions

warnings.filterwarnings("ignore")

SAIDA = Path("dados/tabpfn_v4")
RESUMO = SAIDA / "resumo.json"
CORTE_MIN = 60
TENTATIVAS = ["todas", 10_000, 3_000]
VERSOES_PADRAO = "v2,v3.5,v3.5-fast"


def arquivo(tipo, nome):
    return SAIDA / f"{tipo.lower()}_{TABPFN_VARIANTS[nome]}.json"


def worker(tipo, nome, linhas, seed=42):
    """Uma tentativa: validação cruzada + teste. Grava as previsões das linhas reais."""
    inicio = time.time()
    df = carregar_reais(DEFAULT_NORMALIZED_DB)
    split, treino, grupos, real = dados_segmento(df, tipo, seed)
    y = treino["preco"].to_numpy()
    maximo = len(treino) if linhas == "todas" else int(linhas)

    def modelo():
        m = make_tabpfn(NUM_V3, CATEGORICAL_COLUMNS, seed=seed, version=TABPFN_VARIANTS[nome])
        m.max_train_rows = maximo
        return m

    # Mesmas partes do cross_val_predict de modelo_v4.py; só as linhas reais são previstas.
    oof = np.full(len(treino), np.nan)
    for i, (tr, te) in enumerate(GroupKFold(n_splits=5).split(treino, y, grupos), start=1):
        te = te[real[te]]
        oof[te] = modelo().fit(treino.iloc[tr], y[tr]).predict(treino.iloc[te])
        print(f"  [{tipo}] {nome} ({linhas} linhas): parte {i}/5 em {(time.time() - inicio) / 60:.1f} min", flush=True)

    final = modelo().fit(treino, y)
    teste_tip, teste_comp = split["test_typical"], split["test_full"]
    pred_comp = np.asarray(final.predict(teste_comp), dtype=float)
    pred_tip = pred_comp[teste_comp.index.get_indexer(teste_tip.index)]
    SAIDA.mkdir(parents=True, exist_ok=True)
    arquivo(tipo, nome).write_text(
        json.dumps(
            {
                "tipo": tipo,
                "modelo": nome,
                "linhas_treino": int(min(maximo, len(treino))),
                "linhas_disponiveis": int(len(treino)),
                "minutos": (time.time() - inicio) / 60,
                "cv_imovel": treino.loc[real, "imovel"].astype(int).tolist(),
                "cv_previsto": oof[real].tolist(),
                "teste_tipico": pred_tip.tolist(),
                "teste_completo": pred_comp.tolist(),
                "cv": avaliar(y[real], oof[real]),
            }
        ),
        encoding="utf-8",
    )


def rodar(tipos, corte_min, retomar=False):
    os.environ.setdefault("TABPFN_VERSIONS", VERSOES_PADRAO)
    ativas, avisos = _variantes_ativas()
    pedidas = [n for n, v in TABPFN_VARIANTS.items() if v in requested_versions()]
    resumo = json.loads(RESUMO.read_text(encoding="utf-8")) if RESUMO.exists() else {"tentativas": []}
    # Cada nova rodada substitui as tentativas antigas dos tipos e variantes pedidos; com
    # `retomar`, mantém as já cortadas por tempo e segue do próximo degrau.
    refeitos = {(t, n) for t in tipos for n in pedidas}
    resumo["tentativas"] = [
        a for a in resumo["tentativas"]
        if (a["tipo"], a["modelo"]) not in refeitos or (retomar and a["status"] == "cortado")
    ]
    df = carregar_reais(DEFAULT_NORMALIZED_DB)
    resumo["corte_min"] = corte_min

    def registrar(**tentativa):
        resumo["tentativas"].append({**tentativa, "em": datetime.now().isoformat(timespec="seconds")})
        SAIDA.mkdir(parents=True, exist_ok=True)
        RESUMO.write_text(json.dumps(resumo, ensure_ascii=False, indent=1), encoding="utf-8")

    for tipo in tipos:
        for nome in pedidas:
            if nome not in ativas:
                motivo = next((a for a in avisos if nome in a), "variante indisponível")
                registrar(tipo=tipo, modelo=nome, linhas=None, status="pulado", minutos=0, detalhe=motivo)
                print(motivo, flush=True)
                continue
            arquivo(tipo, nome).unlink(missing_ok=True)
            feitas = {a["linhas"] for a in resumo["tentativas"] if (a["tipo"], a["modelo"]) == (tipo, nome)}
            # Cada parte da validação cruzada treina com 4/5 das linhas: um degrau que não
            # reduz as linhas das partes repetiria a tentativa anterior e é pulado.
            por_parte = len(dados_segmento(df, tipo)[1]) * 4 // 5
            anterior = None
            for linhas in TENTATIVAS:
                efetivas = por_parte if linhas == "todas" else min(int(linhas), por_parte)
                repetida, anterior = efetivas == anterior, efetivas
                if linhas in feitas:
                    continue
                if repetida:
                    registrar(tipo=tipo, modelo=nome, linhas=linhas, status="pulado", minutos=0,
                              detalhe=f"cada parte da validação já treina com {por_parte} linhas")
                    continue
                print(f"[{tipo}] {nome}: tentativa com {linhas} linhas (corte {corte_min} min)", flush=True)
                inicio = time.time()
                cmd = [sys.executable, __file__, "worker", "--tipo", tipo, "--modelo", nome, "--linhas", str(linhas)]
                try:
                    # stdout vai direto para o terminal (progresso); stderr guarda o erro, se houver.
                    proc = subprocess.run(cmd, timeout=corte_min * 60, stderr=subprocess.PIPE, text=True)
                    status = "ok" if proc.returncode == 0 else "erro"
                    detalhe = "" if status == "ok" else (proc.stderr.strip().splitlines() or [""])[-1]
                except subprocess.TimeoutExpired:
                    status, detalhe = "cortado", f"não terminou em {corte_min} minutos"
                minutos = (time.time() - inicio) / 60
                registrar(tipo=tipo, modelo=nome, linhas=linhas, status=status, minutos=minutos, detalhe=detalhe)
                print(f"[{tipo}] {nome} ({linhas} linhas): {status} em {minutos:.1f} min {detalhe}", flush=True)
                if status == "ok":
                    break


def main():
    parser = argparse.ArgumentParser(description="Teste do TabPFN na v4 com corte de tempo.")
    sub = parser.add_subparsers(dest="cmd")
    w = sub.add_parser("worker")
    w.add_argument("--tipo", required=True)
    w.add_argument("--modelo", required=True)
    w.add_argument("--linhas", required=True)
    parser.add_argument("--corte", type=float, default=CORTE_MIN, help="minutos por tentativa")
    parser.add_argument("--tipos", nargs="+", default=list(SEGMENT_TYPES))
    parser.add_argument("--retomar", action="store_true", help="mantém as tentativas já cortadas")
    args = parser.parse_args()
    if args.cmd == "worker":
        worker(args.tipo, args.modelo, args.linhas)
    else:
        rodar(args.tipos, args.corte, args.retomar)


if __name__ == "__main__":
    import tabpfn_v4

    tabpfn_v4.main()
