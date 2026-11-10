"""Gera uma base sintetica no formato do pacote, para verificar a instalacao.

MOTIVO
------
Este repositorio nao distribui as bases de lncRNA. Sem nenhum dado, quem clona
nao consegue distinguir "o codigo esta quebrado" de "faltam os dados". A base
gerada aqui resolve isso: e pequena, roda em segundos e exercita o caminho
completo, de gerar as particoes a calcular as metricas.

Ela NAO serve para comparar metodos. Os numeros que sairem dela nao dizem nada
sobre desempenho: servem apenas para confirmar que a instalacao funciona.

COMO OS DADOS SAO GERADOS
-------------------------
1. Cada rotulo recebe um vetor de pesos aleatorio sobre os atributos, e uma
   instancia e positiva quando o produto interno fica acima de um corte
   escolhido para fixar a prevalencia desejada. Os rotulos ficam, assim,
   correlacionados entre si (compartilham atributos) e aprendiveis, que e o
   que um teste de fumaca precisa.

2. Uma fracao dos positivos e entao APAGADA, por rotulo. E isso que torna a
   base fracamente rotulada: o zero resultante e indistinguivel de um negativo
   real, exatamente como numa base biomedica incompleta.

Uso:
    python -m scripts.gerar_base_exemplo
    python -m scripts.gerar_base_exemplo --saida data/exemplo --instancias 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dados import ARQ_FEATURES, ARQ_LABELS, resumo  # noqa: E402
from src.splits import gerar_splits  # noqa: E402


def gerar(
    n_instancias: int,
    n_atributos: int,
    n_rotulos: int,
    fracao_oculta: float,
    semente: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(semente)

    x = rng.normal(size=(n_instancias, n_atributos))

    # Cada rotulo depende de uma combinacao linear dos atributos; o termo
    # constante e calibrado para que a prevalencia caia entre 15% e 55%, faixa
    # em que ha positivos e negativos suficientes em todos os folds.
    prevalencias = rng.uniform(0.15, 0.55, size=n_rotulos)
    y_completo = np.zeros((n_instancias, n_rotulos), dtype=int)

    for k in range(n_rotulos):
        pesos = rng.normal(size=n_atributos)
        escores = x @ pesos
        escores = (escores - escores.mean()) / escores.std()
        # o quantil garante a prevalencia exata, sem depender do formato da
        # distribuicao dos escores
        corte = np.quantile(escores, 1 - prevalencias[k])
        y_completo[:, k] = (escores >= corte).astype(int)

    # Apagamento dos positivos: e o que cria a rotulagem fraca. Ao menos um
    # positivo por rotulo e preservado, porque uma coluna zerada nao e
    # rotulagem fraca, e sim um rotulo ausente.
    y_observado = y_completo.copy()
    for k in range(n_rotulos):
        positivos = np.flatnonzero(y_completo[:, k] == 1)
        n_ocultar = min(int(round(len(positivos) * fracao_oculta)), len(positivos) - 1)
        if n_ocultar > 0:
            y_observado[rng.choice(positivos, n_ocultar, replace=False), k] = 0

    ids = [f"inst{i:04d}" for i in range(n_instancias)]
    cols_x = [f"atributo{j:02d}" for j in range(n_atributos)]
    cols_y = [f"rotulo{k:02d}" for k in range(n_rotulos)]

    return (
        pd.DataFrame(x, index=ids, columns=cols_x).rename_axis("id"),
        pd.DataFrame(y_observado, index=ids, columns=cols_y).rename_axis("id"),
        pd.DataFrame(y_completo, index=ids, columns=cols_y).rename_axis("id"),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--saida", default="data/exemplo",
                    help="diretorio da base (padrao: data/exemplo)")
    ap.add_argument("--instancias", type=int, default=150)
    ap.add_argument("--atributos", type=int, default=20)
    ap.add_argument("--rotulos", type=int, default=6)
    ap.add_argument("--fracao-oculta", type=float, default=0.3,
                    help="fracao dos positivos apagada por rotulo (padrao: 0,3)")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--semente", type=int, default=0)
    args = ap.parse_args()

    if not 0 <= args.fracao_oculta < 1:
        print("erro: --fracao-oculta deve estar em [0, 1)", file=sys.stderr)
        return 2

    destino = Path(args.saida)
    destino.mkdir(parents=True, exist_ok=True)

    x, y, y_completo = gerar(
        args.instancias, args.atributos, args.rotulos,
        args.fracao_oculta, args.semente,
    )

    x.to_csv(destino / ARQ_FEATURES)
    y.to_csv(destino / ARQ_LABELS)

    splits = gerar_splits(x, y, n_folds=args.folds, semente=args.semente)
    splits.to_csv(destino / "features_splits.csv", index=False)

    ocultados = int((y_completo.values == 1).sum() - (y.values == 1).sum())
    print(f"base de exemplo gravada em {destino.resolve()}")
    print(f"  {resumo(x, y)}")
    print(f"  {ocultados} positivos ocultados "
          f"({ocultados / max((y_completo.values == 1).sum(), 1):.0%} do total)")
    print(f"  {args.folds} folds em features_splits.csv")
    print()
    print("Esta base serve para verificar a instalacao, nao para comparar")
    print("metodos. Proximo passo:")
    print(f"  python -m scripts.rodar --dados {args.saida} "
          f"--metodos gcforest cafe_fla br_rf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
