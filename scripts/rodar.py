"""Treina os metodos sobre uma base e grava as probabilidades preditas por fold.

Este e o ponto de entrada unico: cascatas de florestas e metodos classicos
passam pelo mesmo laco, pelas mesmas particoes e gravam no mesmo formato. Assim
`avaliar.py` consegue compara-los sem saber de qual grupo cada um veio, e a
unica variavel entre dois resultados e o algoritmo.

SAIDA
-----
Um arquivo por metodo e por fold, com as probabilidades do conjunto de teste:

    labels_<metodo>_inductive_multilabel_predictions_test_fold<N>.csv

As linhas seguem a ordem das posicoes em `test_fold<N>` e as colunas sao os
rotulos, com o mesmo nome e na mesma ordem de `labels.csv`. O trecho
`inductive_multilabel` nao e decorativo: mantem os nomes compativeis com os
arquivos gerados pelo codigo original de Ilidio et al., de modo que resultados
antigos continuem legiveis pelos scripts deste repositorio.

RETOMADA
--------
Um metodo cujos dez arquivos ja existem e pulado. Uma execucao completa dos
quinze metodos leva cerca de meia hora, e uma interrupcao no meio nao deve
custar o que ja foi feito. Use `--refazer` para ignorar o que existe e
recalcular.

Uso:
    python -m scripts.rodar --dados data/exemplo --metodos gcforest cafe_fla
    python -m scripts.rodar --dados data/minha_base          # os 15 metodos
    python -m scripts.rodar --listar
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src  # noqa: E402
from src import cascata, classicos  # noqa: E402
from src.dados import BaseInvalida, carregar_base, numero_de_folds, particoes, resumo  # noqa: E402

#: Os quinze metodos do artigo: nove cascatas e seis classicos.
PADRAO = list(cascata.METODOS) + list(classicos.METODOS)


def nome_arquivo(metodo: str, fold: int) -> str:
    return f"labels_{metodo}_inductive_multilabel_predictions_test_fold{fold}.csv"


def ja_concluido(saida: Path, metodo: str, n_folds: int) -> bool:
    return all(
        (saida / nome_arquivo(metodo, f)).exists()
        for f in range(1, n_folds + 1)
    )


def treinar_fold(metodo: str, x_treino, y_treino, x_teste, n_rotulos, n_jobs, verbose):
    """Treina um metodo num fold e devolve a matriz de probabilidades."""
    if src.e_cascata(metodo):
        modelo = cascata.Cascata(metodo, n_jobs=n_jobs, verbose=verbose)
        modelo.fit(x_treino, y_treino)
        return modelo.predict_proba(x_teste)

    modelo = classicos.construir(metodo)
    modelo.fit(np.asarray(x_treino), np.asarray(y_treino))
    return classicos.probabilidades(modelo, np.asarray(x_teste), n_rotulos)


def listar() -> int:
    print("Cascatas de florestas")
    print(f"  {'chave':18s} {'nome':11s} aumento      cruzamento  imputacao")
    for chave in cascata.METODOS:
        aumento, cruzamento, imputacao = cascata.COMPONENTES[chave]
        print(f"  {chave:18s} {cascata.NOMES[chave]:11s} {aumento:12s} "
              f"{'sim' if cruzamento else 'nao':11s} {imputacao or 'nenhuma'}")
    print("\nMetodos multirrotulo classicos")
    for chave, (nome, _) in classicos.METODOS.items():
        print(f"  {chave:18s} {nome}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dados", help="diretorio da base")
    ap.add_argument("--metodos", nargs="+", default=None,
                    help=f"padrao: os {len(PADRAO)} metodos do artigo")
    ap.add_argument("--saida", default=None,
                    help="diretorio das predicoes (padrao: resultados/<base>)")
    ap.add_argument("--folds", type=int, default=None,
                    help="padrao: o numero de folds presente nas particoes")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--refazer", action="store_true",
                    help="recalcula metodos que ja possuem todos os folds")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--listar", action="store_true",
                    help="mostra os metodos disponiveis e encerra")
    args = ap.parse_args()

    if args.listar:
        return listar()
    if not args.dados:
        ap.error("--dados e obrigatorio (ou use --listar)")

    metodos = args.metodos or PADRAO
    desconhecidos = [m for m in metodos if m not in src.NOMES]
    if desconhecidos:
        print(f"erro: metodos desconhecidos: {', '.join(desconhecidos)}",
              file=sys.stderr)
        print("use --listar para ver os disponiveis", file=sys.stderr)
        return 2

    try:
        x, y, splits = carregar_base(args.dados, args.folds)
    except BaseInvalida as erro:
        print(f"erro na base: {erro}", file=sys.stderr)
        return 1

    n_folds = args.folds or numero_de_folds(splits)
    saida = Path(args.saida or Path("resultados") / Path(args.dados).name)
    saida.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"base      {Path(args.dados).resolve()}")
    print(f"           {resumo(x, y)}")
    print(f"folds     {n_folds}")
    print(f"metodos   {len(metodos)}: {', '.join(metodos)}")
    print(f"saida     {saida.resolve()}")
    print("=" * 78)

    falhas: list[str] = []
    for metodo in metodos:
        etiqueta = src.NOMES[metodo]

        if not args.refazer and ja_concluido(saida, metodo, n_folds):
            print(f"  [{etiqueta}] ja concluido, pulando")
            continue

        print(f"  [{etiqueta}] {src.grupo(metodo)}", flush=True)
        inicio = time.time()
        ok = True

        for fold, pos_treino, pos_teste in particoes(splits, n_folds):
            destino = saida / nome_arquivo(metodo, fold)
            if destino.exists() and not args.refazer:
                continue

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    proba = treinar_fold(
                        metodo,
                        x.iloc[pos_treino], y.iloc[pos_treino], x.iloc[pos_teste],
                        y.shape[1], args.n_jobs, args.verbose,
                    )
            except Exception as erro:
                print(f"    fold {fold}: FALHOU -- "
                      f"{type(erro).__name__}: {str(erro)[:120]}")
                if args.verbose:
                    traceback.print_exc()
                falhas.append(f"{etiqueta} fold {fold}")
                ok = False
                break

            proba = np.asarray(proba, dtype=float).reshape(len(pos_teste), -1)
            if proba.shape[1] != y.shape[1]:
                print(f"    fold {fold}: FALHOU -- o metodo devolveu "
                      f"{proba.shape[1]} colunas para {y.shape[1]} rotulos")
                falhas.append(f"{etiqueta} fold {fold}")
                ok = False
                break

            pd.DataFrame(proba, columns=y.columns).to_csv(destino, index=False)
            print(f"    fold {fold}/{n_folds} ok", flush=True)

        if ok:
            print(f"    concluido em {time.time() - inicio:.0f}s")

    print("\n" + "=" * 78)
    if falhas:
        print(f"{len(falhas)} execucoes falharam:")
        for f in falhas:
            print(f"  {f}")
        print("\nOs metodos que terminaram estao gravados. Rode de novo para "
              "retomar apenas o que falta.")
    else:
        print("todas as execucoes concluidas")
    print(f"\nProximo passo:\n  python -m scripts.avaliar --dados {args.dados} "
          f"--resultados {saida}")
    print("=" * 78)
    return 1 if falhas else 0


if __name__ == "__main__":
    raise SystemExit(main())
