"""Gera as particoes de validacao cruzada de uma base e grava features_splits.csv.

As particoes ficam em disco, e nao sao recalculadas a cada execucao, porque
todos os metodos precisam ver exatamente as mesmas. Sem isso, a diferenca
medida entre dois metodos ficaria misturada a diferenca entre dois sorteios.

Por consequencia: regerar as particoes de uma base invalida todos os resultados
ja calculados sobre ela. O script exige `--sobrescrever` para fazer isso.

Uso:
    python -m scripts.gerar_splits --dados data/minha_base
    python -m scripts.gerar_splits --dados data/minha_base --folds 5 --semente 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dados import (  # noqa: E402
    BaseInvalida,
    carregar_atributos_e_rotulos,
    carregar_splits,
    resumo,
)
from src.splits import conferir_cobertura, gerar_e_salvar  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dados", required=True,
                    help="diretorio com features.csv e labels.csv")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--semente", type=int, default=0)
    ap.add_argument("--sobrescrever", action="store_true",
                    help="substitui um features_splits.csv existente")
    args = ap.parse_args()

    if args.folds < 2:
        print("erro: --folds precisa ser pelo menos 2", file=sys.stderr)
        return 2

    try:
        x, y = carregar_atributos_e_rotulos(args.dados)
    except BaseInvalida as erro:
        print(f"erro na base: {erro}", file=sys.stderr)
        return 1

    print(resumo(x, y))

    minimo = y.sum().min()
    if minimo < args.folds:
        rotulo = y.columns[y.sum().argmin()]
        print(
            f"aviso: o rotulo {rotulo!r} tem apenas {minimo} positivos, menos "
            f"que os {args.folds} folds pedidos. Algum fold ficara sem nenhum "
            "positivo desse rotulo, e as metricas dele serao instaveis. "
            "Considere menos folds ou remover o rotulo.",
            file=sys.stderr,
        )

    try:
        destino = gerar_e_salvar(
            args.dados,
            n_folds=args.folds,
            semente=args.semente,
            sobrescrever=args.sobrescrever,
        )
    except FileExistsError as erro:
        print(f"erro: {erro}", file=sys.stderr)
        return 1

    splits = carregar_splits(args.dados, args.folds)
    conferir_cobertura(splits, len(x), args.folds)

    print(f"{args.folds} folds gravados em {destino}")
    print("cada instancia aparece em exatamente um conjunto de teste: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
