"""Leitura e validacao de uma base no formato esperado pelo pacote.

Uma base e um diretorio com tres arquivos CSV:

    features.csv         primeira coluna = identificador, demais = atributos
    labels.csv           primeira coluna = identificador, demais = rotulos 0/1
    features_splits.csv  colunas train_fold1..N e test_fold1..N, com as
                         POSICOES das linhas (base zero) de cada particao

O terceiro arquivo e gerado por `scripts/gerar_splits.py`. As posicoes sao
guardadas em disco, e nao recalculadas a cada execucao, para que todos os
metodos vejam exatamente as mesmas particoes: sem isso a comparacao entre
metodos mediria tambem a diferenca entre sorteios.

Ver `data/README.md` para o detalhamento do formato.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ARQ_FEATURES = "features.csv"
ARQ_LABELS = "labels.csv"
ARQ_SPLITS = "features_splits.csv"


class BaseInvalida(ValueError):
    """A base existe, mas nao respeita o formato documentado."""


def caminhos(diretorio) -> dict[str, Path]:
    d = Path(diretorio)
    return {
        "features": d / ARQ_FEATURES,
        "labels": d / ARQ_LABELS,
        "splits": d / ARQ_SPLITS,
    }


def carregar_atributos_e_rotulos(diretorio) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Le features.csv e labels.csv e confere a consistencia entre os dois."""
    c = caminhos(diretorio)

    for chave in ("features", "labels"):
        if not c[chave].exists():
            raise BaseInvalida(f"arquivo ausente: {c[chave]}")

    x = pd.read_csv(c["features"], index_col=0)
    y = pd.read_csv(c["labels"], index_col=0)

    if len(x) != len(y):
        raise BaseInvalida(
            f"{ARQ_FEATURES} tem {len(x)} linhas e {ARQ_LABELS} tem {len(y)}; "
            "os dois arquivos precisam descrever as mesmas instancias, na "
            "mesma ordem"
        )

    if not x.index.equals(y.index):
        raise BaseInvalida(
            "os identificadores de features.csv e labels.csv nao coincidem "
            "linha a linha. Alinhe os dois arquivos antes de prosseguir: as "
            "particoes sao guardadas por POSICAO, de modo que uma ordem "
            "diferente faria o treino usar os rotulos de outras instancias"
        )

    valores = np.unique(y.values)
    if not np.isin(valores, (0, 1)).all():
        raise BaseInvalida(
            f"{ARQ_LABELS} precisa conter apenas 0 e 1; encontrei "
            f"{valores[:8].tolist()}"
        )

    if not np.isfinite(x.values).all():
        raise BaseInvalida(
            f"{ARQ_FEATURES} contem valores ausentes ou infinitos; as "
            "florestas nao aceitam NaN"
        )

    vazios = y.columns[y.sum() == 0].tolist()
    if vazios:
        raise BaseInvalida(
            f"os rotulos {vazios} nao tem nenhum positivo. Uma coluna zerada "
            "nao e rotulagem fraca, e um rotulo ausente: remova-a da base"
        )

    return x, y


def carregar_splits(diretorio, n_folds: int | None = None) -> pd.DataFrame:
    """Le features_splits.csv e confere que ele cobre a base por completo."""
    c = caminhos(diretorio)
    if not c["splits"].exists():
        raise BaseInvalida(
            f"arquivo ausente: {c['splits']}\n"
            "Gere-o com: python -m scripts.gerar_splits --dados "
            f"{Path(diretorio)}"
        )

    splits = pd.read_csv(c["splits"])
    detectado = sum(1 for col in splits.columns if col.startswith("test_fold"))
    if detectado == 0:
        raise BaseInvalida(
            f"{ARQ_SPLITS} nao tem nenhuma coluna test_foldN"
        )

    n = detectado if n_folds is None else n_folds
    faltando = [
        col
        for i in range(1, n + 1)
        for col in (f"train_fold{i}", f"test_fold{i}")
        if col not in splits.columns
    ]
    if faltando:
        raise BaseInvalida(f"{ARQ_SPLITS} nao tem as colunas {faltando}")

    return splits


def carregar_base(
    diretorio, n_folds: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Le a base inteira e valida atributos, rotulos e particoes em conjunto."""
    x, y = carregar_atributos_e_rotulos(diretorio)
    splits = carregar_splits(diretorio, n_folds)

    n = n_folds or numero_de_folds(splits)
    for i in range(1, n + 1):
        treino = _posicoes(splits, f"train_fold{i}")
        teste = _posicoes(splits, f"test_fold{i}")

        if set(treino) & set(teste):
            raise BaseInvalida(
                f"fold {i}: ha instancias em treino e teste ao mesmo tempo"
            )
        fora = [p for p in np.concatenate([treino, teste]) if not 0 <= p < len(x)]
        if fora:
            raise BaseInvalida(
                f"fold {i}: as posicoes {fora[:5]} estao fora da base, que tem "
                f"{len(x)} instancias. As particoes foram geradas para outra "
                "versao dos dados: gere-as de novo"
            )

    return x, y, splits


def numero_de_folds(splits: pd.DataFrame) -> int:
    return sum(1 for col in splits.columns if col.startswith("test_fold"))


def _posicoes(splits: pd.DataFrame, coluna: str) -> np.ndarray:
    return splits[coluna].dropna().astype(int).to_numpy()


def particoes(splits: pd.DataFrame, n_folds: int | None = None):
    """Itera sobre os folds, devolvendo (fold, posicoes_treino, posicoes_teste).

    O numero do fold comeca em 1, como nos nomes das colunas e dos arquivos de
    predicao.
    """
    n = n_folds or numero_de_folds(splits)
    for i in range(1, n + 1):
        yield i, _posicoes(splits, f"train_fold{i}"), _posicoes(splits, f"test_fold{i}")


def resumo(x: pd.DataFrame, y: pd.DataFrame) -> str:
    """Uma linha com as dimensoes e a densidade de rotulos da base."""
    densidade = y.values.sum() / y.size
    cardinalidade = y.values.sum(axis=1).mean()
    return (
        f"{len(x)} instancias x {x.shape[1]} atributos x {y.shape[1]} rotulos | "
        f"densidade {densidade:.2%} | cardinalidade media {cardinalidade:.2f}"
    )
