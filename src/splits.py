"""Geracao das particoes de validacao cruzada com estratificacao multirrotulo.

A estratificacao comum olha uma classe por vez e nao serve aqui: com rotulos
raros, um sorteio simples produz folds em que um rotulo aparece zero vez no
treino ou zero vez no teste, e a metrica daquele rotulo passa a medir o sorteio.
A estratificacao multirrotulo de Sechidis, Tsoumakas e Vlahavas (2011) distribui
os positivos de cada rotulo entre os folds, e e a implementada pelo pacote
`iterative-stratification`.

As posicoes sao gravadas em disco (`features_splits.csv`) porque todos os
metodos precisam ver as MESMAS particoes. Recalcular a cada execucao tornaria a
diferenca entre dois metodos inseparavel da diferenca entre dois sorteios.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .dados import ARQ_SPLITS, carregar_atributos_e_rotulos


def gerar_splits(
    x: pd.DataFrame,
    y: pd.DataFrame,
    n_folds: int = 10,
    semente: int = 0,
) -> pd.DataFrame:
    """Devolve o DataFrame de particoes, com as POSICOES das linhas.

    Guardar posicoes, e nao identificadores, mantem o formato identico ao do
    codigo original de Ilidio et al. e dispensa qualquer busca por indice na
    hora de treinar.
    """
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    except ImportError as erro:  # pragma: no cover
        raise ImportError(
            "a estratificacao multirrotulo exige o pacote "
            "iterative-stratification: pip install iterative-stratification"
        ) from erro

    mskf = MultilabelStratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=semente
    )

    colunas: dict[str, pd.Series] = {}
    for i, (treino, teste) in enumerate(mskf.split(x.values, y.values), start=1):
        colunas[f"train_fold{i}"] = pd.Series(treino)
        colunas[f"test_fold{i}"] = pd.Series(teste)

    return pd.DataFrame(colunas)


def gerar_e_salvar(
    diretorio,
    n_folds: int = 10,
    semente: int = 0,
    sobrescrever: bool = False,
) -> Path:
    """Gera as particoes de uma base e grava features_splits.csv nela."""
    destino = Path(diretorio) / ARQ_SPLITS
    if destino.exists() and not sobrescrever:
        raise FileExistsError(
            f"{destino} ja existe. Use --sobrescrever para substituir.\n"
            "Atencao: substituir as particoes invalida todos os resultados ja "
            "calculados para esta base, porque eles foram obtidos sobre os "
            "folds antigos."
        )

    x, y = carregar_atributos_e_rotulos(diretorio)
    splits = gerar_splits(x, y, n_folds=n_folds, semente=semente)
    splits.to_csv(destino, index=False)
    return destino


def conferir_cobertura(splits: pd.DataFrame, n_instancias: int, n_folds: int) -> None:
    """Verifica que cada instancia aparece em exatamente um conjunto de teste."""
    vezes = np.zeros(n_instancias, dtype=int)
    for i in range(1, n_folds + 1):
        for p in splits[f"test_fold{i}"].dropna().astype(int):
            vezes[p] += 1
    if not (vezes == 1).all():
        faltam = int((vezes == 0).sum())
        repetem = int((vezes > 1).sum())
        raise ValueError(
            "as particoes nao formam uma validacao cruzada valida: "
            f"{faltam} instancias nunca sao testadas e {repetem} sao testadas "
            "mais de uma vez"
        )
