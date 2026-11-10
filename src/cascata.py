"""Monta as nove cascatas de florestas avaliadas, com e sem imputacao de rotulos.

Os hiperparametros reproduzem a configuracao de Ilidio et al. (IJCNN 2024,
secao 4.3) e nao sao ajustados por base: 150 arvores por floresta, folha minima
de cinco amostras, sqrt(n_atributos) candidatos por no, metade das amostras
recebidas pela cascata em cada arvore e no maximo dez niveis. Alterar qualquer
um deles rompe a comparabilidade com os numeros publicados.

Cada cascata e definida por tres escolhas, resumidas em COMPONENTES:

    aumento de atributos   o que e propagado de um nivel ao seguinte
                           OS = probabilidades preditas (espaco de saida)
                           TE = tree-embeddings
    cruzamento             se a representacao de uma floresta alimenta a outra
    imputacao              qual mecanismo converte zeros em positivos, se algum

Os tres mecanismos de imputacao:

    LC    unanimidade entre as florestas, limiar 0,4, validacao cruzada interna
          e limite de imputacoes estimado pelo TIcE (Wang, Yang e Li, 2020)
    SLC   media das probabilidades out-of-bag, limiar 0,5 e limite fixo que
          nunca pode ser ultrapassado (Ilidio et al., 2024)
    FLA   media das probabilidades out-of-bag divididas pela frequencia de
          rotulo reestimada em cada nivel, sem limite (Ilidio et al., 2024)
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.pipeline import FeatureUnion, Pipeline

from .terceiros import label_complement
from .terceiros.cascade_learn import weak_labels
from .terceiros.cascade_learn.cascade import AlternatingLevel, Cascade, SequentialLevel
from .terceiros.cascade_learn.estimator_adapters import (
    EstimatorAsTransformer,
    MultiOutputVotingRegressor,
    RegressorAsBinaryClassifier,
    UnanimityClassifier,
)
from .terceiros.cascade_learn.tree_embedder import ForestEmbedder

SEMENTE = 0

# Configuracao de Ilidio et al. (IJCNN 2024), sem otimizacao de hiperparametros.
N_ARVORES = 150
MAX_AMOSTRAS = 0.5          # "half the initial number of samples"
MIN_FOLHA = 5
MAX_NIVEIS = 10             # "a maximum of 10 levels for all cascades"
N_COMPONENTES_PCA = 0.8     # fracao de variancia retida nos tree-embeddings
MAX_NO_RF = 0.95            # nos presentes em mais que isso sao descartados
MAX_NO_ET = 0.8
PERCENTIL_FREQUENCIA = 0.95  # p das Equacoes do SLC e do FLA
LIMIAR_LC = 0.4             # theta do mecanismo LC

#: Componentes de cada metodo, como na tabela do artigo.
COMPONENTES = {
    "final_estimator": ("nenhum", False, None),
    "gcforest": ("OS", False, None),
    "cafe": ("TE", True, None),
    "cafe_os": ("TE e OS", True, None),
    "lcforest": ("OS", False, "LC"),
    "slcforest": ("OS", True, "SLC"),
    "flaforest": ("OS", True, "FLA"),
    "cafe_slc": ("TE e OS", True, "SLC"),
    "cafe_fla": ("TE e OS", True, "FLA"),
}

#: Nomes de exibicao, usados nas tabelas e nos graficos.
NOMES = {
    "final_estimator": "RF+ET",
    "gcforest": "gcForest",
    "cafe": "CaFE",
    "cafe_os": "CaFE-OS",
    "lcforest": "LCForest",
    "slcforest": "SLCForest",
    "flaforest": "FLAForest",
    "cafe_slc": "CaFE-SLC",
    "cafe_fla": "CaFE-FLA",
}

METODOS = tuple(COMPONENTES)

#: Metodos que imputam rotulos, isto e, que tratam os zeros como desconhecidos.
COM_IMPUTACAO = tuple(k for k, (_, _, imp) in COMPONENTES.items() if imp)

#: Metodos de cascata sem imputacao, usados como controle: cada um compartilha a
#: arquitetura de um metodo da lista acima e difere apenas pela ausencia do
#: imputador.
SEM_IMPUTACAO = tuple(k for k, (_, _, imp) in COMPONENTES.items() if not imp)


class _Densificador(BaseEstimator, TransformerMixin):
    """Converte matriz esparsa em densa. A PCA do sklearn nao aceita esparsa."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if scipy.sparse.issparse(X):
            return X.toarray()
        return X


def _construir(n_jobs: int, verbose: bool) -> dict:
    """Instancia as nove cascatas. Chamada uma vez por objeto Cascata."""
    params_floresta = dict(
        n_estimators=N_ARVORES,
        max_samples=MAX_AMOSTRAS,
        max_features="sqrt",
        min_samples_leaf=MIN_FOLHA,
        max_depth=None,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=SEMENTE,
    )
    params_cascata = dict(
        max_levels=MAX_NIVEIS,
        verbose=0,
        memory=None,
        keep_original_features=True,
    )

    rf_embedder = ForestEmbedder(
        RandomForestRegressor(**params_floresta, bootstrap=True),
        method="path",
        node_weights="log_node_size",
        max_node_size=MAX_NO_RF,
    )
    et_embedder = ForestEmbedder(
        ExtraTreesRegressor(**params_floresta, bootstrap=True),
        method="path",
        node_weights="log_node_size",
        max_node_size=MAX_NO_ET,
    )

    rf_embedder_pca = Pipeline([
        ("embedder", rf_embedder),
        ("densifier", _Densificador()),
        ("pca", PCA(n_components=N_COMPONENTES_PCA, random_state=SEMENTE)),
    ])
    et_embedder_pca = Pipeline([
        ("embedder", et_embedder),
        ("densifier", _Densificador()),
        ("pca", PCA(n_components=N_COMPONENTES_PCA, random_state=SEMENTE)),
    ])

    rf_proba = EstimatorAsTransformer(
        RandomForestRegressor(**params_floresta, bootstrap=True)
    )
    et_proba = EstimatorAsTransformer(
        ExtraTreesRegressor(**params_floresta, bootstrap=True)
    )

    # Estimador final: media das probabilidades de uma RF e de uma ET. Usado
    # como ultima camada de toda cascata e, isolado, como o metodo RF+ET.
    def _voting():
        return MultiOutputVotingRegressor(
            estimators=[
                ("rf", RandomForestRegressor(
                    **params_floresta, oob_score=True, bootstrap=True)),
                ("xt", ExtraTreesRegressor(
                    **params_floresta, oob_score=True, bootstrap=True)),
            ],
        )

    estimador_final = RegressorAsBinaryClassifier(_voting())

    nivel_fixo_proba = FeatureUnion([("rf", rf_proba), ("xt", et_proba)])

    nivel_alternado_embedding = AlternatingLevel([
        ("rf", rf_embedder_pca),
        ("xt", et_embedder_pca),
    ])
    nivel_alternado_proba = AlternatingLevel([
        ("xt", et_proba),
        ("rf", rf_proba),
    ])
    nivel_alternado_embedding_proba = AlternatingLevel([
        ("rf", FeatureUnion([("embedder", rf_embedder_pca), ("proba", rf_proba)])),
        ("xt", FeatureUnion([("embedder", et_embedder_pca), ("proba", et_proba)])),
    ])

    # Imputadores: SLC e FLA (Ilidio et al., 2024) e LC (Wang, Yang e Li, 2020).
    imputador_slc = weak_labels.LabelComplementImputer(
        label_freq_percentile=PERCENTIL_FREQUENCIA,
        verbose=verbose,
        estimator=_voting(),
        weight_proba=False,
    )
    imputador_fla = weak_labels.SCARImputer(
        label_freq_percentile=PERCENTIL_FREQUENCIA,
        verbose=verbose,
        estimator=_voting(),
    )
    imputador_lc = label_complement.LabelComplementImputer(
        estimator=UnanimityClassifier(
            estimators=[
                ("rf", RandomForestClassifier(**params_floresta)),
                # bootstrap explicito: ExtraTrees usa False por padrao, e
                # max_samples exige bootstrap ativo
                ("et", ExtraTreesClassifier(**params_floresta, bootstrap=True)),
            ],
            threshold=LIMIAR_LC,
        ),
        verbose=verbose,
        # Sem random_state, como no codigo que produziu o artigo: o TIcE sorteia
        # suas particoes internas com o gerador GLOBAL do numpy. Na maioria dos
        # folds o sorteio nao altera nada; em alguns, muda o nivel em que a
        # imputacao para, e com ele a predicao (na base do artigo, os folds 3
        # e 5; no pior caso observado, 3 dos 180 valores da tabela mudaram
        # 0,001). Fixar a semente aqui torna o LCForest deterministico, mas o
        # afasta dos numeros publicados.
        tice_params=dict(max_bepp=5, max_splits=500, min_set_size=5),
        cv_params=dict(cv=5),
    )

    def _cascata(nivel):
        return clone(Cascade(
            level=nivel, final_estimator=estimador_final, **params_cascata
        ))

    def _com_imputador(nivel, imputador):
        return _cascata(SequentialLevel([
            ("alternating_forests", nivel),
            ("label_imputer", imputador),
        ]))

    return {
        "final_estimator": estimador_final,
        "gcforest": _cascata(nivel_fixo_proba),
        "cafe": _cascata(nivel_alternado_embedding),
        "cafe_os": _cascata(nivel_alternado_embedding_proba),
        "lcforest": _cascata(SequentialLevel([
            ("transformer", nivel_fixo_proba),
            ("label_imputer", imputador_lc),
        ])),
        "slcforest": _com_imputador(nivel_alternado_proba, imputador_slc),
        "flaforest": _com_imputador(nivel_alternado_proba, imputador_fla),
        "cafe_slc": _com_imputador(nivel_alternado_embedding_proba, imputador_slc),
        "cafe_fla": _com_imputador(nivel_alternado_embedding_proba, imputador_fla),
    }


class Cascata:
    """Uma das nove cascatas, no cenario indutivo multirrotulo.

    Exemplo:
        >>> modelo = Cascata("cafe_fla")
        >>> modelo.fit(x_treino, y_treino)
        >>> probabilidades = modelo.predict_proba(x_teste)
    """

    def __init__(self, metodo: str, n_jobs: int = -1, verbose: bool = False):
        if metodo not in COMPONENTES:
            raise ValueError(
                f"metodo desconhecido: {metodo!r}. "
                f"Disponiveis: {', '.join(METODOS)}"
            )
        self.metodo = metodo
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.modelo = copy.deepcopy(_construir(n_jobs, verbose)[metodo])

    def fit(self, x, y):
        x = _matriz(x)
        y = _matriz(y)
        self.modelo.fit(x, y)
        return self

    def predict_proba(self, x) -> np.ndarray:
        """Probabilidades da classe positiva, no formato (n_amostras, n_rotulos).

        Os estimadores devolvem uma lista de matrizes (n_amostras, 2), uma por
        rotulo. A coluna 1 e a da classe positiva.
        """
        bruto = np.array(self.modelo.predict_proba(_matriz(x)))[:, :, 1:2].T
        return np.squeeze(bruto)

    @property
    def componentes(self) -> tuple:
        return COMPONENTES[self.metodo]

    @property
    def nome(self) -> str:
        return NOMES[self.metodo]

    def __repr__(self) -> str:
        aumento, cruzamento, imputacao = self.componentes
        return (
            f"Cascata({self.metodo!r}: aumento={aumento}, "
            f"cruzamento={'sim' if cruzamento else 'nao'}, "
            f"imputacao={imputacao or 'nenhuma'})"
        )


def _matriz(dados) -> np.ndarray:
    if isinstance(dados, (pd.DataFrame, pd.Series)):
        return dados.values
    return np.asarray(dados)
