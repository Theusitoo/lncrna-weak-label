"""Metodos classicos de classificacao multirrotulo, usados como referencia.

Estes metodos tratam cada zero como negativo confirmado. Entram na comparacao
justamente para medir o custo dessa suposicao quando a base esta incompleta.

COMPARABILIDADE
---------------
Os classificadores baseados em floresta usam os MESMOS hiperparametros das
cascatas (ver `cascata.py`): 150 arvores, folha minima de cinco amostras,
sqrt(n_atributos) por no e metade das amostras por arvore. Sem isso, a diferenca
observada mediria a configuracao da floresta, e nao a estrategia multirrotulo.

DESENHO FATORIAL
----------------
As tres estrategias de transformacao aparecem com os dois mesmos classificadores
base, {BR, CC, ECC} x {RF, MLP}. O cruzamento e proposital: assim a leitura
consegue separar o efeito da estrategia do efeito do classificador.

    BR   Binary Relevance: um classificador binario independente por rotulo
    CC   Classifier Chain: uma cadeia, na ordem original dos rotulos
    ECC  Ensemble of Classifier Chains: 10 cadeias com permutacoes sorteadas de
         forma independente, probabilidades combinadas pela media
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .cascata import MAX_AMOSTRAS, MIN_FOLHA, N_ARVORES, SEMENTE

N_CADEIAS = 10

# identicos aos das florestas das cascatas
PARAMS_FLORESTA = dict(
    n_estimators=N_ARVORES,
    max_features="sqrt",
    min_samples_leaf=MIN_FOLHA,
    max_samples=MAX_AMOSTRAS,
    bootstrap=True,
    n_jobs=-1,
    random_state=SEMENTE,
)


def rf():
    return RandomForestClassifier(**PARAMS_FLORESTA)


def mlp():
    """Rede rasa: uma unica camada oculta de 100 unidades.

    A padronizacao e obrigatoria, nao cosmetica: sem ela o gradiente nao
    converge em atributos de escalas diferentes, e a rede fica abaixo do
    aleatorio.
    """
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=1000,
            early_stopping=True,
            random_state=SEMENTE,
        ),
    )


class ECC:
    """Ensemble de Classifier Chains: media das probabilidades de N cadeias.

    Cada cadeia percorre os rotulos numa permutacao sorteada de forma
    independente; a media reduz a dependencia do resultado a uma ordem
    especifica. O limiar de decisao e aplicado somente a media, nunca as cadeias
    individuais.
    """

    def __init__(self, base=None, n_cadeias: int = N_CADEIAS, random_state: int = 0):
        self.base = base or rf
        self.n_cadeias = n_cadeias
        self.random_state = random_state

    def fit(self, X, Y):
        rng = np.random.RandomState(self.random_state)
        self.cadeias_ = []
        for i in range(self.n_cadeias):
            ordem = list(rng.permutation(Y.shape[1]))
            cadeia = ClassifierChain(
                self.base(), order=ordem, random_state=self.random_state + i
            )
            cadeia.fit(X, Y)
            self.cadeias_.append(cadeia)
        return self

    def predict_proba(self, X):
        return np.mean([c.predict_proba(X) for c in self.cadeias_], axis=0)


#: Os seis metodos do desenho fatorial.
METODOS = {
    "br_rf": ("BR-RF", lambda: MultiOutputClassifier(rf())),
    "br_mlp": ("BR-MLP", lambda: MultiOutputClassifier(mlp())),
    # sem `order`, o ClassifierChain usa a ordem original das colunas
    "cc_rf": ("CC-RF", lambda: ClassifierChain(rf(), random_state=SEMENTE)),
    "cc_mlp": ("CC-MLP", lambda: ClassifierChain(mlp(), random_state=SEMENTE)),
    "ecc_rf": ("ECC-RF", lambda: ECC(rf, random_state=SEMENTE)),
    "ecc_mlp": ("ECC-MLP", lambda: ECC(mlp, random_state=SEMENTE)),
}

NOMES = {chave: nome for chave, (nome, _) in METODOS.items()}


def construir(metodo: str):
    """Instancia um metodo classico pela chave."""
    if metodo not in METODOS:
        raise ValueError(
            f"metodo classico desconhecido: {metodo!r}. "
            f"Disponiveis: {', '.join(METODOS)}"
        )
    return METODOS[metodo][1]()


def probabilidades(modelo, X, n_rotulos: int) -> np.ndarray:
    """Normaliza a saida para uma matriz (n_amostras, n_rotulos).

    O MultiOutputClassifier (BR) devolve uma lista com uma matriz por rotulo;
    o ClassifierChain e o ECC devolvem diretamente o array denso.
    """
    p = modelo.predict_proba(X)

    if isinstance(p, list):
        saida = np.zeros((X.shape[0], n_rotulos))
        for j, pj in enumerate(p):
            # rotulo constante no treino: predict_proba traz uma coluna so, e
            # a classe presente e a negativa, logo a probabilidade do positivo
            # e zero
            saida[:, j] = pj[:, 1] if pj.shape[1] > 1 else 0.0
        return saida

    return np.asarray(p)
