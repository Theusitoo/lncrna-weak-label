"""As seis metricas de avaliacao e os testes estatisticos pareados.

As metricas se dividem em dois grupos, e a distincao importa para a leitura dos
resultados:

    ORDENACAO (independem do limiar)      AUROC, AP, ranking loss
    IDENTIFICACAO (dependem do limiar)    Hamming, F1-macro, F1-micro

Um metodo que imputa rotulos desloca as probabilidades para cima e fica menos
conservador no limiar. Isso tende a favorecer o segundo grupo sem melhorar o
primeiro, e por isso os dois precisam ser reportados juntos: uma tabela com
apenas um dos grupos permite a leitura oposta.

As metricas por fold NUNCA sao calculadas sobre a concatenacao das predicoes de
todos os folds. A validacao cruzada produz dez estimativas independentes, e e
sobre essas dez que o teste pareado opera; concatenar primeiro descartaria a
variabilidade entre folds e inviabilizaria qualquer teste.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    label_ranking_loss,
    roc_auc_score,
)

LIMIAR = 0.5

#: nome -> (funcao(y, probabilidades, binarias), maior_e_melhor)
METRICAS = {
    "AUROC": (lambda y, p, b: roc_auc_score(y, p, average="micro"), True),
    "AP": (lambda y, p, b: average_precision_score(y, p, average="micro"), True),
    "RankLoss": (lambda y, p, b: label_ranking_loss(y, p), False),
    "Hamming": (lambda y, p, b: hamming_loss(y, b), False),
    "F1-macro": (
        lambda y, p, b: f1_score(y, b, average="macro", zero_division=0),
        True,
    ),
    "F1-micro": (
        lambda y, p, b: f1_score(y, b, average="micro", zero_division=0),
        True,
    ),
}

MAIOR_E_MELHOR = {nome: melhor for nome, (_, melhor) in METRICAS.items()}


def avaliar_fold(
    y_verdadeiro: np.ndarray,
    probabilidades: np.ndarray,
    limiar: float = LIMIAR,
) -> dict[str, float]:
    """Calcula as seis metricas de um unico fold.

    Uma metrica que nao pode ser definida vira NaN em vez de derrubar a
    execucao. Acontece, por exemplo, com a AUROC quando um fold de teste nao
    tem nenhum negativo em algum rotulo.
    """
    y_verdadeiro = np.asarray(y_verdadeiro)
    probabilidades = np.nan_to_num(np.asarray(probabilidades, dtype=float))
    binarias = (probabilidades >= limiar).astype(int)

    resultado = {}
    for nome, (funcao, _) in METRICAS.items():
        try:
            resultado[nome] = float(funcao(y_verdadeiro, probabilidades, binarias))
        except Exception:
            resultado[nome] = float("nan")
    return resultado


def tabela_por_fold(resultados: dict[str, dict[int, dict[str, float]]]) -> dict[str, pd.DataFrame]:
    """Reorganiza {metodo: {fold: {metrica: valor}}} em {metrica: DataFrame}.

    No DataFrame devolvido, as linhas sao os folds e as colunas os metodos, que
    e o formato exigido pelos testes pareados.
    """
    por_metrica: dict[str, dict[str, dict[int, float]]] = {}
    for metodo, folds in resultados.items():
        for fold, valores in folds.items():
            for metrica, valor in valores.items():
                por_metrica.setdefault(metrica, {}).setdefault(metodo, {})[fold] = valor
    return {m: pd.DataFrame(d).sort_index() for m, d in por_metrica.items()}


def resumir(tabela: pd.DataFrame, maior_e_melhor: bool) -> pd.DataFrame:
    """Media, desvio e rank medio por metodo, ordenados do melhor para o pior."""
    completa = tabela.dropna(axis=1, how="any")
    ranks = completa.rank(axis=1, ascending=not maior_e_melhor)
    return pd.DataFrame({
        "media": completa.mean(),
        "desvio": completa.std(),
        "rank_medio": ranks.mean(),
    }).sort_values("rank_medio")


def benjamini_hochberg(valores_p, alfa: float = 0.05) -> np.ndarray:
    """Controla a taxa de falsas descobertas entre varias comparacoes.

    Com 15 metodos sao 105 comparacoes par a par, e ao nivel de 5% esperam-se
    cerca de cinco rejeicoes falsas sem correcao alguma. O procedimento de
    Benjamini e Hochberg (1995) ordena os valores-p e rejeita as hipoteses
    1..i*, com i* o maior indice em que p_(i) <= (i/m) * alfa.
    """
    p = np.asarray(valores_p, dtype=float)
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool)
    ordem = np.argsort(p)
    passou = p[ordem] <= alfa * (np.arange(1, m + 1) / m)
    rejeitadas = np.zeros(m, dtype=bool)
    if passou.any():
        rejeitadas[ordem[: np.max(np.flatnonzero(passou)) + 1]] = True
    return rejeitadas


def comparacoes_pareadas(tabela: pd.DataFrame, alfa: float = 0.05) -> pd.DataFrame:
    """Wilcoxon pareado em todos os pares de metodos, com correcao BH.

    O teste de Wilcoxon e o recomendado por Demsar (2006) para comparar dois
    metodos sobre as mesmas particoes: usa apenas a ordem das diferencas, nao
    pressupoe normalidade e e robusto a um fold atipico.

    Dois metodos com diferenca identicamente nula em todos os folds recebem
    p = 1: o teste nao e definido nesse caso, e tratar como significativo seria
    o erro oposto.
    """
    completa = tabela.dropna(axis=1, how="any")
    metodos = list(completa.columns)
    pares = list(itertools.combinations(metodos, 2))

    linhas = []
    valores_p = []
    for a, b in pares:
        diferenca = completa[a].values - completa[b].values
        if np.allclose(diferenca, 0):
            p = 1.0
        else:
            p = float(wilcoxon(completa[a], completa[b]).pvalue)
        valores_p.append(p)
        linhas.append({
            "metodo_a": a,
            "metodo_b": b,
            "media_a": completa[a].mean(),
            "media_b": completa[b].mean(),
            "diferenca": diferenca.mean(),
            "folds_a_maior": int((diferenca > 0).sum()),
            "p": p,
        })

    rejeitadas = benjamini_hochberg(valores_p, alfa)
    for linha, rejeitada in zip(linhas, rejeitadas):
        linha["significativo_bh"] = bool(rejeitada)
    return pd.DataFrame(linhas)


def friedman(tabela: pd.DataFrame) -> float:
    """Teste omnibus de Friedman. Devolve NaN com menos de tres metodos.

    Serve como porta de entrada: sem evidencia de que ha alguma diferenca no
    conjunto, nao faz sentido sair comparando os pares.
    """
    completa = tabela.dropna(axis=1, how="any")
    if completa.shape[1] < 3:
        return float("nan")
    return float(friedmanchisquare(*[completa[c].values for c in completa.columns]).pvalue)
