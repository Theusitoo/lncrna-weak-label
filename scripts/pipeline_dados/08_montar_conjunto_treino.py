"""
Passo 8 — Monta o conjunto de treino no formato esperado pelo pipeline WELL.

Combina:
    features k-mer (20 atributos)   03_features_kmer12.csv
    rotulos por sistema (14)        08_sistema_lncRNA_x_grupos_K14.csv

e grava em data/<dataset>/ os tres arquivos que o pacote espera:
    features.csv         instancias x 20 atributos
    labels.csv           instancias x 14 rotulos
    features_splits.csv  indices de treino/teste dos 10 folds

Os splits usam MultilabelStratifiedKFold(n_splits=10, shuffle=True,
random_state=0) -- mesma configuracao do trabalho anterior, para que a
comparacao entre metodos nao seja afetada por mudanca no protocolo.

Uso:
    python 08_montar_conjunto_treino.py
"""
import argparse

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from caminhos import BASES, PROC, RAIZ

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="lncRNA_sis14",
                help="nome da pasta em data/")
ap.add_argument("--rotulos", default="08_sistema_lncRNA_x_grupos_K14.csv",
                help="matriz de rotulos em data/processado/")
args = ap.parse_args()

print("=" * 78)
print("MONTAGEM DO CONJUNTO DE TREINO")
print("=" * 78)

Y = pd.read_csv(PROC / args.rotulos, index_col=0)
X = pd.read_csv(PROC / "03_features_kmer12.csv", index_col=0)
print(f"\nrotulos : {Y.shape}")
print(f"features: {X.shape}")

comuns = sorted(set(Y.index) & set(X.index))
Y = Y.loc[comuns]
X = X.loc[comuns]
print(f"instancias em comum: {len(comuns):,}")

# instancia sem rotulo nao treina nada
vazias = Y.index[Y.sum(axis=1) == 0]
if len(vazias):
    Y = Y.drop(index=vazias)
    X = X.drop(index=vazias)
    print(f"removidas {len(vazias)} instancias sem rotulo")

assert list(X.index) == list(Y.index), "indices de features e rotulos divergem"

# ------------------------------------------------------------------ splits
mskf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=0)
colunas, dados = [], []
for i, (tr, te) in enumerate(mskf.split(X.values, Y.values), 1):
    colunas += [f"train_fold{i}", f"test_fold{i}"]
    dados += [pd.Series(tr), pd.Series(te)]
splits = pd.concat(dados, axis=1)
splits.columns = colunas

# ------------------------------------------------------------------ gravacao
destino = BASES / args.dataset
destino.mkdir(parents=True, exist_ok=True)

X.to_csv(destino / "features.csv")
Y.to_csv(destino / "labels.csv")
splits.to_csv(destino / "features_splits.csv", index=False)

print(f"\ngravado em {destino.relative_to(RAIZ)}/")
print(f"  features.csv        {X.shape}")
print(f"  labels.csv          {Y.shape}")
print(f"  features_splits.csv {splits.shape}")

# ------------------------------------------------------------- verificacao
print("\nVERIFICACAO")
tamanhos = [int(splits[f'test_fold{i}'].notna().sum()) for i in range(1, 11)]
todos = np.concatenate([splits[f"test_fold{i}"].dropna().astype(int).values
                        for i in range(1, 11)])
print(f"  tamanhos dos folds de teste: {tamanhos}  (soma={sum(tamanhos)})")
print(f"  cobre todas as instancias  : {sorted(todos) == list(range(len(X)))}")
print(f"  sem repeticao entre folds  : {len(todos) == len(set(todos))}")

pos = Y.sum(axis=0)
print(f"  positivos por rotulo       : min={int(pos.min())} max={int(pos.max())}")
print(f"  densidade                  : {Y.values.mean()*100:.2f}%")
print(f"  cardinalidade media        : {Y.sum(axis=1).mean():.2f}")

# menor numero de positivos que um rotulo tera num fold de treino
pior = min(Y.values[splits[f"train_fold{i}"].dropna().astype(int).values].sum(axis=0).min()
           for i in range(1, 11))
print(f"  min positivos no pior fold de treino: {int(pior)}")

print("\n" + "=" * 78)
print(f"Para treinar:  python -m scripts.rodar --dados data/{args.dataset}")
print("=" * 78)
