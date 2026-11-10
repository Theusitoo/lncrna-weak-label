"""
Passo 7 — Verificacao de integridade de TODOS os arquivos gerados.

Checa, para cada artefato:
  - matrizes de rotulo contem apenas 0 e 1, sem NaN
  - features k-mer sao frequencias validas (somam 1 por bloco, sem NaN)
  - indices sao simbolos de gene plausiveis (sem entradas compostas ou textuais)
  - indices estao alinhados entre as bases
  - nao ha simbolos duplicados

Sai com codigo != 0 se qualquer checagem falhar.
"""
import re
import sys

import numpy as np
import pandas as pd

from caminhos import BASES, PROC

falhas = []


def checar(condicao, descricao, detalhe=""):
    marca = "OK  " if condicao else "FALHA"
    print(f"  [{marca}] {descricao}" + (f"  -- {detalhe}" if detalhe and not condicao else ""))
    if not condicao:
        falhas.append(descricao)
    return condicao


def checar_matriz_binaria(caminho, nome):
    print(f"\n{nome}  ({caminho.name})")
    if not caminho.exists():
        checar(False, "arquivo existe")
        return None
    df = pd.read_csv(caminho, index_col=0)
    vals = set(np.unique(df.values).tolist())

    checar(vals <= {0, 1}, "matriz contem apenas 0 e 1", f"encontrados: {sorted(vals)[:8]}")
    checar(not df.isna().any().any(), "sem valores ausentes")
    checar(df.index.is_unique, "sem simbolos duplicados",
           f"{int(df.index.duplicated().sum())} duplicados")
    checar(not df.columns.duplicated().any(), "sem colunas duplicadas")
    checar((df.sum(axis=1) > 0).all(), "toda instancia tem ao menos um rotulo",
           f"{int((df.sum(axis=1) == 0).sum())} instancias sem rotulo")
    checar((df.sum(axis=0) > 0).all(), "todo rotulo tem ao menos um positivo",
           f"{int((df.sum(axis=0) == 0).sum())} rotulos vazios")

    # simbolos plausiveis
    idx = pd.Series(df.index.astype(str))
    ruim = idx[idx.str.contains(r"\band\b|,|/|&|\(", case=False, regex=True)]
    checar(len(ruim) == 0, "nenhum simbolo composto ou textual",
           f"{len(ruim)}: {list(ruim[:4])}")
    print(f"         shape={df.shape}  positivos={int(df.values.sum()):,}  "
          f"densidade={df.values.mean()*100:.3f}%")
    return df


print("=" * 78)
print("VERIFICACAO DE INTEGRIDADE")
print("=" * 78)

b1 = checar_matriz_binaria(PROC / "06_base1_lncRNA_x_doencas.csv",
                           "BASE 1 - doencas individuais")

# Base do artigo: agrupamento por sistema fisiologico. Por construcao, e um
# subconjunto da base 1 (lncRNAs com pelo menos tres doencas e com sequencia
# recuperada).
sistemas = {}
for f in sorted(PROC.glob("08_sistema_lncRNA_x_grupos_K*.csv")):
    sistemas[f.stem.rsplit("_", 1)[-1]] = checar_matriz_binaria(
        f, f"SISTEMAS - {f.stem.rsplit('_', 1)[-1]}")

# ------------------------------------------------------- alinhamento de indices
print("\nALINHAMENTO ENTRE AS BASES")
for k, s in sistemas.items():
    if b1 is not None and s is not None:
        fora = set(s.index) - set(b1.index)
        checar(not fora, f"todo lncRNA da base por sistema ({k}) esta na base 1",
               f"{len(fora)} ausentes: {sorted(fora)[:4]}")

# ------------------------------------------------------------------- features
print("\nFEATURES K-MER  (03_features_kmer12.csv)")
fk = PROC / "03_features_kmer12.csv"
if fk.exists():
    X = pd.read_csv(fk, index_col=0)
    k1 = [c for c in X.columns if c.startswith("k1_")]
    k2 = [c for c in X.columns if c.startswith("k2_")]
    checar(len(k1) == 4 and len(k2) == 16, "20 atributos (4 de k=1, 16 de k=2)",
           f"k1={len(k1)} k2={len(k2)}")
    checar(not X.isna().any().any(), "sem valores ausentes")
    checar(bool(((X.values >= 0) & (X.values <= 1)).all()), "todos os valores em [0,1]")
    checar(bool(np.allclose(X[k1].sum(axis=1), 1.0)), "frequencias de k=1 somam 1")
    checar(bool(np.allclose(X[k2].sum(axis=1), 1.0)), "frequencias de k=2 somam 1")
    checar(X.index.is_unique, "sem simbolos duplicados")
    if b1 is not None:
        cob = len(set(X.index) & set(b1.index))
        print(f"         shape={X.shape}  cobertura da base 1: {cob:,} de {len(b1):,}")
else:
    print("  (ainda nao gerado)")

# ------------------------------------------------- bases prontas para treino
print("\nBASES PRONTAS (data/)")
bases_prontas = sorted(BASES.glob("*/features_splits.csv")) if BASES.exists() else []
if not bases_prontas:
    print("  (nenhuma ainda; o passo 08 as gera)")
for f in bases_prontas:
    nome = f.parent.name
    sp = pd.read_csv(f)
    n = len(pd.read_csv(f.parent / "labels.csv", index_col=0))
    testes = [sp[c].dropna().astype(int) for c in sp.columns
              if c.startswith("test_fold")]
    todos = np.concatenate([t.values for t in testes]) if testes else np.array([])
    checar(len(todos) == len(np.unique(todos)),
           f"{nome}: nenhuma instancia repetida entre folds de teste")
    checar(sorted(todos.tolist()) == list(range(n)),
           f"{nome}: toda instancia e testada exatamente uma vez",
           f"{len(todos)} testes para {n} instancias")
    print(f"         {nome}: {len(testes)} folds, {len(todos)} instancias de teste")

print("\n" + "=" * 78)
if falhas:
    print(f"RESULTADO: {len(falhas)} FALHA(S)")
    for f in falhas:
        print(f"   - {f}")
    sys.exit(1)
print("RESULTADO: todas as checagens passaram")
print("=" * 78)
