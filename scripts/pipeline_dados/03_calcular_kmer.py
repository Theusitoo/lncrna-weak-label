"""
Passo 3 — Calculo das features k-mer (k=1 e k=2) a partir das sequencias.

Gera 20 atributos, no mesmo formato da base atual:
    k1_A, k1_C, k1_G, k1_U                      ->  4 atributos
    k2_AA, k2_AC, ..., k2_UU                    -> 16 atributos

Cada bloco k e normalizado SEPARADAMENTE (as 4 frequencias de k=1 somam 1,
as 16 de k=2 somam 1), reproduzindo a convencao da base existente.

As sequencias vem do NCBI em DNA (A/C/G/T); sao convertidas para RNA (T -> U).

Uso:
    python 03_calcular_kmer.py                  # gera features a partir do cache
    python 03_calcular_kmer.py --regra maior    # usa a isoforma mais longa
"""
import argparse
import json
from itertools import product

import numpy as np
import pandas as pd

from caminhos import PROC

CACHE = PROC / "cache_sequencias.json"
SAIDA = PROC / "03_features_kmer12.csv"

BASES = ["A", "C", "G", "U"]
COLS_K1 = [f"k1_{b}" for b in BASES]
COLS_K2 = [f"k2_{a}{b}" for a, b in product(BASES, BASES)]
COLUNAS = COLS_K1 + COLS_K2


def kmer12(seq: str) -> np.ndarray:
    """Frequencias de k-mer para k=1 e k=2, cada bloco normalizado a parte."""
    s = seq.upper().replace("T", "U")
    # descarta caracteres ambiguos (N, R, Y, ...) que nao sao A/C/G/U
    s = "".join(c for c in s if c in "ACGU")

    if len(s) < 2:
        return np.full(len(COLUNAS), np.nan)

    idx1 = {b: i for i, b in enumerate(BASES)}
    idx2 = {a + b: i for i, (a, b) in enumerate(product(BASES, BASES))}

    c1 = np.zeros(4)
    for ch in s:
        c1[idx1[ch]] += 1

    c2 = np.zeros(16)
    for i in range(len(s) - 1):
        c2[idx2[s[i:i + 2]]] += 1

    return np.concatenate([c1 / c1.sum(), c2 / c2.sum()])


def escolher_isoforma(isoformas: list, regra: str) -> dict:
    """Escolhe o transcrito representativo entre as isoformas do gene.

    canonico -> ncRNA curado (NR_) antes de predito (XR_) e, dentro de cada
                grupo, o menor accession. O menor NR_ e o registro mais antigo,
                que corresponde ao transcrito originalmente descrito. E a regra
                usada na base do artigo.
    maior    -> sequencia mais longa
    menor    -> sequencia mais curta
    """
    if regra == "canonico":
        # a ordenacao textual colocaria NM_ antes de NR_; a chave abaixo garante
        # que curado (NR_) venha antes de predito (XR_) independentemente disso
        return min(isoformas,
                   key=lambda r: (0 if r["acc"].upper().startswith("NR_") else 1,
                                  r["acc"]))
    if regra == "maior":
        return max(isoformas, key=lambda r: r["n"])
    if regra == "menor":
        return min(isoformas, key=lambda r: r["n"])
    raise ValueError(f"regra desconhecida: {regra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regra", default="canonico",
                    choices=["canonico", "maior", "menor"],
                    help="criterio de escolha da isoforma (padrao: canonico)")
    args = ap.parse_args()

    if not CACHE.exists():
        raise SystemExit(f"Cache nao encontrado: {CACHE}\n"
                         f"Rode antes: python 02_baixar_sequencias.py")

    with open(CACHE, encoding="utf-8") as f:
        cache = json.load(f)

    resolvidos = {s: v for s, v in cache.items() if v.get("ok")}
    print(f"sequencias no cache: {len(resolvidos):,}")
    print(f"regra de isoforma  : {args.regra}")

    linhas, indices, descartados, escolhidas = [], [], [], []
    for simbolo, v in sorted(resolvidos.items()):
        iso = escolher_isoforma(v["isoformas"], args.regra)
        vec = kmer12(iso["seq"])
        if np.isnan(vec).any():
            descartados.append(simbolo)
            continue
        linhas.append(vec)
        indices.append(simbolo)
        escolhidas.append({"symbol": simbolo, "accession": iso["acc"],
                           "n_nt": iso["n"], "n_isoformas": len(v["isoformas"]),
                           "estrategia": v.get("estrategia", "E1"),
                           "titulo": iso.get("titulo", "")})

    feats = pd.DataFrame(linhas, index=indices, columns=COLUNAS)
    feats.index.name = "symbol"

    # rastreabilidade: qual transcrito foi usado para cada lncRNA
    pd.DataFrame(escolhidas).to_csv(PROC / "03_transcritos_usados.csv",
                                    index=False, encoding="utf-8-sig")

    print(f"features geradas   : {feats.shape}")
    if descartados:
        print(f"descartados (seq invalida): {len(descartados)} -> {descartados[:5]}")

    # verificacao de sanidade
    s1 = feats[COLS_K1].sum(axis=1)
    s2 = feats[COLS_K2].sum(axis=1)
    print(f"soma k=1: min={s1.min():.6f} max={s1.max():.6f}  (esperado 1.0)")
    print(f"soma k=2: min={s2.min():.6f} max={s2.max():.6f}  (esperado 1.0)")

    feats.to_csv(SAIDA, encoding="utf-8")
    print(f"\nOK: salvo em {SAIDA.name}")


if __name__ == "__main__":
    main()
