"""
Passo 4 — Verificacao independente de que a base contem apenas Homo sapiens.

Faz tres checagens:
  1. Todos os registros da base filtrada sao 'homo sapiens' (apos normalizacao)?
  2. Algum simbolo da nossa base aparece com OUTRA especie no arquivo bruto
     inteiro (incluindo o bloco alem do ID 12881)? Isso indicaria um simbolo
     ambiguo, compartilhado entre especies.
  3. Ha simbolos com cara de gene de camundongo/rato (padrao Xxxx em vez de
     XXXX, sufixos -ps, Rik) que teriam escapado pelo campo Species?

As checagens leem os registros completos gravados pelo passo 01, que vem da
planilha bruta e nao e versionado. Sem eles, a verificacao inteira e pulada; a
checagem 2 tambem precisa da propria planilha em data/bruto/.
"""
import re
import sys

import pandas as pd

from caminhos import PROC, fonte_bruta

FONTE = fonte_bruta(obrigatoria=False)
REGISTROS = PROC / "01_associacoes_humanas.csv"

print("=" * 78)
print("VERIFICACAO DE ESPECIE")
print("=" * 78)

if not REGISTROS.exists():
    print(f"\n{REGISTROS.name} ausente: ele e gerado pelo passo 01 a partir da")
    print("planilha bruta, que nao e distribuida. Verificacao pulada.")
    print("\n" + "=" * 78)
    sys.exit(0)

base = pd.read_csv(REGISTROS)
simbolos = set(base["ncRNA Symbol"].astype(str).str.strip())
print(f"\nbase filtrada: {len(base):,} registros | {len(simbolos):,} simbolos")

# ------------------------------------------------------ 1. especie na base
sp = base["Species"].astype(str).str.strip().str.lower().unique()
print(f"\n[1] valores distintos de Species na base filtrada: {list(sp)}")
print("    OK" if list(sp) == ["homo sapiens"] else "    *** FALHA ***")

# --------------------------------- 2. simbolos ambiguos no arquivo completo
print("\n[2] procurando os mesmos simbolos em outras especies no arquivo bruto...")
if FONTE is None:
    print("    planilha bruta ausente em data/bruto/ -- checagem pulada")
    outras = pd.DataFrame()
else:
    bruto = pd.read_excel(FONTE)
    bruto.columns = [c.strip() for c in bruto.columns]
    bruto["sym"] = bruto["ncRNA Symbol"].astype(str).str.strip()
    bruto["sp"] = bruto["Species"].astype(str).str.strip().str.lower()

    nosso = bruto[bruto["sym"].isin(simbolos)]
    outras = nosso[nosso["sp"] != "homo sapiens"]

    print(f"    registros do arquivo inteiro com nossos simbolos: {len(nosso):,}")
    print(f"    destes, com especie != homo sapiens             : {len(outras):,}")

if FONTE is None:
    pass
elif len(outras):
    resumo = (outras.groupby(["sym", "sp"]).size()
              .reset_index(name="n").sort_values("n", ascending=False))
    print(f"\n    simbolos ambiguos: {resumo['sym'].nunique():,}")
    print("    (o mesmo simbolo aparece em outra especie -- nomes de gene sao")
    print("     frequentemente compartilhados entre ortologos, entao isto e")
    print("     esperado; o que importa e que USAMOS so as linhas humanas)")
    print(resumo.head(15).to_string(index=False))
    resumo.to_csv(PROC / "04_simbolos_em_outras_especies.csv",
                  index=False, encoding="utf-8-sig")
else:
    print("    nenhum simbolo nosso aparece em outra especie")

# ------------------------------- 3. simbolos com aparencia nao-humana
print("\n[3] simbolos com nomenclatura tipica de roedor...")
# Convencao: humano = MALAT1 (tudo maiusculo); camundongo/rato = Malat1
# Riken clones (xxxxRik) e pseudogenes -ps sao marcadores de camundongo.
suspeitos = []
for s in sorted(simbolos):
    if re.search(r"Rik$", s) or re.search(r"-ps\d*$", s):
        suspeitos.append((s, "padrao de camundongo (Rik/-ps)"))
    elif re.fullmatch(r"[A-Z][a-z]+\d*", s):
        suspeitos.append((s, "capitalizacao de roedor (Xxxx)"))

if suspeitos:
    print(f"    {len(suspeitos)} simbolos suspeitos:")
    for s, motivo in suspeitos[:25]:
        n = int((base["ncRNA Symbol"] == s).sum())
        print(f"      {s:22s} {motivo:32s} ({n} registros)")
    print("\n    ATENCAO: verifique se sao variantes de caixa de simbolos humanos")
    print("    (ex.: 'Malat1' e o mesmo gene que 'MALAT1') ou entradas de roedor")
    print("    mal rotuladas na coluna Species.")
    pd.DataFrame(suspeitos, columns=["simbolo", "motivo"]).to_csv(
        PROC / "04_simbolos_suspeitos.csv", index=False, encoding="utf-8-sig")
else:
    print("    nenhum simbolo com nomenclatura de roedor")

print("\n" + "=" * 78)
