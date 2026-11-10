"""
Passo 1 — Filtragem da base bruta de associacoes lncRNA-doenca.

Regras aplicadas:
  1. Usar apenas os registros ate o ID 12881 (bloco 'LncRNA' validado experimentalmente).
     Verificado: linhas 0..12879 do pandas == 12.880 registros da categoria 'LncRNA'.
     A partir da linha 12880 comeca outro bloco ('lncRNA' minusculo, inicia em Mus musculus).
  2. Manter apenas Species == Homo sapiens (normalizando maiusculas/minusculas e espacos).

Entrada: data/bruto/<planilha da LncRNADisease v3.0>.xlsx
Saida:   data/processado/01_associacoes_humanas.csv
         data/processado/01_pares_unicos.csv
"""
import pandas as pd
import numpy as np

from caminhos import PROC as SAIDA_DIR, fonte_bruta

FONTE = fonte_bruta()

CORTE_ID = 12881  # ID do ultimo registro do bloco 'LncRNA' (1-based, cabecalho = linha 1)

print("=" * 78)
print("PASSO 1 - FILTRAGEM DA BASE BRUTA")
print("=" * 78)

print(f"\nLendo {FONTE.name} ...")
df = pd.read_excel(FONTE)
df.columns = [c.strip() for c in df.columns]
print(f"  registros brutos: {len(df):,}")

# ---------------------------------------------------------------- corte por ID
# Excel: linha 1 = cabecalho, logo o "ID 12881" corresponde ao indice 12879 do
# pandas. O slice [0:12880] cobre ate esse registro, inclusive.
df = df.iloc[:CORTE_ID - 1].copy()
print(f"\nApos corte no ID {CORTE_ID}: {len(df):,} registros")
print(f"  categorias: {df['ncRNA Category'].value_counts().to_dict()}")

# ------------------------------------------------------- filtro Homo sapiens
sp_norm = df["Species"].astype(str).str.strip().str.lower()

# registros de especie mista sao reportados e descartados
mistos = df[sp_norm.str.contains("homo sapiens") & (sp_norm != "homo sapiens")]
if len(mistos):
    print(f"\n  registros de especie mista descartados: {len(mistos)}")
    for v in mistos["Species"].unique():
        print(f"     - {v}")

df = df[sp_norm == "homo sapiens"].copy()
print(f"\nApos filtro Homo sapiens: {len(df):,} registros")

# ----------------------------------------------------------- normalizacao
df["ncRNA Symbol"] = df["ncRNA Symbol"].astype(str).str.strip()
df["Disease Name"] = df["Disease Name"].astype(str).str.strip()

# remove linhas sem simbolo ou sem doenca
antes = len(df)
df = df[(df["ncRNA Symbol"] != "") & (df["ncRNA Symbol"].str.lower() != "nan")]
df = df[(df["Disease Name"] != "") & (df["Disease Name"].str.lower() != "nan")]
if len(df) < antes:
    print(f"  removidos {antes - len(df)} registros sem simbolo ou sem doenca")

# ------------------------------------------------------------ caracterizacao
print("\n" + "=" * 78)
print("CARACTERIZACAO DA BASE FILTRADA")
print("=" * 78)

n_lnc = df["ncRNA Symbol"].nunique()
n_dis = df["Disease Name"].nunique()
pares = df[["ncRNA Symbol", "Disease Name"]].drop_duplicates()

print(f"\n  registros            : {len(df):,}")
print(f"  lncRNAs unicos       : {n_lnc:,}")
print(f"  doencas unicas       : {n_dis:,}")
print(f"  pares unicos         : {len(pares):,}")
print(f"  densidade da matriz  : {len(pares) / (n_lnc * n_dis) * 100:.2f}%")

dpl = pares.groupby("ncRNA Symbol").size()
print(f"\n  --- doencas por lncRNA ---")
print(f"  min={dpl.min()}  mediana={dpl.median():.0f}  media={dpl.mean():.1f}  max={dpl.max()}")
for lim in [1, 2, 3, 5, 10, 20]:
    print(f"    lncRNAs com >= {lim:2d} doencas: {int((dpl >= lim).sum()):6,}")

lpd = pares.groupby("Disease Name").size()
print(f"\n  --- lncRNAs por doenca ---")
print(f"  min={lpd.min()}  mediana={lpd.median():.0f}  media={lpd.mean():.1f}  max={lpd.max()}")
for lim in [1, 3, 5, 10, 30]:
    print(f"    doencas com >= {lim:2d} lncRNAs: {int((lpd >= lim).sum()):5,}")

# Nota: agregar as 443 doencas em K categorias so pode MANTER ou REDUZIR o numero
# de rotulos positivos por lncRNA (varias doencas do mesmo grupo colapsam em 1).
print(f"\n  --- 15 doencas mais frequentes ---")
for d, c in lpd.sort_values(ascending=False).head(15).items():
    print(f"    {d[:52]:54s} {c:5,}")

# ------------------------------------------------------------------- salvar
saida = SAIDA_DIR / "01_associacoes_humanas.csv"
df.to_csv(saida, index=False, encoding="utf-8-sig")
print(f"\n  OK: salvo em {saida}")

saida_pares = SAIDA_DIR / "01_pares_unicos.csv"
pares.to_csv(saida_pares, index=False, encoding="utf-8-sig")
print(f"  OK: pares unicos em {saida_pares.name}")

print("\n" + "=" * 78)
