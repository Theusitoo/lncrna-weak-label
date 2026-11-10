"""
Passo 5 — BASE 1: matriz lncRNA x doencas individuais.

Antes de montar a matriz, resolve dois problemas achados na verificacao:

  (a) VARIANTES DE NOME: o mesmo gene aparece como entidades separadas
      ('MALAT1', 'MALAT-1', 'Malat1', 'LncRNA MALAT1'). Isso fragmenta as
      associacoes e, pior, criaria instancias duplicadas com features
      identicas e rotulos diferentes -- vazamento entre folds da CV.

      Regra de fusao:
        - normaliza tracos Unicode (U+2010, U+2011...) para '-' ASCII
        - ignora caixa, hifen, underscore e espaco
        - remove o prefixo descritivo 'lncRNA'/'lincRNA'
      O simbolo representativo do grupo e a variante com mais associacoes.

      NOTA: a remocao do prefixo funde 'lincRNA-p21' com 'p21'. Fora de contexto
      'p21' designaria a proteina CDKN1A, mas numa base de associacoes
      lncRNA-doenca a entrada 'p21' se refere ao lincRNA-p21. Confira em
      06_base1_fusoes.csv se concorda.

  (b) CONTAMINACAO DE ESPECIE: identificadores de camundongo rotulados como
      humanos na fonte (Gm####, ####Rik) e lncRNAs conhecidamente de roedor.

  (c) ENTRADAS QUE NAO SAO UM GENE: rotulos que nomeiam dois genes
      ('CCHE1 and TCF21'), condicoes experimentais, microRNAs e identificadores
      de montagem (TCONS_*) que nao existem no RefSeq. Sem sequencia unica,
      nao podem virar instancia.

Saidas:
    06_base1_lncRNA_x_doencas.csv     matriz binaria (lncRNA x doenca)
    06_base1_fusoes.csv               variantes fundidas (AUDITAR)
    06_base1_removidos.csv            simbolos descartados e motivo
"""
import re

import pandas as pd

from caminhos import PROC

# Identificadores de roedor mal rotulados como humanos na fonte, e rotulos
# nao resolviveis a um gene (usados apenas dentro de um artigo especifico).
NAO_HUMANOS = {
    "1700020I14Rik", "Gm15290", "Gm4419", "Gm44593", "Gm47283",   # camundongo
    "Aerrie", "Blnc1", "Chaer", "Hmrhl", "Nespas", "Nostrill",    # lncRNA de roedor
}
NAO_RESOLVIVEIS = {
    "Lnc408", "Lnc5150", "Lnc712", "Linc1749808", "Linc8087", "Linc8986", "Loc490",
}


# Entradas que nao designam UM gene e nao podem virar instancia.
# Cada instancia tem uma unica sequencia; um rotulo que nomeia dois genes
# ('CCHE1 and TCF21') ou uma condicao experimental nao tem sequencia definida.
PADROES_INVALIDOS = [
    (r"\band\b|,|/|&", "nomeia mais de um gene"),
    (r"^combination\b", "condicao experimental, nao um gene"),
    (r"^mir-|^hsa-mir", "microRNA, nao lncRNA"),
    (r"^( enst|nr )\s*\d|^tcons_|^\d{4,}-\d", "identificador de transcrito, nao simbolo"),
    (r"^\d+[A-Za-z]?\d*\.\d+$|^\d+\s*[A-Z]$", "identificador de clone"),
    (r"\b(short rna|long noncoding rna)\b", "descricao, nao simbolo"),
    (r"^lncrna\s*(loc|w\d|\d+)$", "identificador incompleto"),
]

# Entradas da coluna de doenca que nao sao doencas humanas. Removidas aqui, na
# base 1, para que as duas bases fiquem consistentes: se o filtro ficasse so no
# agrupamento, instancias cuja unica anotacao fosse uma dessas sobrariam com a
# linha inteira zerada na base 2.
NAO_DOENCAS = {
    "RNA, Long Noncoding", "Molecular Epidemiology", "Chlorophyll A",
    "Epithelial-Mesenchymal Transition", "Aging", "Prostate", "Urinary Bladder",
    "Breast Diseases", "Lung Diseases", "Wounds and Injuries", "Sclerosis",
    "Fusarium graminearum", "Peste-des-Petits-Ruminants",
    "Duck Hepatitis A Virus Type 1 (Dhav-1)", "Mammary Neoplasms, Animal",
    "心功能障碍",
}

# Entradas por extenso que correspondem a um simbolo conhecido.
SINONIMOS = {
    "Plasmacytoma variant translocation 1": "PVT1",
    "HOXD antisense growth-associated long noncoding RNA (lncRNA) (HAGLR)": "HAGLR",
    "FOXO3A-induced LINC00926": "LINC00926",
    "m6A-induced LINC00958": "LINC00958",
    "lncRNA 00152": "LINC00152",
    "lncRNA 00312": "LINC00312",
    "CCND2 AS1": "CCND2-AS1",
}


def salvar(df, caminho, **kwargs):
    """Grava o CSV; avisa em vez de abortar se o arquivo estiver aberto."""
    try:
        df.to_csv(caminho, encoding="utf-8-sig", **kwargs)
        return True
    except PermissionError:
        print(f"  AVISO: {caminho.name} esta aberto em outro programa; nao foi gravado.")
        return False


def motivo_invalido(simbolo: str):
    s = str(simbolo).strip().lower()
    for padrao, motivo in PADROES_INVALIDOS:
        if re.search(padrao, s):
            return motivo
    return None


def chave_canonica(simbolo: str) -> str:
    """Chave usada para agrupar variantes de nome do mesmo gene."""
    s = SINONIMOS.get(str(simbolo).strip(), str(simbolo).strip())
    # a fonte usa varios tracos Unicode (U+2010, U+2011, U+2012...) que nao
    # casariam com um '-' ASCII; padroniza tudo antes de comparar
    s = re.sub(r"[‐-―−]", "-", s)
    # prefixo 'lncRNA'/'lincRNA' e descritivo, nao faz parte do simbolo
    m = re.match(r"^(?:lnc|linc)RNA[\s_-]*(.+)$", s, flags=re.I)
    if m:
        s = m.group(1)
    return re.sub(r"[\s_-]", "", s).upper()


print("=" * 78)
print("BASE 1 - lncRNA x DOENCAS INDIVIDUAIS")
print("=" * 78)

pares = pd.read_csv(PROC / "01_pares_unicos.csv")
pares.columns = ["simbolo", "doenca"]
print(f"\npares de entrada          : {len(pares):,}")
print(f"  simbolos distintos      : {pares['simbolo'].nunique():,}")
print(f"  doencas distintas       : {pares['doenca'].nunique():,}")

# --------------------------------------------------- (b) remocao de nao-humanos
invalidos = {s: motivo_invalido(s) for s in pares["simbolo"].unique()}
invalidos = {s: mot for s, mot in invalidos.items() if mot}
if invalidos:
    print(f"\nentradas que nao designam um gene unico: {len(invalidos)}")
    for mot in sorted(set(invalidos.values())):
        exemplos = [s for s, m in invalidos.items() if m == mot][:3]
        print(f"    {mot:42s} {sum(1 for m in invalidos.values() if m == mot):3d}"
              f"  ex.: {exemplos}")

descartar = NAO_HUMANOS | NAO_RESOLVIVEIS | set(invalidos)
removidos = pares[pares["simbolo"].isin(descartar)]
if len(removidos):
    reg = (removidos.groupby("simbolo").size().reset_index(name="n_doencas"))
    reg["motivo"] = reg["simbolo"].map(
        lambda s: "roedor rotulado como humano" if s in NAO_HUMANOS
        else invalidos.get(s, "identificador nao resolvivel a um gene"))
    salvar(reg, PROC / "06_base1_removidos.csv", index=False)
    print(f"\nremovidos {reg['simbolo'].nunique()} simbolos "
          f"({len(removidos)} pares) -> 06_base1_removidos.csv")
pares = pares[~pares["simbolo"].isin(descartar)].copy()

# --------------------------------------------------- (a) fusao de variantes
pares["chave"] = pares["simbolo"].map(chave_canonica)
freq = pares.groupby("simbolo").size()
# representante = variante com mais associacoes (desempate alfabetico estavel)
rep = (pares.groupby("chave")["simbolo"]
       .apply(lambda ss: sorted(set(ss), key=lambda s: (-freq[s], s))[0]))
pares["simbolo_canonico"] = pares["chave"].map(rep)

grupos = pares.groupby("chave")["simbolo"].nunique()
fundidos = grupos[grupos > 1]
if len(fundidos):
    linhas = []
    for k in fundidos.index:
        # o desempate pelo proprio nome torna a listagem identica entre
        # execucoes; sem ele, variantes com a mesma frequencia saem na ordem de
        # iteracao do set, que o Python sorteia a cada processo
        vs = sorted(set(pares.loc[pares["chave"] == k, "simbolo"]),
                    key=lambda s: (-freq[s], s))
        n_antes = freq[vs[0]]
        n_depois = int((pares["chave"] == k).sum())
        linhas.append({"representante": rep[k], "variantes": " | ".join(vs),
                       "n_variantes": len(vs),
                       "doencas_antes": int(n_antes), "doencas_depois": n_depois,
                       "ganho": n_depois - int(n_antes)})
    df_f = pd.DataFrame(linhas).sort_values("ganho", ascending=False)
    salvar(df_f, PROC / "06_base1_fusoes.csv", index=False)
    print(f"\nvariantes fundidas: {len(fundidos)} grupos "
          f"({int(pares.loc[pares['chave'].isin(fundidos.index), 'simbolo'].nunique())} simbolos)")
    print("  -> 06_base1_fusoes.csv  (AUDITE antes de publicar)")
    print("\n  maiores ganhos de associacoes:")
    for _, r in df_f.head(8).iterrows():
        print(f"    {r['representante']:12s} {r['doencas_antes']:3d} -> "
              f"{r['doencas_depois']:3d} doencas  ({r['n_variantes']} variantes)")

# --------------------------------------------------------- matriz binaria
pares_final = (pares[["simbolo_canonico", "doenca"]]
               .drop_duplicates()
               .rename(columns={"simbolo_canonico": "simbolo"}))

matriz = (pd.crosstab(pares_final["simbolo"], pares_final["doenca"])
          .astype("int8"))
matriz.index.name = "ncRNA_Symbol"

# descarta colunas que nao sao doencas e, em seguida, instancias que ficaram
# sem nenhuma anotacao (a unica que tinham era uma dessas colunas)
nao_doenca = [c for c in matriz.columns if c in NAO_DOENCAS]
if nao_doenca:
    matriz = matriz.drop(columns=nao_doenca)
    print(f"\nremovidas {len(nao_doenca)} colunas que nao sao doencas humanas")
sem_rotulo = matriz.index[matriz.sum(axis=1) == 0]
if len(sem_rotulo):
    matriz = matriz.drop(index=sem_rotulo)
    print(f"removidas {len(sem_rotulo)} instancias que ficaram sem rotulo: "
          f"{list(sem_rotulo[:6])}")

print("\n" + "-" * 78)
print(f"MATRIZ FINAL: {matriz.shape[0]:,} lncRNAs x {matriz.shape[1]:,} doencas")
print(f"  associacoes (positivos) : {int(matriz.values.sum()):,}")
print(f"  densidade               : {matriz.values.mean() * 100:.3f}%")

dpl = matriz.sum(axis=1)
print(f"\n  doencas por lncRNA: min={dpl.min()} mediana={dpl.median():.0f} "
      f"media={dpl.mean():.1f} max={dpl.max()}")
for lim in [1, 2, 3, 5, 10, 20]:
    print(f"    com >= {lim:2d} doencas: {int((dpl >= lim).sum()):6,}")

lpd = matriz.sum(axis=0)
print(f"\n  lncRNAs por doenca: min={lpd.min()} mediana={lpd.median():.0f} "
      f"media={lpd.mean():.1f} max={lpd.max()}")
for lim in [1, 5, 10, 30]:
    print(f"    com >= {lim:2d} lncRNAs: {int((lpd >= lim).sum()):5,}")

saida = PROC / "06_base1_lncRNA_x_doencas.csv"
salvar(matriz, saida)
print(f"\nOK: {saida.name}")
print("=" * 78)
