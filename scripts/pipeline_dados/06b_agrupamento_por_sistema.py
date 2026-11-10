"""
Passo 6b — agrupamento das doencas por SISTEMA FISIOLOGICO.

POR QUE ESTE EIXO
-----------------
Uma primeira versao do agrupamento organizava os rotulos no eixo neoplasico /
nao-neoplasico. O efeito colateral era que o cancer ganhava 11 rotulos
detalhados por orgao enquanto todo o resto virava um punhado de grupos pequenos
e um coletor 'Outras_Nao_Cancer' -- o que convertia o estudo, na pratica, num
estudo sobre cancer.

Aqui o eixo e o SISTEMA FISIOLOGICO. Neoplasias e nao-neoplasias do mesmo orgao
ficam juntas: 'Carcinoma Hepatocelular' e 'Cirrose Hepatica' sao ambas doencas
do sistema digestivo. Assim nenhum rotulo e definido por 'nao ser cancer', e o
objeto declarado -- lncRNA associado a DOENCAS -- fica representado de forma
simetrica.

Saidas:
    08_sistema_lncRNA_x_grupos_K{K}.csv
    08_sistema_mapa_doenca_grupo_K{K}.csv
"""
import argparse
import re

import pandas as pd

from caminhos import PROC

MIN_POSITIVOS = 30

NAO_DOENCAS = {
    "RNA, Long Noncoding", "Molecular Epidemiology", "Chlorophyll A",
    "Epithelial-Mesenchymal Transition", "Aging", "Prostate", "Urinary Bladder",
    "Breast Diseases", "Lung Diseases", "Wounds and Injuries", "Sclerosis",
    "Fusarium graminearum", "Peste-des-Petits-Ruminants",
    "Duck Hepatitis A Virus Type 1 (Dhav-1)", "Mammary Neoplasms, Animal",
    "心功能障碍",
}

# ---------------------------------------------------------------------------
# Taxonomia por sistema. A ORDEM IMPORTA: a primeira regra que casar vence.
# Regras mais especificas primeiro.
# ---------------------------------------------------------------------------
REGRAS = [
    # complicacoes do diabetes acompanham o sistema endocrino, qualquer que
    # seja o tecido atingido (nefropatia, retinopatia, macroangiopatia)
    ("Endocrino_Metabolico",  r"^diabetic |diabetes|thyroid|adrenal|adrenocortical|"
                              r"parathyroid|"
                              r"neuroendocrin|obesit|metabolic|insulin|dyslipid|"
                              r"cholesterol|hypothyroid|graves|prolactinoma|"
                              r"pituitary|growth hormone|klinefelter"),
    ("Oftalmologico",         r"pterygium|retinopath|glaucom|cataract|macular|"
                              r"corneal|uveitis|myopia|ocular|retinal|retinoblastom|"
                              r"vitreoretinopath|capsule opacification|vogt-koyanagi"),
    ("Digestivo",             r"esophag|oesophag|stomach|gastric|colorect|colonic|"
                              r"colon |rectal|rectum|bowel|hepatocellular|"
                              r"liver|hepatic|hepatoblastom|cholangiocarcinom|"
                              r"bile duct|gallbladder|pancrea|cirrhos|peptic ulcer|"
                              r"irritable bowel|hirschsprung|celiac|steatohepatit|"
                              r"crohn|colitis|digestive|intestinal|nonalcoholic|"
                              r"fatty liver|esophagus"),
    ("Respiratorio",          r"lung|pulmonary|asthma|copd|obstructive|respiratory|"
                              r"rhinitis|sinusitis|bronchopulmonary|silicosis|"
                              r"sleep apnea|small-cell|mesotheliom|pneumoni"),
    ("Ginecologico_Reprodutivo",
                              r"cervic|ovarian|endometri|uterine|vulva|vagin|"
                              r"choriocarcinom|trophoblast|polycystic ovary|"
                              r"pre-eclampsia|preeclampsia|abortion|pregnan|"
                              r"preterm birth|leiomyoma|testicular|prostatitis|"
                              r"prostatic hyperplas"),
    ("Renal_Urologico",       r"prostat|renal cell|kidney|bladder|urothelial|"
                              r"urinary|wilms|nephropath|nephrit|glomerul|"
                              r"urolith|cystitis|nephrotic|renal"),
    ("Mama",                  r"breast|mammary"),
    ("Neurologico",           r"gliom|glioblastom|neuroblastom|medulloblastom|"
                              r"astrocytom|meningiom|ependymom|brain|nervous system|"
                              r"alzheimer|parkinson|depress|epilep|stroke|"
                              r"multiple sclerosis|neuropath|dementia|schizophren|"
                              r"autism|amyotroph|huntington|cerebral|migrain|"
                              r"myasthenia|bipolar|spinal cord|neuronal|"
                              r"demyelinating|neurom|neurosyphilis"),
    ("Cardiovascular",        r"atrial fibrillat|coronary|atheroscleros|cardiomyopath|"
                              r"myocardial|heart|cardiac|hypertens|aneurysm|"
                              r"vascular|angina|arrhythm|aort|thrombos|"
                              r"hemangiom|cardiomegal|tetralogy"),
    ("Hematologico",          r"leukemi|leukaem|lymphom|myelom|myelodysplas|"
                              r"hematolog|haematolog|thalassemi|hemophili|"
                              r"purpura|thrombocytopen|anemia|coagul|hodgkin"),
    ("Musculoesqueletico",    r"osteosarcom|sarcom|chordom|bone|chondrosarcom|"
                              r"rhabdomyosarcom|fibrosarcom|osteoarthrit|"
                              r"intervertebral disc|osteonecros|fracture|"
                              r"degenerative disc|muscular dystroph|ankylosing|"
                              r"spondylit|osteoporos|femur head"),
    ("Cabeca_Pescoco",        r"head and neck|nasopharyng|tongue|oral |laryng|"
                              r"hypopharyng|oropharyng|salivary|mouth|parotid|"
                              r"tonsil|periodont|pulpit|stomatitis|leukoplakia|"
                              r"ameloblastom|maxillary|sinonasal"),
    ("Dermatologico",         r"melanom|skin|cutaneous|basal cell carcinom|keloid|"
                              r"vitiligo|rosacea|ichthyos|pemphigus|cicatrix|"
                              r"alopecia|dermatit|psorias|mycosis fungoides"),
    ("Infeccioso",            r"covid|sars|sepsis|hepatitis|tuberculos|infect|"
                              r"hiv|influenza|viral|bacteri|chlamydia|escherichia|"
                              r"mycobacterium|mycoplasma|chikungunya|meningitis"),
    ("Inflamatorio_Autoimune",
                              r"arthrit|lupus|sjogren|autoimmun|rheumat|"
                              r"inflammation|gout|behcet|dermatomyosit|"
                              r"scleroderma|mucocutaneous lymph"),
]


def classificar(nome):
    if nome in NAO_DOENCAS:
        return "DESCARTAR"
    n = nome.lower()
    for categoria, padrao in REGRAS:
        if re.search(padrao, n):
            return categoria
    return "NAO_CLASSIFICADA"


ap = argparse.ArgumentParser()
ap.add_argument("--min-doencas", type=int, default=3,
                help="corte de doencas por lncRNA para o conjunto de treino")
args = ap.parse_args()

print("=" * 78)
print("AGRUPAMENTO POR SISTEMA FISIOLOGICO")
print("=" * 78)

m = pd.read_csv(PROC / "06_base1_lncRNA_x_doencas.csv", index_col=0)
X = pd.read_csv(PROC / "03_features_kmer12.csv", index_col=0)

cat = {d: classificar(d) for d in m.columns}
desc = [d for d, c in cat.items() if c == "DESCARTAR"]
if desc:
    m = m.drop(columns=desc)
    cat = {d: c for d, c in cat.items() if c != "DESCARTAR"}

nc = [d for d, c in cat.items() if c == "NAO_CLASSIFICADA"]
lpd = m.sum(axis=0)
print(f"\ncobertura: {m.shape[1]-len(nc):,} de {m.shape[1]:,} doencas "
      f"({(m.shape[1]-len(nc))/m.shape[1]*100:.1f}%)")
print(f"  nao classificadas: {len(nc)} doencas, "
      f"{int(sum(lpd[d] for d in nc)):,} associacoes "
      f"({sum(lpd[d] for d in nc)/lpd.sum()*100:.1f}%)")
if nc:
    top = sorted(nc, key=lambda d: -lpd[d])[:6]
    print(f"  maiores: {', '.join(f'{d} ({int(lpd[d])})' for d in top)}")

# ------------------------------------------------- conjunto de treino
n_doencas = m.sum(axis=1)
idx = sorted(set(n_doencas[n_doencas >= args.min_doencas].index) & set(X.index))
print(f"\nconjunto de treino (>= {args.min_doencas} doencas e com features): "
      f"{len(idx):,} instancias")

sistemas = sorted({c for c in cat.values() if c != "NAO_CLASSIFICADA"})
G = pd.DataFrame(index=m.index)
for s in sistemas:
    ds = [d for d in m.columns if cat[d] == s]
    G[s] = (m[ds].sum(axis=1) > 0).astype("int8")

sub = G.loc[idx]
pos = sub.sum(axis=0).sort_values()

print(f"\n{'sistema':28s} {'positivos':>10s} {'doencas':>9s}  composicao")
print("-" * 78)
for s in pos.index:
    ds = sorted([d for d in m.columns if cat[d] == s], key=lambda d: -lpd[d])
    n_neo = sum(1 for d in ds if re.search(r"neoplas|carcinom|cancer|sarcom|"
                                           r"melanom|leukemi|lymphom|myelom|gliom|"
                                           r"blastom|tumor", d.lower()))
    marca = "  <-- abaixo de 30" if pos[s] < MIN_POSITIVOS else ""
    print(f"{s:28s} {int(pos[s]):10d} {len(ds):9d}  "
          f"{n_neo} neoplasicas / {len(ds)-n_neo} nao{marca}")

viaveis = [s for s in pos.index if pos[s] >= MIN_POSITIVOS]
Gf = sub[viaveis]
Gf = Gf[Gf.sum(axis=1) > 0]

print("-" * 78)
print(f"\nRESULTADO: {len(viaveis)} sistemas viaveis | {len(Gf):,} instancias")
print(f"  densidade={Gf.values.mean()*100:.2f}%  "
      f"cardinalidade={Gf.sum(axis=1).mean():.2f}  "
      f"min positivos={int(Gf.sum().min())}")
print(f"  instancias perdidas ao remover sistemas inviaveis: {len(idx)-len(Gf)}")

Gf.index.name = "ncRNA_Symbol"
Gf.to_csv(PROC / f"08_sistema_lncRNA_x_grupos_K{len(viaveis)}.csv", encoding="utf-8-sig")
pd.DataFrame([{"doenca": d, "sistema": cat[d], "n_lncRNAs": int(lpd[d])}
              for d in m.columns]).sort_values(
    ["sistema", "n_lncRNAs"], ascending=[True, False]).to_csv(
    PROC / f"08_sistema_mapa_doenca_grupo_K{len(viaveis)}.csv",
    index=False, encoding="utf-8-sig")

print(f"\nOK: 08_sistema_lncRNA_x_grupos_K{len(viaveis)}.csv")
print(f"OK: 08_sistema_mapa_doenca_grupo_K{len(viaveis)}.csv")
print("=" * 78)
