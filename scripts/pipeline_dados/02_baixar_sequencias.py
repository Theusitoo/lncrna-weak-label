"""
Passo 2 — Recuperacao das sequencias dos lncRNAs a partir do ncRNA Symbol.

Consulta o NCBI (banco `nucleotide`) via Entrez/Biopython, restringindo a
transcritos ncRNA do RefSeq humano, e guarda a sequencia mais longa de cada
simbolo (o transcrito representativo).

Caracteristicas:
  - CACHE RESUMIVEL: cada simbolo resolvido e gravado em cache_sequencias.json.
    Se o script for interrompido, basta rodar de novo que ele continua de onde parou.
  - RATE LIMIT: respeita o limite do NCBI (3 req/s sem chave, 10 req/s com chave).
  - LOG DE FALHAS: simbolos nao resolvidos vao para sequencias_nao_encontradas.csv

O CACHE E O DADO DE REFERENCIA
------------------------------
O RefSeq e atualizado continuamente: transcritos sao revisados, isoformas novas
aparecem e accessions antigos sao suprimidos. Consultar o NCBI hoje pode,
portanto, devolver sequencias diferentes das usadas no artigo, e com elas
atributos k-mer diferentes. O cache distribuido em data/processado/ congela as
sequencias efetivamente usadas. Com ele presente, nenhum simbolo e consultado de
novo e o script roda sem acesso a rede.

O NCBI exige um e-mail de contato em toda requisicao. Ele so e pedido quando ha
simbolos fora do cache, e nunca fica gravado no codigo:

    python 02_baixar_sequencias.py --email voce@exemplo.com
    (ou defina a variavel de ambiente NCBI_EMAIL)

Uma chave de API, opcional, sobe o limite de 3 para 10 requisicoes por segundo:
crie-a em https://www.ncbi.nlm.nih.gov/account/settings/ e defina NCBI_API_KEY.

Uso:
    python 02_baixar_sequencias.py --min-doencas 3 # os lncRNAs da base do artigo
    python 02_baixar_sequencias.py                 # todos os lncRNAs da base 1
    python 02_baixar_sequencias.py --limite 50     # so os primeiros 50 (teste rapido)
"""
import argparse
import json
import os
import re
import sys
import time

import pandas as pd

from caminhos import PROC

CACHE = PROC / "cache_sequencias.json"
SAIDA_FASTA = PROC / "02_sequencias.fasta"
SAIDA_FALHAS = PROC / "02_sequencias_nao_encontradas.csv"

PAUSA = 0.34  # segundos entre requisicoes; reduzida para 0.11 com chave de API
TENTATIVAS = 3
MAX_ISOFORMAS = 100  # teto de transcritos recuperados por gene

# Importado sob demanda: o Biopython so e necessario quando ha simbolos fora do
# cache. Com o cache completo, o script roda sem ele.
Entrez = None


def configurar_entrez(email: str) -> None:
    global Entrez, PAUSA
    try:
        from Bio import Entrez as _Entrez
    except ImportError:
        sys.exit("Ha simbolos fora do cache, e consulta-los exige o Biopython:\n"
                 "  pip install biopython")
    Entrez = _Entrez
    Entrez.email = email
    chave = os.environ.get("NCBI_API_KEY")
    if chave:
        Entrez.api_key = chave
        PAUSA = 0.11


def carregar_cache() -> dict:
    if CACHE.exists():
        with open(CACHE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def salvar_cache(cache: dict) -> None:
    tmp = CACHE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    tmp.replace(CACHE)


def _entrez_com_retry(func, **kwargs):
    """Executa uma chamada Entrez com retentativas e backoff exponencial."""
    for tentativa in range(TENTATIVAS):
        try:
            handle = func(**kwargs)
            resultado = handle.read()
            handle.close()
            # o Entrez pode devolver bytes ou str dependendo do endpoint
            if isinstance(resultado, bytes):
                resultado = resultado.decode("utf-8", errors="replace")
            return resultado
        except Exception:
            if tentativa == TENTATIVAS - 1:
                raise
            time.sleep(2 ** tentativa)
    return None


# Cascata de consultas. A primeira que retornar resultados vence.
#   E1 - estrita: simbolo oficial + classificado como ncRNA. Sem risco de erro.
#   E2 - sem biomol_ncrna: recupera pseudogenes e transcritos classificados como
#        misc_RNA, que o filtro de ncRNA barra (DUXAP8, BLACAT1, FER1L4...).
#   E3 - [All Fields] em vez de [Gene Name]: recupera APELIDOS usados na
#        literatura mas que nao sao o simbolo oficial (ATB, AWPPH, IUR...).
#
# E2 e E3 sao mais permissivas e podem casar com o gene errado, por isso os
# resultados delas passam por validacao de titulo (ver `titulo_confere`).
ESTRATEGIAS = [
    ("E1", '"{s}"[Gene Name] AND "Homo sapiens"[Organism] '
           'AND biomol_ncrna[PROP] AND srcdb_refseq[PROP]'),
    ("E2", '"{s}"[Gene Name] AND "Homo sapiens"[Organism] AND srcdb_refseq[PROP]'),
    ("E3", '{s}[All Fields] AND "Homo sapiens"[Organism] '
           'AND biomol_ncrna[PROP] AND srcdb_refseq[PROP]'),
]


def buscar_ids(simbolo: str, usar_fallback: bool) -> tuple:
    """Retorna (ids, estrategia) da primeira consulta que devolver resultados."""
    estrategias = ESTRATEGIAS if usar_fallback else ESTRATEGIAS[:1]
    for nome, molde in estrategias:
        bruto = _entrez_com_retry(
            Entrez.esearch, db="nucleotide",
            term=molde.format(s=simbolo), retmax=MAX_ISOFORMAS,
        )
        time.sleep(PAUSA)
        ids = re.findall(r"<Id>([^<]+)</Id>", bruto) if bruto else []
        if ids:
            return ids, nome
    return [], None


# Prefixos de accession do RefSeq que representam transcritos NAO-CODIFICANTES.
#   NR_ = ncRNA curado | XR_ = ncRNA predito por automacao
# Excluidos de proposito:
#   NM_/XM_ = mRNA (gene codificante de proteina, nao e lncRNA)
#   NC_/NG_/NT_/NW_ = registros genomicos (cromossomo inteiro, regiao gênica)
PREFIXOS_NCRNA = ("NR_", "XR_")


def eh_ncrna(accession: str) -> bool:
    """True se o accession for de um transcrito nao-codificante."""
    return accession.upper().startswith(PREFIXOS_NCRNA)


def titulo_confere(titulo: str, simbolo: str) -> bool:
    """Verifica se o simbolo aparece no titulo do FASTA como termo isolado.

    Evita que uma busca em texto livre associe a sequencia errada: "ATB" nao
    pode casar com "GATB" nem com "ATBF1". Hifen e underscore contam como parte
    do termo, pois simbolos como ABHD11-AS1 os contem.
    """
    padrao = rf"(?<![A-Za-z0-9_-]){re.escape(simbolo)}(?![A-Za-z0-9_-])"
    return re.search(padrao, titulo, flags=re.IGNORECASE) is not None


def buscar_fasta(ids: list) -> str:
    """Baixa as sequencias FASTA de uma lista de accessions."""
    if not ids:
        return ""
    return _entrez_com_retry(
        Entrez.efetch, db="nucleotide", id=",".join(ids), rettype="fasta", retmode="text"
    ) or ""


def parse_fasta(fasta_texto: str) -> list:
    """Converte o texto FASTA em [{'acc':..., 'titulo':..., 'seq':..., 'n':...}].

    Guardamos TODAS as isoformas: a escolha do transcrito representativo e feita
    no passo 3, para que mudar o criterio nao exija baixar tudo de novo.
    """
    registros, cab, partes = [], None, []

    def fechar():
        if cab is None:
            return
        seq = "".join(partes)
        registros.append({"acc": cab.split()[0], "titulo": cab,
                          "seq": seq, "n": len(seq)})

    for linha in fasta_texto.splitlines():
        if linha.startswith(">"):
            fechar()
            cab, partes = linha[1:].strip(), []
        elif linha.strip():
            partes.append(linha.strip())
    fechar()
    return [r for r in registros if r["n"] > 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-doencas", type=int, default=1,
                    help="manter apenas lncRNAs com pelo menos N doencas (padrao: 1)")
    ap.add_argument("--limite", type=int, default=None,
                    help="processar apenas os N primeiros simbolos (teste)")
    ap.add_argument("--email", default=os.environ.get("NCBI_EMAIL"),
                    help="e-mail de contato exigido pelo NCBI (ou NCBI_EMAIL). "
                         "So e necessario se houver simbolos fora do cache")
    ap.add_argument("--sem-fallback", action="store_true",
                    help="usar apenas a consulta estrita E1, sem as alternativas")
    ap.add_argument("--refazer-falhas", action="store_true",
                    help="reprocessar simbolos marcados como 'sem resultado' no cache")
    args = ap.parse_args()

    # ------------------------------------------------------- lista de simbolos
    # Usa a BASE 1 (simbolos ja normalizados e sem contaminacao de especie).
    # Cai para os pares brutos se a base ainda nao tiver sido montada.
    base1 = PROC / "06_base1_lncRNA_x_doencas.csv"
    if base1.exists():
        b1 = pd.read_csv(base1, index_col=0)
        contagem = b1.sum(axis=1)
    else:
        pares = pd.read_csv(PROC / "01_pares_unicos.csv")
        contagem = pares.groupby("ncRNA Symbol").size()

    simbolos = sorted(contagem[contagem >= args.min_doencas].index)
    print(f"lncRNAs com >= {args.min_doencas} doenca(s): {len(simbolos):,}")

    if args.limite:
        simbolos = simbolos[: args.limite]
        print(f"limitado aos primeiros {len(simbolos)}")

    # ------------------------------------------------------------------ cache
    cache = carregar_cache()

    def precisa_buscar(s: str) -> bool:
        if s not in cache:
            return True
        if cache[s].get("ok"):
            return False
        motivo = str(cache[s].get("motivo", ""))
        # falhas transitorias (rede, parsing) sao sempre reprocessadas
        if motivo.startswith("erro:"):
            return True
        # "sem resultado" so e reprocessado sob pedido -- util depois de
        # habilitar o fallback, que pode resolver o que E1 nao resolveu.
        return args.refazer_falhas

    pendentes = [s for s in simbolos if precisa_buscar(s)]
    reprocessar = sum(1 for s in pendentes if s in cache)
    print(f"ja em cache: {len(simbolos) - len(pendentes):,} | a buscar: {len(pendentes):,}"
          + (f" (dos quais {reprocessar:,} sao retentativas)" if reprocessar else ""))

    if pendentes:
        if not args.email:
            sys.exit(f"\n{len(pendentes):,} simbolos precisam ser consultados no NCBI, "
                     "que exige um e-mail de contato.\n"
                     "Informe-o com --email voce@exemplo.com ou pela variavel NCBI_EMAIL.")
        configurar_entrez(args.email)
        est = len(pendentes) * PAUSA * 2 / 60
        print(f"tempo estimado: ~{est:.0f} min\n")

    # ------------------------------------------------------------- busca
    for i, simbolo in enumerate(pendentes, 1):
        try:
            ids, estrategia = buscar_ids(simbolo, usar_fallback=not args.sem_fallback)
            if not ids:
                cache[simbolo] = {"ok": False, "motivo": "sem resultado"}
            else:
                fasta = buscar_fasta(ids[:MAX_ISOFORMAS])
                time.sleep(PAUSA)
                isoformas = parse_fasta(fasta)
                motivo = "fasta vazio"

                # (a) so transcritos nao-codificantes. E2 dispensa o filtro
                # biomol_ncrna do NCBI, entao pode trazer mRNA e ate cromossomo
                # inteiro; aqui isso e barrado pelo prefixo do accession.
                antes = len(isoformas)
                isoformas = [r for r in isoformas if eh_ncrna(r["acc"])]
                if antes and not isoformas:
                    motivo = "nenhum transcrito nao-codificante"

                # (b) E1 usa o simbolo oficial e e confiavel por construcao.
                # E2/E3 sao permissivas: exigimos que o titulo nomeie o simbolo.
                if estrategia != "E1" and isoformas:
                    isoformas = [r for r in isoformas
                                 if titulo_confere(r["titulo"], simbolo)]
                    if not isoformas:
                        motivo = "descartado na validacao de titulo"

                if isoformas:
                    cache[simbolo] = {"ok": True, "estrategia": estrategia,
                                      "isoformas": isoformas}
                else:
                    cache[simbolo] = {"ok": False, "motivo": motivo}
        except Exception as e:
            cache[simbolo] = {"ok": False, "motivo": f"erro: {type(e).__name__}"}

        if i % 25 == 0 or i == len(pendentes):
            salvar_cache(cache)
            ok = sum(1 for s in simbolos if cache.get(s, {}).get("ok"))
            pct = i / len(pendentes) * 100
            print(f"  [{i:5,}/{len(pendentes):,}] {pct:5.1f}%  resolvidos: {ok:,}", flush=True)

    salvar_cache(cache)

    # ------------------------------------------------------------- resultados
    resolvidos = {s: cache[s] for s in simbolos if cache.get(s, {}).get("ok")}
    falhas = [s for s in simbolos if not cache.get(s, {}).get("ok")]

    print("\n" + "=" * 70)
    print(f"  resolvidos: {len(resolvidos):,} de {len(simbolos):,} "
          f"({len(resolvidos)/len(simbolos)*100:.1f}%)")
    print(f"  falhas    : {len(falhas):,}")

    if resolvidos:
        est = pd.Series([v.get("estrategia", "E1") for v in resolvidos.values()])
        rotulo = {"E1": "E1 estrita (simbolo oficial + ncRNA)",
                  "E2": "E2 sem filtro de ncRNA",
                  "E3": "E3 busca por apelido"}
        print("\n  resolvidos por estrategia:")
        for k, v in est.value_counts().sort_index().items():
            print(f"    {rotulo.get(k, k):42s} {v:5,}")
        if (est != "E1").any():
            print("    (E2/E3 passaram por validacao de titulo; audite-os antes de publicar)")

        n_iso = pd.Series([len(v["isoformas"]) for v in resolvidos.values()])
        # o transcrito representativo e o de MENOR accession RefSeq (canonico)
        tam = pd.Series([min(v["isoformas"], key=lambda r: r["acc"])["n"]
                         for v in resolvidos.values()])
        print(f"\n  isoformas por lncRNA: mediana={n_iso.median():.0f}  max={n_iso.max()}")
        print(f"  comprimento do transcrito canonico (nt):")
        print(f"    min={tam.min():,}  mediana={tam.median():,.0f}  "
              f"media={tam.mean():,.0f}  max={tam.max():,}")

    with open(SAIDA_FASTA, "w", encoding="utf-8") as f:
        for s, v in sorted(resolvidos.items()):
            r = min(v["isoformas"], key=lambda x: x["acc"])
            f.write(f">{s}|{r['acc']}|{r['n']}\n")
            for j in range(0, len(r["seq"]), 60):
                f.write(r["seq"][j:j + 60] + "\n")
    print(f"\n  OK: FASTA em {SAIDA_FASTA.name}")

    if falhas:
        pd.DataFrame({
            "ncRNA Symbol": falhas,
            "motivo": [cache[s].get("motivo", "?") for s in falhas],
            "n_doencas": [int(contagem.get(s, 0)) for s in falhas],
        }).to_csv(SAIDA_FALHAS, index=False, encoding="utf-8-sig")
        print(f"  OK: falhas em {SAIDA_FALHAS.name}")

    print("=" * 70)


if __name__ == "__main__":
    main()
