"""Executa o pipeline de dados inteiro, na ordem correta, e para no primeiro erro.

A numeracao dos arquivos reflete a ordem em que foram escritos, nao a ordem de
execucao. O passo 05 precisa rodar antes do 02, porque e ele que unifica as
variantes de nome (MALAT1, MALAT-1, Malat1) e define quais simbolos tem a
sequencia consultada. Este script cuida disso.

    01  filtra a planilha bruta (Homo sapiens, bloco validado experimentalmente)
    04  verifica que nao sobrou nenhum identificador de outra especie
    05  unifica variantes de nome e monta a matriz lncRNA x doenca
    02  recupera as sequencias no NCBI                     [so com --baixar]
    03  calcula as frequencias de k-mers (k = 1 e 2)
    06b agrupamento por sistema fisiologico (a base do artigo)
    08  grava a base pronta para treino em data/lncRNA_sis14/
    07  verifica a integridade de todos os arquivos gerados, inclusive a base

SEM A PLANILHA BRUTA
--------------------
A saida do passo 01 de que os demais dependem, data/processado/01_pares_unicos.csv,
ja esta publicada. Sem a planilha em data/bruto/, os passos 01 e 04 sao pulados
e o pipeline continua a partir dela, com o mesmo resultado.

SEM ACESSO A REDE
-----------------
O cache de sequencias em data/processado/ cobre todos os lncRNAs da base do
artigo. Por isso o passo 02 e pulado por padrao: consultar o NCBI de novo pode
devolver sequencias revisadas desde a construcao da base e, com elas, atributos
diferentes. Use --baixar apenas se quiser estender a base a outros lncRNAs.

Uso (a partir da raiz do repositorio):
    python -m scripts.pipeline_dados.rodar_pipeline
    python -m scripts.pipeline_dados.rodar_pipeline --baixar --email voce@exemplo.com
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

from caminhos import PROC, fonte_bruta  # noqa: E402


def executar(script: str, *args: str, env: dict | None = None) -> None:
    print("\n" + "#" * 78)
    print(f"# {script} {' '.join(args)}".rstrip())
    print("#" * 78, flush=True)
    inicio = time.time()
    proc = subprocess.run(
        [sys.executable, script, *args],
        cwd=AQUI,
        env={**os.environ, **(env or {}), "PYTHONIOENCODING": "utf-8"},
    )
    if proc.returncode != 0:
        sys.exit(f"\n{script} falhou (codigo {proc.returncode}). "
                 "Corrija o problema acima e rode de novo.")
    print(f"[{script}: {time.time() - inicio:.0f}s]")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--baixar", action="store_true",
                    help="consulta o NCBI para simbolos fora do cache (exige rede)")
    ap.add_argument("--email", default=os.environ.get("NCBI_EMAIL"),
                    help="e-mail exigido pelo NCBI; so usado com --baixar")
    ap.add_argument("--min-doencas", type=int, default=3,
                    help="corte de doencas por lncRNA (padrao: 3, o do artigo)")
    args = ap.parse_args()

    if fonte_bruta(obrigatoria=False):
        executar("01_filtrar_base_humana.py")
    else:
        if not (PROC / "01_pares_unicos.csv").exists():
            sys.exit("Sem a planilha bruta e sem data/processado/01_pares_unicos.csv "
                     "nao ha de onde partir. Veja data/bruto/README.md.")
        print("planilha bruta ausente em data/bruto/: passo 01 pulado, "
              "usando data/processado/01_pares_unicos.csv")

    executar("04_verificar_especie.py")
    executar("05_montar_base1_doencas_individuais.py")

    if args.baixar:
        extra = ["--email", args.email] if args.email else []
        executar("02_baixar_sequencias.py", "--min-doencas", str(args.min_doencas), *extra)
    elif not (PROC / "cache_sequencias.json").exists():
        sys.exit("cache de sequencias ausente: rode com --baixar --email voce@exemplo.com")
    else:
        print("\ncache de sequencias presente: passo 02 pulado "
              "(use --baixar para consultar o NCBI)")

    executar("03_calcular_kmer.py")
    executar("06b_agrupamento_por_sistema.py", "--min-doencas", str(args.min_doencas))
    executar("08_montar_conjunto_treino.py")
    # a verificacao roda por ultimo para cobrir tambem a base recem-gravada
    executar("07_verificar_integridade.py")

    print("\n" + "=" * 78)
    print("pipeline concluido. Base pronta em data/lncRNA_sis14/")
    print("Para treinar:  python -m scripts.rodar --dados data/lncRNA_sis14")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
