"""Caminhos usados por todos os passos do pipeline de dados.

Tudo e relativo a raiz do repositorio, de modo que o pipeline roda no mesmo
lugar em que foi clonado, em qualquer sistema operacional.

    data/bruto/          o arquivo baixado da LncRNADisease v3.0
    data/processado/     saidas intermediarias de cada passo (auditaveis)
    data/lncRNA_sis14/   a base pronta para treino, no formato do pacote
"""
from pathlib import Path

# este arquivo esta em scripts/pipeline_dados/, dois niveis abaixo da raiz
RAIZ = Path(__file__).resolve().parents[2]
DADOS = RAIZ / "data"
BRUTO = DADOS / "bruto"
PROC = DADOS / "processado"
# as bases prontas ficam direto em data/, uma pasta por base
BASES = DADOS

PROC.mkdir(parents=True, exist_ok=True)


def fonte_bruta(obrigatoria: bool = True):
    """Localiza a planilha da LncRNADisease em data/bruto/.

    Aceita qualquer nome de arquivo, desde que haja exatamente uma planilha
    .xlsx no diretorio: o nome do arquivo distribuido pelo site pode mudar entre
    downloads, e exigir um nome fixo so criaria um passo manual de renomeacao.
    """
    planilhas = sorted(BRUTO.glob("*.xlsx")) if BRUTO.exists() else []
    planilhas = [p for p in planilhas if not p.name.startswith("~$")]  # trava do Excel

    if len(planilhas) == 1:
        return planilhas[0]
    if not obrigatoria:
        return None
    if not planilhas:
        raise SystemExit(
            f"Nenhuma planilha encontrada em {BRUTO}\n"
            "Baixe a base da LncRNADisease v3.0 e coloque o arquivo .xlsx nesse "
            "diretorio. Instrucoes em data/bruto/README.md"
        )
    raise SystemExit(
        f"Ha {len(planilhas)} planilhas em {BRUTO}; deixe apenas uma:\n"
        + "\n".join(f"  {p.name}" for p in planilhas)
    )
