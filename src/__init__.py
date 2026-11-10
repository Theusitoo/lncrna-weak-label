"""Aprendizado multirrotulo fracamente rotulado com cascatas de florestas.

O pacote reune, sob um unico protocolo experimental, dois grupos de metodos:

    cascatas de florestas profundas, com e sem imputacao de rotulos
    metodos classicos de classificacao multirrotulo

A ideia do cenario WELL (weak label learning) e que um zero na matriz de rotulos
nao significa "negativo", mas "nao se sabe". Os metodos com imputacao estimam,
camada a camada, quais desses zeros sao positivos nao anotados e os convertem
durante o treino. Os metodos sem imputacao e os classicos tratam cada zero como
negativo confirmado, e entram na comparacao para medir o custo dessa suposicao.

Pontos de entrada:

    src.dados          leitura e validacao de uma base
    src.splits         geracao das particoes estratificadas
    src.cascata        as nove cascatas de florestas
    src.classicos      os metodos multirrotulo classicos
    src.metricas       as seis metricas e os testes pareados

Os scripts em `scripts/` cobrem o uso pela linha de comando.
"""
from . import cascata, classicos, dados, metricas, splits

__version__ = "1.0.0"

#: Todos os metodos disponiveis, das cascatas e dos classicos, chave -> nome.
NOMES = {**cascata.NOMES, **classicos.NOMES}


def grupo(metodo: str) -> str:
    """Classifica um metodo em um dos tres grupos usados na analise."""
    if metodo in cascata.COM_IMPUTACAO:
        return "WELL com imputacao"
    if metodo in cascata.SEM_IMPUTACAO:
        return "WELL sem imputacao"
    if metodo in classicos.METODOS:
        return "multirrotulo classico"
    raise ValueError(f"metodo desconhecido: {metodo!r}")


def e_cascata(metodo: str) -> bool:
    return metodo in cascata.COMPONENTES


__all__ = [
    "cascata",
    "classicos",
    "dados",
    "metricas",
    "splits",
    "NOMES",
    "grupo",
    "e_cascata",
    "__version__",
]
