"""Calcula as seis metricas por fold e aplica os testes estatisticos pareados.

Le as probabilidades gravadas por `rodar.py`, confronta cada fold com os
rotulos verdadeiros usando as posicoes de `features_splits.csv` e produz tres
arquivos:

    metricas_por_fold.csv   uma linha por metodo, fold e metrica
    resumo.csv              media, desvio e rank medio por metodo e metrica
    comparacoes.csv         Wilcoxon pareado com correcao Benjamini-Hochberg

As metricas sao calculadas FOLD A FOLD, nunca sobre a concatenacao das
predicoes de todos os folds. A validacao cruzada produz dez estimativas
independentes, e e sobre essas dez que o teste pareado opera: concatenar antes
descartaria a variabilidade entre folds e tornaria qualquer teste impossivel.

Uma ressalva que vale para toda leitura destes numeros: as metricas sao
calculadas contra rotulos sabidamente incompletos. Um metodo que acerte uma
associacao real ainda nao registrada na base e penalizado por todas elas, e a
penalidade recai justamente sobre os metodos com imputacao, cujo proposito e
recuperar essas associacoes.

Uso:
    python -m scripts.avaliar --dados data/exemplo --resultados resultados/exemplo
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src  # noqa: E402
from src.dados import (  # noqa: E402
    BaseInvalida,
    carregar_atributos_e_rotulos,
    carregar_splits,
    numero_de_folds,
    particoes,
)
from src.metricas import (  # noqa: E402
    MAIOR_E_MELHOR,
    avaliar_fold,
    comparacoes_pareadas,
    friedman,
    resumir,
    tabela_por_fold,
)

PADRAO_ARQUIVO = re.compile(
    r"^labels_(?P<metodo>.+?)"
    r"_inductive_multilabel_predictions_test_fold(?P<fold>\d+)\.csv$"
)


def descobrir(resultados: Path) -> dict[str, dict[int, Path]]:
    """Mapeia metodo -> {fold: arquivo}, considerando so os metodos conhecidos."""
    encontrados: dict[str, dict[int, Path]] = {}
    for caminho in sorted(resultados.glob("labels_*_predictions_test_fold*.csv")):
        m = PADRAO_ARQUIVO.match(caminho.name)
        if not m or m.group("metodo") not in src.NOMES:
            continue
        encontrados.setdefault(m.group("metodo"), {})[int(m.group("fold"))] = caminho
    return encontrados


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dados", required=True, help="diretorio da base")
    ap.add_argument("--resultados", required=True,
                    help="diretorio com as predicoes gravadas por rodar.py")
    ap.add_argument("--saida", default=None,
                    help="onde gravar os CSV (padrao: o proprio --resultados)")
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--alfa", type=float, default=0.05)
    ap.add_argument("--limiar", type=float, default=0.5,
                    help="limiar das metricas de identificacao (padrao: 0,5)")
    args = ap.parse_args()

    try:
        _, y = carregar_atributos_e_rotulos(args.dados)
        splits = carregar_splits(args.dados, args.folds)
    except BaseInvalida as erro:
        print(f"erro na base: {erro}", file=sys.stderr)
        return 1

    resultados = Path(args.resultados)
    if not resultados.is_dir():
        print(f"erro: {resultados} nao e um diretorio", file=sys.stderr)
        return 1

    n_folds = args.folds or numero_de_folds(splits)
    encontrados = descobrir(resultados)

    if not encontrados:
        print(f"erro: nenhuma predicao encontrada em {resultados}", file=sys.stderr)
        return 1

    print("=" * 78)
    print(f"base        {Path(args.dados).resolve()}")
    print(f"predicoes   {resultados.resolve()}")
    print(f"limiar      {args.limiar}")
    print("=" * 78)

    por_metodo: dict[str, dict[int, dict[str, float]]] = {}
    incompletos: list[str] = []

    for metodo, arquivos in sorted(encontrados.items()):
        nome = src.NOMES.get(metodo, metodo)
        if len(arquivos) < n_folds:
            incompletos.append(f"{nome} ({len(arquivos)}/{n_folds} folds)")
            continue

        valores: dict[int, dict[str, float]] = {}
        erro_de_forma = None
        for fold, pos_treino, pos_teste in particoes(splits, n_folds):
            proba = pd.read_csv(arquivos[fold])
            if len(proba) != len(pos_teste):
                erro_de_forma = (
                    f"fold {fold}: {len(proba)} predicoes para "
                    f"{len(pos_teste)} instancias de teste"
                )
                break
            if proba.shape[1] != y.shape[1]:
                erro_de_forma = (
                    f"fold {fold}: {proba.shape[1]} colunas para "
                    f"{y.shape[1]} rotulos"
                )
                break
            valores[fold] = avaliar_fold(
                y.values[pos_teste], proba.values, limiar=args.limiar
            )

        if erro_de_forma:
            print(f"  {nome}: ignorado -- {erro_de_forma}")
            print("    As predicoes foram geradas sobre outras particoes. "
                  "Rode de novo ou restaure o features_splits.csv original.")
            continue

        por_metodo[nome] = valores

    if incompletos:
        print("\nignorados por estarem incompletos:")
        for i in incompletos:
            print(f"  {i}")

    if not por_metodo:
        print("\nerro: nenhum metodo completo para avaliar", file=sys.stderr)
        return 1

    tabelas = tabela_por_fold(por_metodo)
    destino = Path(args.saida or resultados)
    destino.mkdir(parents=True, exist_ok=True)

    # metricas por fold
    longo = [
        {"metrica": metrica, "metodo": metodo, "fold": fold, "valor": valor}
        for metrica, tabela in tabelas.items()
        for metodo in tabela.columns
        for fold, valor in tabela[metodo].items()
    ]
    pd.DataFrame(longo).to_csv(
        destino / "metricas_por_fold.csv", index=False, encoding="utf-8-sig"
    )

    grupos = {nome: _grupo_por_nome(nome) for nome in por_metodo}

    linhas_resumo = []
    linhas_comparacao = []

    for metrica, tabela in tabelas.items():
        maior_melhor = MAIOR_E_MELHOR[metrica]
        res = resumir(tabela, maior_melhor)

        print(f"\n{metrica}  ({'maior' if maior_melhor else 'menor'} e melhor)")
        for metodo, linha in res.iterrows():
            print(f"  {grupos[metodo]:22s} {metodo:11s} "
                  f"{linha['media']:.4f} +/- {linha['desvio']:.4f}"
                  f"   rank {linha['rank_medio']:.2f}")
            linhas_resumo.append({
                "metrica": metrica,
                "grupo": grupos[metodo],
                "metodo": metodo,
                "media": linha["media"],
                "desvio": linha["desvio"],
                "rank_medio": linha["rank_medio"],
            })

        for g in sorted(set(grupos.values())):
            do_grupo = [m for m in res.index if grupos[m] == g]
            if do_grupo:
                print(f"    media do grupo {g:22s} "
                      f"{res.loc[do_grupo, 'media'].mean():.4f}")

        p_omnibus = friedman(tabela)
        if np.isnan(p_omnibus):
            print("    Friedman: menos de tres metodos, teste nao aplicavel")
            continue

        print(f"    Friedman p = {p_omnibus:.2e}", end="")
        if p_omnibus > args.alfa:
            print("  -> nenhuma diferenca detectada no conjunto")
            continue
        print()

        comparacoes = comparacoes_pareadas(tabela, args.alfa)
        comparacoes.insert(0, "metrica", metrica)
        linhas_comparacao.append(comparacoes)

        melhor = res.index[0]
        empates = [
            outro for outro in res.index[1:]
            if not _significativo(comparacoes, melhor, outro)
        ]
        print(f"    melhor: {melhor}", end="")
        print(f" | empata com {', '.join(empates)}" if empates
              else " | vence sozinho")

    pd.DataFrame(linhas_resumo).to_csv(
        destino / "resumo.csv", index=False, encoding="utf-8-sig"
    )
    if linhas_comparacao:
        pd.concat(linhas_comparacao, ignore_index=True).to_csv(
            destino / "comparacoes.csv", index=False, encoding="utf-8-sig"
        )

    print("\n" + "=" * 78)
    print("gravado em", destino.resolve())
    print("  metricas_por_fold.csv")
    print("  resumo.csv")
    if linhas_comparacao:
        print("  comparacoes.csv")
    print("=" * 78)
    return 0


def _grupo_por_nome(nome: str) -> str:
    """Encontra o grupo a partir do nome de exibicao do metodo."""
    for chave, exibicao in src.NOMES.items():
        if exibicao == nome:
            return src.grupo(chave)
    return "desconhecido"


def _significativo(comparacoes: pd.DataFrame, a: str, b: str) -> bool:
    linha = comparacoes[
        ((comparacoes.metodo_a == a) & (comparacoes.metodo_b == b))
        | ((comparacoes.metodo_a == b) & (comparacoes.metodo_b == a))
    ]
    return bool(linha.significativo_bh.iloc[0]) if len(linha) else False


if __name__ == "__main__":
    raise SystemExit(main())
