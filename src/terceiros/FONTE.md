# Origem do código desta pasta

O conteúdo de `src/terceiros/` parte do código de **Pedro Ilídio**, publicado
junto ao artigo:

> P. Ilidio, R. Cerri, C. Vens, F. K. Nakano.
> *Deep forests with tree-embeddings and label imputation for weak-label
> learning*. International Joint Conference on Neural Networks (IJCNN), 2024.
> IEEE, pp. 1-8.

Licença: **BSD 3-Clause**, Copyright (c) 2023, Pedro Ilídio. O texto integral,
que deve acompanhar este código em qualquer redistribuição, está no arquivo
[`LICENSE`](LICENSE) desta pasta.

## O que há aqui

| arquivo | conteúdo |
| --- | --- |
| `cascade_learn/cascade.py` | a cascata propriamente dita: `Cascade`, `AlternatingLevel` (cruzamento) e `SequentialLevel` |
| `cascade_learn/tree_embedder.py` | `ForestEmbedder`, a representação por caminhos nas árvores (tree-embeddings) |
| `cascade_learn/weak_labels.py` | os imputadores `SCARImputer` (FLA) e `LabelComplementImputer` (SLC) |
| `cascade_learn/tice.py` | estimativa da frequência de rótulo pelo TIcE, de Bekker e Davis (2018) |
| `cascade_learn/estimator_adapters.py` | adaptadores entre regressores, classificadores e transformadores |
| `label_complement.py` | o imputador LC de Wang, Yang e Li (2020), usado pelo LCForest |
| `LICENSE` | a licença original, que deve acompanhar este código |

## O que difere do repositório original

Os arquivos daqui são **os mesmos que produziram os resultados do artigo**,
copiados sem nenhuma edição da pasta `models/` usada nos experimentos. Ficou de
fora apenas o `estimators.py`, com cascatas já montadas pelo autor, que este
código não usa. O nome da pasta virou `src/terceiros/`; como os arquivos só usam
imports relativos, isso não alterou nenhuma linha.

Essa versão contém ajustes de compatibilidade com versões recentes do Python,
do NumPy e do scikit-learn:

- imports alternativos (`try ... except ImportError`) em `cascade.py`,
  `weak_labels.py` e `label_complement.py`, para APIs que mudaram de lugar
  entre versões;
- em `estimator_adapters.py`, `UnanimityClassifier._pos_proba` trata o caso em
  que um fold interno da validação cruzada só contém uma classe, e o
  `predict_proba` devolve uma coluna em vez de duas.

Como não há histórico de versões da pasta original, não é possível listar com
certeza tudo o que difere do repositório do autor. Para a versão dele, consulte
o repositório publicado com o artigo.

O código deste repositório que constrói as cascatas está em
[`src/cascata.py`](../cascata.py), fora desta pasta.

## Por que copiar em vez de usar submódulo

Um submódulo git aponta para um commit de um repositório de terceiros. Se aquele
repositório for movido, renomeado ou tiver o histórico reescrito, o submódulo
deixa de resolver e este repositório para de funcionar, sem que haja o que fazer
deste lado. Como a licença BSD 3-Clause permite a redistribuição com a
manutenção do aviso de copyright, a cópia é a opção que mantém o repositório
executável por conta própria.

O repositório original é a referência para qualquer dúvida sobre o código desta
pasta, e correções feitas nele não chegam aqui automaticamente.
