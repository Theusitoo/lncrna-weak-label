# Dados

Tudo o que está aqui foi produzido pelos scripts de
[`scripts/pipeline_dados/`](../scripts/pipeline_dados/) a partir de duas fontes
públicas:

- **LncRNADisease v3.0** (Lin et al., *Nucleic Acids Research*, 2024): as
  associações lncRNA-doença validadas experimentalmente.
- **RefSeq / NCBI** (Goldfarb et al., *Nucleic Acids Research*, 2025): as
  sequências dos transcritos.

Rodar o pipeline de novo reproduz estes arquivos byte a byte.

## Organização

```
data/
├── lncRNA_sis14/    a base do artigo, pronta para treino
├── processado/      saída de cada passo do pipeline, para auditoria
└── bruto/           planilha da LncRNADisease (não versionada, ver bruto/README.md)
```

## `lncRNA_sis14/`: a base do artigo

| | |
| --- | --- |
| instâncias | 483 lncRNAs humanos |
| atributos | 20 frequências de k-mers (k = 1 e k = 2) |
| rótulos | 14 sistemas fisiológicos |
| densidade de rótulos | 36,2% |
| cardinalidade média | 5,06 sistemas por lncRNA |
| partições | 10 folds, estratificação multirrótulo, semente 0 |

- **`features.csv`**: primeira coluna `symbol` (o símbolo do lncRNA) e as
  colunas `k1_A, k1_C, k1_G, k1_U` e `k2_AA ... k2_UU`. Cada bloco soma 1 em
  cada linha.
- **`labels.csv`**: primeira coluna `ncRNA_Symbol`, com os mesmos símbolos na
  mesma ordem, e uma coluna 0/1 por sistema.
- **`features_splits.csv`**: colunas `train_fold1..10` e `test_fold1..10`, com
  as **posições** (base zero) das linhas de cada partição.

Os 14 rótulos são: Cabeca_Pescoco, Cardiovascular, Dermatologico, Digestivo,
Endocrino_Metabolico, Ginecologico_Reprodutivo, Hematologico, Infeccioso, Mama,
Musculoesqueletico, Neurologico, Oftalmologico, Renal_Urologico e Respiratorio.
A atribuição de cada uma das 422 doenças a um sistema está em
`processado/08_sistema_mapa_doenca_grupo_K14.csv`. O mapa tem duas categorias a
mais que não viraram rótulo: `Inflamatorio_Autoimune`, que ficou abaixo do
mínimo de 30 lncRNAs positivos, e `NAO_CLASSIFICADA`, para as doenças que
nenhuma regra cobre.

Um zero nesta base não é um negativo confirmado: significa "associação não
registrada", e não "associação inexistente". Esse é o ponto de partida do
trabalho. Os métodos com imputação tratam os zeros como desconhecidos, e todas
as métricas são calculadas contra rótulos sabidamente incompletos.

## `processado/`: arquivos intermediários

O prefixo numérico indica o passo do pipeline que gerou o arquivo.

| arquivo | conteúdo |
| --- | --- |
| `01_pares_unicos.csv` | pares (símbolo, doença) distintos dos registros humanos validados experimentalmente; é o ponto de partida dos passos seguintes |
| `02_sequencias.fasta` | transcrito representativo de cada lncRNA resolvido |
| `02_sequencias_nao_encontradas.csv` | símbolos sem sequência no RefSeq e o motivo |
| `cache_sequencias.json` | **todas** as isoformas recuperadas do NCBI, por símbolo. É o registro das sequências usadas: o RefSeq é revisado continuamente, e uma consulta feita hoje pode devolver outras |
| `03_features_kmer12.csv` | os 20 atributos de k-mers de cada lncRNA com sequência |
| `03_transcritos_usados.csv` | qual accession foi usado para cada lncRNA, com o comprimento e a estratégia de busca |
| `04_simbolos_em_outras_especies.csv` | símbolos humanos que também aparecem com outra espécie na fonte |
| `04_simbolos_suspeitos.csv` | símbolos com nomenclatura típica de roedor, para conferência manual |
| `06_base1_lncRNA_x_doencas.csv` | **base 1**: matriz binária de 4.948 lncRNAs por 422 doenças individuais |
| `06_base1_fusoes.csv` | variantes de nome unificadas (ex.: MALAT1, MALAT-1, LncRNA MALAT1). **Vale auditar** |
| `06_base1_removidos.csv` | símbolos descartados e o motivo (roedor, mais de um gene, microRNA...) |
| `08_sistema_lncRNA_x_grupos_K14.csv` | rótulos por sistema fisiológico, já restritos aos lncRNAs com pelo menos três doenças |
| `08_sistema_mapa_doenca_grupo_K14.csv` | cada doença, o sistema a que foi atribuída e quantos lncRNAs ela tem |

Duas decisões de construção mudam a base e merecem ser conferidas por quem a
usar: a unificação de variantes de nome (`06_base1_fusoes.csv`) e a atribuição
de cada doença a um sistema (`08_sistema_mapa_doenca_grupo_K14.csv`). Os dois
arquivos estão publicados para que isso possa ser feito sem rodar nada.

## Formato para usar outra base

Os métodos funcionam com qualquer problema multirrótulo. Basta criar uma pasta
com dois arquivos CSV e gerar o terceiro com o script:

```
data/minha_base/
├── features.csv
├── labels.csv
└── features_splits.csv     gerado por scripts/gerar_splits.py
```

**`features.csv`**: a primeira coluna identifica a instância; as demais são
atributos numéricos, sem valores ausentes (as florestas do scikit-learn não
aceitam NaN). Atributos categóricos precisam ser codificados antes. A escala
não importa: as florestas são insensíveis a ela, e a MLP padroniza os atributos
internamente.

```
id,atributo1,atributo2,atributo3
gene_A,0.12,3.4,1
gene_B,0.98,2.1,0
```

**`labels.csv`**: a primeira coluna identifica a instância, com **os mesmos
valores e na mesma ordem** de `features.csv` (o nome da coluna pode ser
diferente). As demais são rótulos 0/1, um por coluna, e todo rótulo precisa de
pelo menos um positivo. No cenário fracamente rotulado, um 1 é uma anotação
confiável e um 0 significa apenas "não anotado"; não é preciso marcar quais
zeros são duvidosos.

```
id,rotulo1,rotulo2,rotulo3
gene_A,1,0,1
gene_B,0,0,1
```

**`features_splits.csv`**: gerado com

```bash
python -m scripts.gerar_splits --dados data/minha_base
```

O script usa estratificação multirrótulo (Sechidis et al., 2011), que distribui
os positivos de cada rótulo entre os folds; `--folds` (padrão 10) e `--semente`
(padrão 0) controlam a partição. As partições ficam em disco para que todos os
métodos vejam exatamente as mesmas. Por isso, **gerar de novo invalida qualquer
resultado já calculado**, e o script se recusa a sobrescrever sem
`--sobrescrever`.

Os scripts validam a base antes de treinar e explicam o problema quando
encontram algum: arquivos ausentes, contagens de linhas diferentes,
identificadores fora de ordem, rótulos diferentes de 0 e 1, atributos com NaN,
rótulos sem positivos ou partições inconsistentes.

**Rótulos raros.** Com 10 folds, um rótulo com menos de 10 positivos fica sem
nenhum positivo em algum fold de teste, e suas métricas ficam instáveis. O
`gerar_splits` avisa quando isso acontece; as saídas são usar menos folds,
agrupar rótulos semelhantes ou remover os mais raros. Na base do artigo, o
mínimo adotado foi de 30 positivos por rótulo.
