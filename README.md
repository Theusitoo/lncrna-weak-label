# Predição de Doenças Associadas a lncRNAs com Aprendizado Multirrótulo Fracamente Rotulado

Repositório do projeto de Iniciação Científica sobre a predição de doenças
associadas a RNAs longos não codificantes (lncRNAs). O projeto investiga métodos
de aprendizado multirrótulo fracamente rotulado (*weak label learning*, WELL)
baseados em cascatas de florestas, com e sem imputação de rótulos, e os compara
com métodos multirrótulo clássicos.

## 📝 Visão geral

Um mesmo lncRNA pode estar associado a várias doenças, o que torna a predição um
problema multirrótulo. As bases biomédicas, porém, só registram as associações
já investigadas: um zero na matriz de rótulos significa "não registrado", e não
"não associado". Tratar esses zeros como negativos introduz falsos negativos no
treino.

Para investigar esse cenário, construímos uma base de 483 lncRNAs humanos a
partir da LncRNADisease v3.0 e avaliamos quinze métodos sob o mesmo protocolo.
Os métodos com imputação estimam, nível a nível da cascata, quais zeros são
positivos não anotados e os convertem durante o treino. As mesmas cascatas sem
imputação servem de controle, para separar o efeito do imputador do efeito da
arquitetura.

## 🔍 Características

- **Dados:** associações lncRNA-doença da LncRNADisease v3.0 e sequências do
  RefSeq, com todo o pipeline de construção incluído
- **Atributos:** frequências de k-mers (k = 1 e k = 2), 20 por lncRNA
- **Rótulos:** 422 doenças agrupadas em 14 sistemas fisiológicos
- **WELL com imputação:** LCForest, SLCForest, FLAForest, CaFE-SLC e CaFE-FLA
- **WELL sem imputação (controles):** RF+ET, gcForest, CaFE e CaFE-OS
- **Multirrótulo clássico:** BR, CC e ECC, cada um com floresta aleatória e com
  rede neural rasa
- **Avaliação:** validação cruzada de 10 folds com estratificação multirrótulo,
  seis métricas, teste de Friedman e Wilcoxon pareado com correção de
  Benjamini-Hochberg

## 📊 Dados

A base do artigo está em `data/lncRNA_sis14/`, pronta para uso:

| | |
| --- | --- |
| instâncias | 483 lncRNAs humanos |
| atributos | 20 frequências de k-mers |
| rótulos | 14 sistemas fisiológicos |
| densidade de rótulos | 36,2% |
| partições | 10 folds, estratificação multirrótulo |

Os arquivos intermediários de cada passo da construção estão em
`data/processado/`, para que as decisões (unificação de nomes de genes, escolha
do transcrito, agrupamento das doenças) possam ser auditadas. A descrição de
cada arquivo e o formato para usar outra base estão em
[`data/README.md`](data/README.md).

## 📂 Estrutura do projeto

```
.
├── data/                          # Bases e arquivos intermediários
│   ├── lncRNA_sis14/              # Base do artigo, pronta para treino
│   ├── processado/                # Saída de cada passo da construção da base
│   └── bruto/                     # Onde colocar a planilha da LncRNADisease
├── scripts/                       # Scripts de execução
│   ├── rodar.py                   # Treina os métodos e grava as predições
│   ├── avaliar.py                 # Calcula as métricas e os testes estatísticos
│   ├── gerar_splits.py            # Gera as partições de uma base nova
│   ├── gerar_base_exemplo.py      # Gera uma base sintética para testar a instalação
│   └── pipeline_dados/            # Construção da base, das fontes até o CSV final
├── src/                           # Implementação dos métodos
│   ├── terceiros/                 # Cascatas e imputadores de Ilidio et al. (2024)
│   ├── cascata.py                 # As nove cascatas de florestas
│   ├── classicos.py               # BR, CC e ECC
│   ├── metricas.py                # Métricas e testes pareados
│   ├── dados.py                   # Leitura e validação das bases
│   └── splits.py                  # Partições estratificadas
├── requirements.txt               # Dependências do projeto
└── README.md                      # Este arquivo
```

## ⚙️ Instalação

Clone o repositório:

```bash
git clone https://github.com/Theusitoo/lncrna-weak-label
```

```bash
cd lncrna-weak-label
```

Crie um ambiente virtual (Python 3.10 ou mais recente):

```bash
python -m venv .venv
```

Ative o ambiente.

Windows

```bash
.\.venv\Scripts\activate
```

Linux/macOS

```bash
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Para conferir a instalação, gere uma base sintética pequena e rode três métodos
nela, um de cada grupo (cerca de cinco minutos):

```bash
python -m scripts.gerar_base_exemplo
```

```bash
python -m scripts.rodar --dados data/exemplo --metodos gcforest cafe_fla br_rf
```

## 🚀 Executando os experimentos

Todos os comandos rodam a partir da raiz do repositório.

Treinar um ou mais métodos. As probabilidades preditas em cada fold vão para
`resultados/<nome da base>/`:

```bash
python -m scripts.rodar --dados data/lncRNA_sis14 --metodos cafe_fla gcforest br_rf
```

Avaliar as predições. O script gera `resumo.csv`, `metricas_por_fold.csv` e
`comparacoes.csv`:

```bash
python -m scripts.avaliar --dados data/lncRNA_sis14 --resultados resultados/lncRNA_sis14
```

Listar os métodos disponíveis e seus componentes:

```bash
python -m scripts.rodar --listar
```

Sem `--metodos`, o `rodar` treina os quinze métodos do artigo, o que levou cerca
de 35 minutos num computador pessoal; os mais lentos (LCForest, ECC-RF e as
variantes do CaFE) levam de 4 a 6 minutos cada. Um método já concluído é
pulado, de modo que uma execução interrompida pode ser retomada com o mesmo
comando. Para reproduzir o artigo:

```bash
python -m scripts.rodar --dados data/lncRNA_sis14
```

```bash
python -m scripts.avaliar --dados data/lncRNA_sis14 --resultados resultados/lncRNA_sis14
```

### Métodos

| chave | método | atributos propagados | cruzamento | imputação |
| --- | --- | --- | --- | --- |
| `final_estimator` | RF+ET | nenhum (não é cascata) | não | nenhuma |
| `gcforest` | gcForest | probabilidades (OS) | não | nenhuma |
| `cafe` | CaFE | tree-embeddings (TE) | sim | nenhuma |
| `cafe_os` | CaFE-OS | TE e OS | sim | nenhuma |
| `lcforest` | LCForest | OS | não | LC |
| `slcforest` | SLCForest | OS | sim | SLC |
| `flaforest` | FLAForest | OS | sim | FLA |
| `cafe_slc` | CaFE-SLC | TE e OS | sim | SLC |
| `cafe_fla` | CaFE-FLA | TE e OS | sim | FLA |
| `br_rf`, `br_mlp` | BR-RF, BR-MLP | | | |
| `cc_rf`, `cc_mlp` | CC-RF, CC-MLP | | | |
| `ecc_rf`, `ecc_mlp` | ECC-RF, ECC-MLP | | | |

Os hiperparâmetros seguem Ilidio et al. (2024), sem otimização por base: 150
árvores por floresta, folha mínima de cinco amostras, √d atributos por nó,
metade das amostras por árvore e dez níveis por cascata. Os imputadores SLC e
FLA usam limiar 0,5 e percentil 0,95; o LC usa limiar 0,4 com unanimidade entre
as florestas. A MLP tem uma camada oculta de 100 unidades, e o ECC combina dez
cadeias. Os valores estão em `src/cascata.py` e `src/classicos.py`.

### Usando outra base

Os métodos funcionam com qualquer problema multirrótulo. Crie uma pasta com
`features.csv` e `labels.csv` no formato descrito em
[`data/README.md`](data/README.md), gere as partições e treine:

```bash
python -m scripts.gerar_splits --dados data/minha_base
```

```bash
python -m scripts.rodar --dados data/minha_base
```

### Reconstruindo a base

```bash
python -m scripts.pipeline_dados.rodar_pipeline
```

O pipeline roda sem a planilha bruta e sem acesso à rede, a partir de
`data/processado/`. Os passos e as decisões de construção estão em
[`scripts/pipeline_dados/README.md`](scripts/pipeline_dados/README.md).
