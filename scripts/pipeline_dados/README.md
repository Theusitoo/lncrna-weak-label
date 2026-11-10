# Pipeline de dados

Os scripts desta pasta constroem a base do artigo a partir das fontes públicas.
Para rodar tudo, na ordem correta, a partir da raiz do repositório:

```bash
python -m scripts.pipeline_dados.rodar_pipeline
```

Sem a planilha bruta e sem acesso à rede, o pipeline reproduz byte a byte os
arquivos de `data/processado/` e `data/lncRNA_sis14/`: os passos 01 e 04 são
pulados (a saída do 01 de que os demais dependem, `01_pares_unicos.csv`, já está
publicada) e o passo 02 usa o cache de sequências.

## Ordem de execução

A numeração reflete a ordem em que os scripts foram escritos, não a de
execução. O passo 05 roda antes do 02 porque é ele que define quais símbolos
existem depois da unificação de nomes.

| ordem | script | o que faz |
| --- | --- | --- |
| 1 | `01_filtrar_base_humana.py` | filtra a planilha: bloco validado experimentalmente e *Homo sapiens* (só com a planilha) |
| 2 | `04_verificar_especie.py` | confere que não restou identificador de outra espécie (só com a planilha) |
| 3 | `05_montar_base1_doencas_individuais.py` | unifica variantes de nome e monta a matriz lncRNA x doença |
| 4 | `02_baixar_sequencias.py` | recupera as sequências no RefSeq (só com `--baixar`) |
| 5 | `03_calcular_kmer.py` | calcula as 20 frequências de k-mers |
| 6 | `06b_agrupamento_por_sistema.py` | agrupa as doenças por sistema fisiológico |
| 7 | `08_montar_conjunto_treino.py` | grava a base em `data/lncRNA_sis14/` |
| 8 | `07_verificar_integridade.py` | confere todos os arquivos gerados, inclusive a base |

Cada script também roda isoladamente, por exemplo:
`python scripts/pipeline_dados/05_montar_base1_doencas_individuais.py`.

## Decisões de construção

Estas são as escolhas que alteram a base. Cada uma está documentada no
cabeçalho do script correspondente.

**Recorte da fonte (passo 01).** A planilha tem dois blocos: os registros da
categoria `LncRNA`, validados experimentalmente, até o ID 12881, e um segundo
bloco a partir daí. Só o primeiro é usado, e dele só os registros de
*Homo sapiens*.

**Unificação de nomes (passo 05).** O mesmo gene aparece com grafias diferentes
(`MALAT1`, `MALAT-1`, `Malat1`, `LncRNA MALAT1`). Sem unificá-las, a base teria
instâncias duplicadas, com atributos idênticos e rótulos diferentes, que
vazariam entre os folds da validação cruzada. A regra ignora caixa, hífen,
espaço e o prefixo descritivo `lncRNA`, e o representante de cada grupo é a
grafia com mais associações. As 213 unificações estão em
`data/processado/06_base1_fusoes.csv`.

**Remoções (passo 05).** Saem identificadores de roedor rotulados como humanos,
entradas que nomeiam mais de um gene, microRNAs, identificadores de transcrito
sem símbolo e colunas que não são doenças humanas. A lista, com o motivo de
cada remoção, está em `data/processado/06_base1_removidos.csv`.

**Sequência representativa (passos 02 e 03).** O passo 02 só aceita transcritos
não codificantes (`NR_` e `XR_`), descartando mRNAs (`NM_`, `XM_`) e registros
genômicos. Como um gene tem várias isoformas, o passo 03 usa o ncRNA curado
(`NR_`) antes do predito (`XR_`) e, entre eles, o de menor accession, que
corresponde ao transcrito originalmente descrito.

**Atributos (passo 03).** As frequências de k = 1 e de k = 2 são normalizadas
separadamente, de modo que cada bloco soma 1. Caracteres ambíguos (N, R, Y...)
são descartados antes da contagem.

**Agrupamento (passo 06b).** As 422 doenças são agrupadas por sistema
fisiológico, por regras sobre o nome da doença. Neoplasias e não neoplasias do
mesmo órgão ficam juntas: uma primeira versão, que separava os cânceres por
órgão, transformava o estudo, na prática, num estudo sobre câncer.
Só viram rótulo os sistemas com pelo menos 30 lncRNAs positivos.

**Corte de instâncias (passo 06b).** Entram os lncRNAs com pelo menos três
doenças associadas e com sequência recuperada: 483.

## Requisitos adicionais

O pipeline usa as dependências do `requirements.txt` e, para ler a planilha
bruta, o `openpyxl`. O `biopython` só é necessário com `--baixar`, para
consultar o NCBI:

```bash
pip install openpyxl biopython
```

O NCBI exige um e-mail de contato em cada consulta. Informe-o com
`--email voce@exemplo.com` ou pela variável de ambiente `NCBI_EMAIL`; ele nunca
fica gravado no código.
