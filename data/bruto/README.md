# Planilha bruta da LncRNADisease v3.0

Esta pasta recebe o arquivo baixado da LncRNADisease v3.0. Ele **não é
versionado** neste repositório: é um dado de terceiros, e quem o usa deve obtê-lo
na fonte oficial.

Ele só é necessário para rodar o passo 01 do pipeline. A saída desse passo de
que os demais dependem, `data/processado/01_pares_unicos.csv`, já está
publicada, e sem a planilha o pipeline segue a partir dela, com o mesmo
resultado.

## Como obter

1. Acesse a LncRNADisease v3.0 (Lin et al., *Nucleic Acids Research*, 2024) e
   baixe a planilha de associações lncRNA-doença validadas experimentalmente.
2. Coloque o arquivo `.xlsx` nesta pasta. O nome não importa, desde que haja
   **uma única** planilha aqui.
3. Rode o pipeline a partir da raiz do repositório:

```bash
python -m scripts.pipeline_dados.rodar_pipeline
```

## Versão usada na construção da base

| | |
| --- | --- |
| recorte | o passo 01 usa os registros até o ID 12881, que formam o bloco da categoria `LncRNA` |
| registros humanos após o filtro | 12.452 |
| SHA-256 do arquivo | `52ef6545f9f9a2d389ddae18f83baf9423d67588261fd4a9eb70e807cd5011b5` |

Para conferir se o arquivo baixado é o mesmo:

```bash
python -c "import hashlib,glob; f=glob.glob('data/bruto/*.xlsx')[0]; print(hashlib.sha256(open(f,'rb').read()).hexdigest())"
```

Se o hash for outro, a base pode ter sido atualizada desde a construção deste
trabalho, e o passo 01 pode produzir registros diferentes. Nesse caso, os
arquivos de `data/processado/` continuam sendo o registro exato da versão usada,
e é a partir deles que a base do artigo é reproduzida.
