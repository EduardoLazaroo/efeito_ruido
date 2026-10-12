# Efeito do Ruido na Robustez de Mapas Grad-CAM

## Visao geral

Experimento academico que investiga como perturbacoes estocasticas afetam mapas de explicabilidade Grad-CAM produzidos por uma ResNet-50. O projeto compara ruido gaussiano, sal e pimenta e speckle com metricas de similaridade cosseno, Intersection over Union (IoU) e entropia espacial.

O objetivo e didatico e experimental: apoiar a discussao sobre estabilidade de explicacoes visuais em visao computacional.

## Estrutura verificada

| Recurso | Responsabilidade |
|---|---|
| `experiment_optimized.py` | execucao otimizada do experimento |
| `analysis.py` | analise dos resultados disponiveis |
| scripts de visualizacao | geracao de visualizacoes |
| `requirements.txt` | dependencias Python |
| `data/cifar-10-python.tar.gz` | arquivo local de dados |
| `test_image.jpeg` | imagem de entrada disponivel |
| `RELATORIO_SIMPLIFICADO.md` | material de apoio do estudo |

Arquivos e pastas anteriormente descritos como `experiment.py`, `results/` e `RESULTADOS_EXPERIMENTAIS.md` nao foram confirmados na versao atual do diretorio. Por isso, nao sao tratados nesta documentacao como artefatos distribuidos.

## Instalacao

Requer Python 3.10 ou superior.

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Em Linux ou macOS, adapte o comando de ativacao ao shell utilizado.

## Execucao

```bash
python experiment_optimized.py
```

O primeiro uso pode exigir o download dos pesos da ResNet-50. O tempo de execucao depende do dispositivo, das bibliotecas instaladas e das entradas efetivamente utilizadas pelo script.

## Metodologia

- modelo: ResNet-50 com pesos ImageNet;
- explicacao visual: Grad-CAM, sobre a camada final usada pelo script;
- perturbacoes: gaussiana, sal e pimenta e speckle;
- medidas: similaridade cosseno, IoU e entropia espacial.

Os valores e interpretacoes presentes em relatorios historicos devem ser reproduzidos a partir da versao atual do script antes de serem usados em artigo, TCC ou comparacao cientifica. O recorte experimental e limitado e nao permite generalizar conclusoes para todos os modelos, imagens ou cenarios de aquisicao.

## Limites conhecidos

- nao foi identificada uma suite automatizada de testes;
- resultados gerados nao estao versionados como parte confirmada da estrutura atual;
- a reprodutibilidade exige registrar versoes de bibliotecas, dispositivo e sementes aleatorias.

## Apresentacao profissional

O texto de publicacao, as capturas recomendadas e o prompt de imagem estao em [APRESENTACAO.md](APRESENTACAO.md).
