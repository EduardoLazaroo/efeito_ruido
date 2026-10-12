# Apresentacao profissional

## Identidade e valor

Efeito Ruido avalia a robustez de mapas Grad-CAM diante de ruido gaussiano, sal e pimenta e speckle. O experimento usa ResNet-50/ImageNet, metricas de similaridade cosseno, IoU e entropia espacial, com scripts separados para execucao, analise e visualizacao.

## Texto para publicacao

> Neste estudo, investi como perturbacoes aparentemente simples podem alterar a explicacao visual produzida por um modelo de classificacao. O experimento aplica tres tipos de ruido a imagens e compara os mapas Grad-CAM resultantes com metricas de similaridade, sobreposicao e distribuicao espacial. O desafio foi tratar a robustez da explicacao como um objeto mensuravel, e nao apenas como uma inspecao visual. Os resultados documentados devem ser interpretados no recorte experimental adotado, com poucas imagens, mas o pipeline cria uma base clara para ampliar a avaliacao de confiabilidade em XAI.

## Evidencias visuais recomendadas

1. Imagem original e versoes com ruido.
2. Pares de mapas Grad-CAM.
3. Curvas de degradacao das metricas.
4. Distribuicoes de IoU e similaridade.

## Prompt para imagem complementar

```text
Figura cientifica sobre robustez de explicabilidade: imagem original de um objeto, tres versoes com ruido gaussiano, sal e pimenta e speckle, cada uma com mapa Grad-CAM sobreposto; ao lado, curvas de similaridade e IoU; estilo de artigo de visao computacional, fundo branco, sem logotipos.
```
