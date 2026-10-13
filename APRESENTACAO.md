# Apresentação Profissional: Efeito Ruído

## Identidade e Proposta de Valor

**Efeito Ruído** é uma pesquisa experimental rigorosa em **Explicabilidade de Inteligência Artificial (XAI)** e **Visão Computacional**, avaliando quantitativamente a estabilidade e robustez de mapas de atenção **Grad-CAM** gerados pela arquitetura **ResNet-50** pré-treinada na ImageNet frente a três classes de perturbações estocásticas: **Ruído Gaussiano**, **Sal e Pimenta** e **Speckle** sob níveis controlados de severidade ($\sigma = [0.05, 0.10, 0.20]$).

Diferente de abordagens qualitativas meramente visuais, o projeto fundamenta-se em formulações matemáticas objetivas: **Similaridade de Cosseno**, **Interseção sobre União (IoU binarizado por Otsu)** e **Entropia de Informação Espacial de Shannon**, submetendo as variações a testes de hipóteses estatísticas (**ANOVA de uma via**).

---

## Texto Sugerido para Publicação (LinkedIn / Portfólio)

> *Quão estáveis são as explicações visuais de uma Rede Neural Convolucional quando a imagem de entrada sofre ruído ambiental ou interferência de sensor?*
>
> Desenvolvi e executei o estudo experimental **Efeito Ruído**, focado em quantificar a robustez de mapas **Grad-CAM** extraídos da última camada convolucional de uma **ResNet-50**. Submetendo imagens a ruídos Gaussiano, Sal & Pimenta e Speckle em três intensidades controladas, avaliamos o impacto sobre 45 execuções instrumentadas.
>
> **Principais Descobertas:**
> 1. **Sensibilidade Diferenciada:** O ruído de Sal e Pimenta causou a degradação mais severa nas atenções visuais, provocando uma queda de **16,77% no IoU** entre os mapas de ativação originais e perturbados, dispersando a atenção da rede.
> 2. **Validação Estatística:** A análise de variância (ANOVA) confirmou diferença estatisticamente significante entre os tipos de ruído ($p < 0.001$).
> 3. **Similaridade Global vs. Local:** Enquanto a similaridade de cosseno global permaneceu alta ($0.9946 \pm 0.007$), o IoU das regiões focais revelou grande sensibilidade ($0.9092 \pm 0.076$), demonstrando que métricas globais podem mascarar desvios críticos em sistemas de apoio à decisão (como diagnóstico médico ou direção autônoma).
>
> Todos os scripts de experimentação, análise estatística e evidências gráficas estão versionados com reprodutibilidade total!

---

## Evidências Reais Produzidas e Validadas

Todas as evidências foram geradas via execução real do pipeline (`experiment_optimized.py` e `analysis.py`):
1. **[evidencias/degradacao_metricas_por_ruido.png](file:///d:/projetos/efeito_ruido/evidencias/degradacao_metricas_por_ruido.png)**: Curvas de decaimento de Cosseno, IoU e Variação de Entropia em função dos níveis de ruído.
2. **[evidencias/distribuicao_iou_boxplot.png](file:///d:/projetos/efeito_ruido/evidencias/distribuicao_iou_boxplot.png)**: Boxplot comparando a dispersão de sobreposição espacial entre classes de ruído.
3. **[evidencias/matriz_correlacao_metricas.png](file:///d:/projetos/efeito_ruido/evidencias/matriz_correlacao_metricas.png)**: Matriz de correlação de Pearson entre as métricas.
4. **[evidencias/sumario_estatistico_execucao.png](file:///d:/projetos/efeito_ruido/evidencias/sumario_estatistico_execucao.png)**: Tabela consolidada com médias, desvios e testes de hipótese ANOVA.
5. **[evidencias/log_execucao_real.txt](file:///d:/projetos/efeito_ruido/evidencias/log_execucao_real.txt)**: Log completo com tempo de execução e checkpoints.

---

## Documento Executivo Word (.docx)

O relatório editorial completo foi compilado e está pronto para apresentação:
- **Arquivo**: [Efeito_Ruido_Relatorio_Publicacao.docx](file:///d:/projetos/efeito_ruido/Efeito_Ruido_Relatorio_Publicacao.docx)
- **Estrutura**: Resumo executivo, formulações matemáticas, arquitetura do pipeline, análise estatística aprofundada, 4 figuras de alta resolução incorporadas com legendas técnicas e considerações para sistemas críticos em produção.

