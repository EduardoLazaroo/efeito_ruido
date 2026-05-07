# Projeto: Impacto do Ruído Estocástico na Robustez de Mapas Grad-CAM

## 📋 Descrição

Este projeto implementa um experimento completo em Python para investigar como ruídos estocásticos afetam a robustez das explicações visuais geradas pelo Grad-CAM em redes neurais convolucionais.

**Status**: ✅ Experimento concluído com sucesso

## 📦 Estrutura do Projeto

```
efeito_ruido/
├── artigo.txt                      # Artigo original (teórico)
├── RESULTADOS_EXPERIMENTAIS.md     # Resultados e análise
├── README.md                       # Este arquivo
├── requirements.txt                # Dependências Python
├── experiment.py                   # Script inicial (versão completa)
├── experiment_optimized.py         # Script otimizado (recomendado)
├── results/
│   ├── metrics.csv                # Dados brutos das avaliações
│   ├── plots/
│   │   ├── metrics_analysis.png    # Gráficos principais
│   │   └── degradation_trend.png   # Curva de degradação
│   └── maps/                       # Mapas Grad-CAM gerados
└── .venv/                          # Ambiente virtual Python
```

## 🚀 Como Executar

### 1. Pré-requisitos

- Python 3.10+
- pip (gerenciador de pacotes)
- ~500MB de espaço em disco (para modelo ResNet50)

### 2. Configuração do Ambiente

```bash
# Clone ou navegue para o diretório do projeto
cd efeito_ruido

# Crie um ambiente virtual (opcional mas recomendado)
python -m venv .venv

# Ative o ambiente
# No Windows:
.venv\Scripts\activate
# No Linux/Mac:
source .venv/bin/activate
```

### 3. Instale as Dependências

```bash
pip install -r requirements.txt
```

**Pacotes instalados:**
- `torch>=2.0.0` - Framework de deep learning
- `torchvision>=0.15.0` - Modelos e datasets
- `pytorch-grad-cam>=0.2.0` - Implementação de Grad-CAM
- `numpy` - Computação numérica
- `matplotlib` - Visualização
- `scikit-image` - Processamento de imagens
- `seaborn` - Gráficos estatísticos
- `pandas` - Análise de dados
- `tqdm` - Barra de progresso

### 4. Execute o Experimento

```bash
# Versão otimizada (recomendada, ~30s em CPU)
python experiment_optimized.py

# Versão completa (mais lenta, múltiplas imagens)
python experiment.py
```

## 📊 Resultados

### Métricas Principais

| Tipo de Ruído | Similaridade | IoU | Status |
|---|---|---|---|
| Gaussiano | 0.9983 | 0.9383 | ✅ Robusto |
| Sal-e-Pimenta | 0.9887 | 0.8340 | ⚠️ Moderado |
| Speckle | 0.9992 | 0.9598 | ✅ Muito Robusto |

### Degradação por Nível

- **Nível Baixo**: 0.9983 (linha de base)
- **Nível Médio**: 0.9965 (-0.18%)
- **Nível Alto**: 0.9913 (-0.70%)

## 📈 Visualizações

### Gráfico 1: Análise de Métricas
![Metrics Analysis](results/plots/metrics_analysis.png)

Mostra:
- Distribuição de similaridade de cosseno
- Distribuição de IoU
- Entropia espacial
- Resumo por nível de ruído

### Gráfico 2: Degradação Progressiva
![Degradation Trend](results/plots/degradation_trend.png)

Mostra:
- Queda progressiva de robustez por tipo de ruído
- Sal-e-pimenta com maior degradação (~1.5% a 2.0% por nível)
- Speckle com melhor manutenção de robustez

## 🔬 Metodologia Experimental

### Dataset
- 5 imagens sintéticas de teste
- Tamanho: 224×224 pixels
- Padrões: Aleatoriedade com formas geométricas

### Tipos de Ruído

1. **Gaussiano** (N(0,σ²))
   - Parâmetros: σ ∈ {0.01, 0.05, 0.1}
   - Tipo: Aditivo

2. **Sal-e-Pimenta**
   - Parâmetros: amount ∈ {0.02, 0.05, 0.1}
   - Tipo: Impulsivo

3. **Speckle** (I' = I(1+n))
   - Parâmetros: σ ∈ {0.01, 0.05, 0.1}
   - Tipo: Multiplicativo

### Modelo
- **Arquitetura**: ResNet-50
- **Pesos**: ImageNet ILSVRC-2012
- **Camada analisada**: layer4[-1] (penúltima)

### Métricas

1. **Similaridade de Cosseno**
   ```
   cos(x,y) = (x·y) / (||x|| × ||y||)
   ```
   Intervalo: [0, 1] onde 1 = idêntico

2. **Intersection over Union (IoU)**
   ```
   IoU = |A ∩ B| / |A ∪ B|
   ```
   Intervalo: [0, 1]

3. **Entropia Espacial**
   ```
   H = -Σ p(x) × log(p(x))
   ```
   Mede dispersão do mapa

## 💡 Interpretação

### Achados Principais

1. **Robustez Surpreendente**: Grad-CAM mostra resistência a ruído (~99.5% de similaridade)

2. **Sensibilidade Diferencial**: 
   - Ruído impulsivo (sal-e-pimenta) mais prejudicial
   - Ruído multiplicativo (speckle) melhor tolerado

3. **Degradação em IoU**: Maior que em similaridade (-8.6% vs -0.7%)
   - Indica dispersão espacial de ativações
   - Preservação de magnitude

4. **Implicações Práticas**:
   - Seguro usar Grad-CAM em ambientes com ruído baixo-médio
   - Validar com técnicas complementares em ambientes adversos
   - Maior cuidado com sensores impulsivos

## 🔄 Reprodução dos Resultados

### Sistema Testado
- **OS**: Windows 11
- **Python**: 3.13.13
- **PyTorch**: 2.11.0
- **Device**: CPU (Intel Core i7)
- **Tempo**: ~30 segundos

### Variabilidade Esperada
Resultados podem variar <2% devido a:
- Inicialização aleatória de imagens
- Implementações de GPU
- Versões de bibliotecas

Para maior consistência, defina `seed` no script:
```python
import random
import numpy as np
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

## 📚 Referências

- [Grad-CAM Paper](https://arxiv.org/abs/1610.02055)
- [PyTorch Documentation](https://pytorch.org/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)

## 🎓 Para Seu Artigo/TCC

### Seções Sugeridas

1. **Metodologia** - Descrever este experimento
2. **Resultados** - Incluir gráficos gerados
3. **Discussão** - Interpretar achados
4. **Apêndice** - Código do experimento

### Citação Sugerida
```
Experimento conduzido em Python 3.13 com PyTorch 2.11 e Grad-CAM para
avaliar robustez de explicações visuais sob perturbação estocástica.
Dataset: 5 imagens sintéticas. Métricas: Similaridade de Cosseno, IoU,
Entropia Espacial. ResNet-50 pré-treinado em ImageNet.
```

## 🐛 Troubleshooting

### Erro: "No module named 'pytorch_grad_cam'"
```bash
pip install --upgrade pytorch-gradcam
# Ou reinstale as dependências
pip install -r requirements.txt --force-reinstall
```

### Erro: "CUDA out of memory"
O script detecta automaticamente CPU/GPU. Para forçar CPU:
```python
device = torch.device('cpu')
```

### Script muito lento
Use `experiment_optimized.py` em vez de `experiment.py`
- Menos imagens de teste
- Feature maps simplificados
- Tempo típico: 30s em CPU

### Gráficos não salvam
Crie o diretório manualmente:
```bash
mkdir -p results/plots results/maps
```

## 📝 Licença

Este projeto é fornecido como referência educacional para fins de pesquisa acadêmica.

## 👨‍💼 Autor

Experimento implementado como complemento prático do artigo sobre robustez de técnicas XAI.

---

**Última atualização**: 5 de maio de 2026
**Status do Experimento**: ✅ Concluído e Validado
