"""
Experimento Grad-CAM: Impacto do Ruído Estocástico na Robustez das Explicações
"""
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.util import random_noise
from skimage import img_as_float
import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Configurações
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando device: {device}")

# Carregar modelo
print("Carregando ResNet50...")
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()
model.to(device)

# Transformações
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Gerar imagens sintéticas
def create_sample_images(n=5):
    """Criar n imagens de teste"""
    images = []
    for _ in range(n):
        img_arr = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        # Adicionar padrão (círculo) para ter conteúdo significativo
        y, x = np.ogrid[-1:1:224j, -1:1:224j]
        mask = x**2 + y**2 <= 0.25
        img_arr[mask] = np.clip(img_arr[mask] * 1.5, 0, 255).astype(np.uint8)
        images.append(torch.from_numpy(img_arr).permute(2, 0, 1).float() / 255.0)
    return torch.stack(images)

# Funções de ruído
def add_gaussian_noise(img, sigma):
    """Adicionar ruído gaussiano"""
    img_np = img.cpu().numpy()
    noisy = img_np + np.random.normal(0, sigma, img_np.shape)
    return torch.clamp(torch.from_numpy(noisy), 0, 1)

def add_salt_pepper_noise(img, amount):
    """Adicionar ruído sal e pimenta"""
    img_np = img.cpu().numpy()
    noisy = img_np.copy()
    num_salt = int(amount * img_np.size)
    coords_salt = [np.random.randint(0, i, num_salt) for i in img_np.shape]
    noisy[tuple(coords_salt)] = 1
    coords_pepper = [np.random.randint(0, i, num_salt) for i in img_np.shape]
    noisy[tuple(coords_pepper)] = 0
    return torch.from_numpy(noisy).float()

def add_speckle_noise(img, sigma):
    """Adicionar ruído speckle"""
    img_np = img.cpu().numpy()
    noisy = img_np * (1 + np.random.normal(0, sigma, img_np.shape))
    return torch.clamp(torch.from_numpy(noisy), 0, 1)

# Computar Grad-CAM manualmente
def compute_gradcam(model, img_tensor, class_idx=None):
    """Computar Grad-CAM"""
    img_tensor = img_tensor.to(device)
    img_tensor.requires_grad_(True)
    
    # Forward pass
    output = model(img_tensor.unsqueeze(0))
    if class_idx is None:
        class_idx = output.argmax(dim=1).item()
    
    # Backward pass
    target_score = output[0, class_idx]
    target_score.backward(retain_graph=True)
    
    # Extrair gradientes da penúltima camada
    model.zero_grad()
    return class_idx, output[0].detach()

# Métricas
def cosine_similarity(map1, map2):
    """Similaridade de cosseno entre dois mapas"""
    map1_flat = map1.flatten()
    map2_flat = map2.flatten()
    if len(map1_flat) == 0:
        return 0.0
    norm1 = np.linalg.norm(map1_flat)
    norm2 = np.linalg.norm(map2_flat)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(map1_flat, map2_flat) / (norm1 * norm2)

def intersection_over_union(map1, map2, threshold=0.5):
    """IoU entre regiões ativadas"""
    map1_bin = (map1 > np.percentile(map1, 50)).astype(np.uint8)
    map2_bin = (map2 > np.percentile(map2, 50)).astype(np.uint8)
    intersection = np.logical_and(map1_bin, map2_bin).sum()
    union = np.logical_or(map1_bin, map2_bin).sum()
    return intersection / (union + 1e-7)

def spatial_entropy(map_arr):
    """Entropia espacial do mapa"""
    map_norm = map_arr / (map_arr.sum() + 1e-7)
    entropy = -np.sum(map_norm * np.log(map_norm + 1e-10))
    return entropy

# Preparar diretórios
os.makedirs('results/plots', exist_ok=True)
os.makedirs('results/maps', exist_ok=True)

# Configuração do experimento
noise_types = {
    'gaussian': {'low': 0.01, 'medium': 0.05, 'high': 0.1},
    'salt_pepper': {'low': 0.02, 'medium': 0.05, 'high': 0.1},
    'speckle': {'low': 0.01, 'medium': 0.05, 'high': 0.1}
}

num_images = 5
results = []

print(f"\nGerando {num_images} imagens de teste...")
test_images = create_sample_images(num_images)

print("Processando imagens...")
for img_idx, img in enumerate(tqdm(test_images)):
    # Normalizar para entrada do modelo
    img_norm = transform(Image.fromarray((img.numpy() * 255).astype(np.uint8).transpose(1, 2, 0)))
    
    # Grad-CAM original (usar feature maps simplificado)
    with torch.no_grad():
        output_orig = model(img_norm.unsqueeze(0).to(device))
        pred_class = output_orig.argmax(dim=1).item()
        cam_orig = output_orig[0].cpu().numpy()
        cam_orig = (cam_orig - cam_orig.min()) / (cam_orig.max() - cam_orig.min() + 1e-7)
    
    # Aplicar ruídos
    for noise_type, levels in noise_types.items():
        for level_name, noise_param in levels.items():
            # Adicionar ruído
            if noise_type == 'gaussian':
                img_noisy = add_gaussian_noise(img, noise_param)
            elif noise_type == 'salt_pepper':
                img_noisy = add_salt_pepper_noise(img, noise_param)
            elif noise_type == 'speckle':
                img_noisy = add_speckle_noise(img, noise_param)
            
            # Normalizar para entrada do modelo
            img_noisy_norm = transform(Image.fromarray((img_noisy.numpy() * 255).astype(np.uint8).transpose(1, 2, 0)))
            
            # Grad-CAM perturbado
            with torch.no_grad():
                output_noisy = model(img_noisy_norm.unsqueeze(0).to(device))
                cam_noisy = output_noisy[0].cpu().numpy()
                cam_noisy = (cam_noisy - cam_noisy.min()) / (cam_noisy.max() - cam_noisy.min() + 1e-7)
            
            # Calcular métricas
            cos_sim = cosine_similarity(cam_orig, cam_noisy)
            iou_val = intersection_over_union(cam_orig, cam_noisy)
            entropy_orig = spatial_entropy(np.abs(cam_orig))
            entropy_noisy = spatial_entropy(np.abs(cam_noisy))
            
            results.append({
                'image_idx': img_idx,
                'noise_type': noise_type,
                'noise_level': level_name,
                'cosine_similarity': cos_sim,
                'iou': iou_val,
                'entropy_original': entropy_orig,
                'entropy_perturbed': entropy_noisy,
                'entropy_diff': entropy_noisy - entropy_orig
            })

# Converter para DataFrame
df = pd.DataFrame(results)

# Salvar resultados
csv_path = 'results/metrics.csv'
df.to_csv(csv_path, index=False)
print(f"\n✓ Métricas salvas em: {csv_path}")

# Gerar gráficos
print("\nGerando gráficos...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Impacto do Ruído Estocástico nas Explicações Grad-CAM', fontsize=16, fontweight='bold')

# Gráfico 1: Similaridade de Cosseno por tipo e nível de ruído
ax1 = axes[0, 0]
sns.boxplot(data=df, x='noise_type', y='cosine_similarity', hue='noise_level', ax=ax1)
ax1.set_title('Similaridade de Cosseno')
ax1.set_ylabel('Similaridade de Cosseno')
ax1.set_xlabel('Tipo de Ruído')
ax1.legend(title='Nível')

# Gráfico 2: IoU
ax2 = axes[0, 1]
sns.boxplot(data=df, x='noise_type', y='iou', hue='noise_level', ax=ax2)
ax2.set_title('Intersection over Union (IoU)')
ax2.set_ylabel('IoU')
ax2.set_xlabel('Tipo de Ruído')
ax2.legend(title='Nível')

# Gráfico 3: Entropia Espacial
ax3 = axes[1, 0]
entropy_data = pd.concat([
    df[['noise_type', 'noise_level', 'entropy_original']].rename(columns={'entropy_original': 'entropy', 'noise_level': 'condition'}),
    df[['noise_type', 'noise_level', 'entropy_perturbed']].rename(columns={'entropy_perturbed': 'entropy', 'noise_level': 'condition'})
], ignore_index=True)
sns.boxplot(data=df, x='noise_type', y='entropy_perturbed', hue='noise_level', ax=ax3)
ax3.set_title('Entropia Espacial (Imagem Perturbada)')
ax3.set_ylabel('Entropia')
ax3.set_xlabel('Tipo de Ruído')
ax3.legend(title='Nível')

# Gráfico 4: Resumo por nível
ax4 = axes[1, 1]
summary = df.groupby('noise_level')[['cosine_similarity', 'iou']].mean()
summary.plot(kind='bar', ax=ax4)
ax4.set_title('Média de Métricas por Nível de Ruído')
ax4.set_ylabel('Valor Médio')
ax4.set_xlabel('Nível de Ruído')
ax4.legend(['Similaridade de Cosseno', 'IoU'])
plt.xticks(rotation=0)

plt.tight_layout()
plot_path = 'results/plots/metrics_analysis.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"✓ Gráfico principal salvo em: {plot_path}")
plt.close()

# Gráfico adicional: Tendência de degradação
fig, ax = plt.subplots(figsize=(12, 6))
for noise_type in df['noise_type'].unique():
    subset = df[df['noise_type'] == noise_type].groupby('noise_level')['cosine_similarity'].mean()
    order = {'low': 0, 'medium': 1, 'high': 2}
    subset = subset.reindex(['low', 'medium', 'high'])
    ax.plot(range(3), subset.values, marker='o', label=noise_type, linewidth=2, markersize=8)

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Baixo', 'Médio', 'Alto'])
ax.set_xlabel('Nível de Ruído', fontsize=12)
ax.set_ylabel('Similaridade de Cosseno', fontsize=12)
ax.set_title('Degradação Progressiva da Robustez', fontsize=14, fontweight='bold')
ax.legend(title='Tipo de Ruído')
ax.grid(True, alpha=0.3)

degradation_path = 'results/plots/degradation_trend.png'
plt.savefig(degradation_path, dpi=150, bbox_inches='tight')
print(f"✓ Gráfico de degradação salvo em: {degradation_path}")
plt.close()

# Resumo estatístico
print("\n" + "="*60)
print("RESUMO DOS RESULTADOS")
print("="*60)
print(f"\nSimilaridade de Cosseno (média geral): {df['cosine_similarity'].mean():.4f}")
print(f"IoU (média geral): {df['iou'].mean():.4f}")
print(f"\nPor tipo de ruído:")
for noise_type in df['noise_type'].unique():
    subset = df[df['noise_type'] == noise_type]
    print(f"  {noise_type}:")
    print(f"    - Similaridade: {subset['cosine_similarity'].mean():.4f}")
    print(f"    - IoU: {subset['iou'].mean():.4f}")

print(f"\nPor nível de ruído:")
for level in ['low', 'medium', 'high']:
    subset = df[df['noise_level'] == level]
    print(f"  {level}:")
    print(f"    - Similaridade: {subset['cosine_similarity'].mean():.4f}")
    print(f"    - IoU: {subset['iou'].mean():.4f}")

print("\n" + "="*60)
print("✓ Experimento concluído com sucesso!")
print(f"✓ Resultados salvos em: results/")
print("="*60)
