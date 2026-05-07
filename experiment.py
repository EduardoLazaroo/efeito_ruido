import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
from skimage.util import random_noise
from skimage import img_as_float
import os
from tqdm import tqdm
import cv2
from PIL import Image

# Criar dataset de exemplo com imagens geradas aleatoriamente
class SyntheticImageDataset(Dataset):
    def __init__(self, num_images=10, size=224, transform=None):
        self.num_images = num_images
        self.size = size
        self.transform = transform
    
    def __len__(self):
        return self.num_images
    
    def __getitem__(self, idx):
        # Criar imagem sintética
        img = np.random.randint(0, 256, (self.size, self.size, 3), dtype=np.uint8)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, idx % 10  # Labels de 0 a 9

# Configurações
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()
model.to(device)

# Grad-CAM
target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

# Transformações
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset
dataset = SyntheticImageDataset(num_images=10, size=224, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Funções de ruído
def apply_gaussian_noise(img, var):
    img_np = img_as_float(img.permute(1, 2, 0).numpy())
    noisy = random_noise(img_np, mode='gaussian', var=var)
    return torch.tensor(noisy).permute(2, 0, 1)

def apply_salt_pepper_noise(img, amount):
    img_np = img_as_float(img.permute(1, 2, 0).numpy())
    noisy = random_noise(img_np, mode='s&p', amount=amount)
    return torch.tensor(noisy).permute(2, 0, 1)

def apply_speckle_noise(img, var):
    img_np = img_as_float(img.permute(1, 2, 0).numpy())
    noisy = random_noise(img_np, mode='speckle', var=var)
    return torch.tensor(noisy).permute(2, 0, 1)

# Níveis de ruído
noise_levels = {
    'low': {'gaussian': 0.0005, 's&p': 0.01, 'speckle': 0.0005},
    'medium': {'gaussian': 0.005, 's&p': 0.05, 'speckle': 0.005},
    'high': {'gaussian': 0.01, 's&p': 0.1, 'speckle': 0.01}
}

# Funções de métricas
def cosine_similarity(map1, map2):
    map1_flat = map1.flatten()
    map2_flat = map2.flatten()
    return np.dot(map1_flat, map2_flat) / (np.linalg.norm(map1_flat) * np.linalg.norm(map2_flat))

def iou(map1, map2, threshold=0.5):
    map1_bin = (map1 > threshold).astype(np.uint8)
    map2_bin = (map2 > threshold).astype(np.uint8)
    intersection = np.logical_and(map1_bin, map2_bin).sum()
    union = np.logical_or(map1_bin, map2_bin).sum()
    return intersection / union if union > 0 else 0

def spatial_entropy(map):
    map_norm = map / map.sum() if map.sum() > 0 else map
    return -np.sum(map_norm * np.log(map_norm + 1e-10))

# Diretórios
os.makedirs('results/maps', exist_ok=True)
os.makedirs('results/plots', exist_ok=True)

# Experimento
num_images = 10  # Para teste, usar poucas imagens
results = []

for i, (img, label) in enumerate(tqdm(dataloader, total=num_images)):
    if i >= num_images:
        break
    img = img.to(device)

    # Grad-CAM original
    targets = [ClassifierOutputTarget(label.item())]
    grayscale_cam_orig = cam(input_tensor=img, targets=targets)[0, :]
    img_np = img.cpu().permute(0, 2, 3, 1).numpy()[0]
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    cam_image_orig = show_cam_on_image(img_np, grayscale_cam_orig, use_rgb=True)

    # Salvar mapa original
    plt.imsave(f'results/maps/original_{i}.png', cam_image_orig)

    for noise_type in ['gaussian', 's&p', 'speckle']:
        for level, params in noise_levels.items():
            if noise_type == 'gaussian':
                noisy_img = apply_gaussian_noise(img.cpu()[0], params[noise_type])
            elif noise_type == 's&p':
                noisy_img = apply_salt_pepper_noise(img.cpu()[0], params[noise_type])
            elif noise_type == 'speckle':
                noisy_img = apply_speckle_noise(img.cpu()[0], params[noise_type])

            noisy_img = noisy_img.unsqueeze(0).to(device)

            # Grad-CAM perturbado
            grayscale_cam_noisy = cam(input_tensor=noisy_img, targets=targets)[0, :]
            img_noisy_np = noisy_img.cpu().permute(0, 2, 3, 1).numpy()[0]
            img_noisy_np = (img_noisy_np - img_noisy_np.min()) / (img_noisy_np.max() - img_noisy_np.min())
            cam_image_noisy = show_cam_on_image(img_noisy_np, grayscale_cam_noisy, use_rgb=True)

            # Salvar mapa perturbado
            plt.imsave(f'results/maps/{noise_type}_{level}_{i}.png', cam_image_noisy)

            # Métricas
            cos_sim = cosine_similarity(grayscale_cam_orig, grayscale_cam_noisy)
            iou_val = iou(grayscale_cam_orig, grayscale_cam_noisy)
            entropy_orig = spatial_entropy(grayscale_cam_orig)
            entropy_noisy = spatial_entropy(grayscale_cam_noisy)

            results.append({
                'image': i,
                'noise_type': noise_type,
                'level': level,
                'cosine_sim': cos_sim,
                'iou': iou_val,
                'entropy_orig': entropy_orig,
                'entropy_noisy': entropy_noisy
            })

# Salvar resultados
import pandas as pd
df = pd.DataFrame(results)
df.to_csv('results/metrics.csv', index=False)

# Gráficos
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Cosseno
df.boxplot(column='cosine_sim', by=['noise_type', 'level'], ax=axes[0])
axes[0].set_title('Similaridade de Cosseno')
axes[0].set_ylabel('Similaridade')

# IoU
df.boxplot(column='iou', by=['noise_type', 'level'], ax=axes[1])
axes[1].set_title('IoU')

# Entropia
df_melt = df.melt(id_vars=['noise_type', 'level'], value_vars=['entropy_orig', 'entropy_noisy'], var_name='type', value_name='entropy')
df_melt.boxplot(column='entropy', by=['noise_type', 'level', 'type'], ax=axes[2])
axes[2].set_title('Entropia Espacial')

plt.tight_layout()
plt.savefig('results/plots/metrics_boxplot.png')
plt.show()

print("Experimento concluído. Resultados salvos em 'results/'.")