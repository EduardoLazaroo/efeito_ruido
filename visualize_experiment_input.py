import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import argparse
import cv2


BASE_DIR = Path('results')
VIS_DIR = BASE_DIR / 'visualization'
SYNTH_DIR = VIS_DIR / 'synthetic_images'
NOISE_DIR = VIS_DIR / 'noise_comparison'
GRADCAM_DIR = VIS_DIR / 'gradcam_comparison'
METRICS_DIR = VIS_DIR / 'metrics'
STATS_DIR = VIS_DIR / 'statistics'
PUB_DIR = VIS_DIR / 'publication'

NOISE_CONFIGS = {
    'gaussian': {'label': 'Gaussiano', 'param': 'σ', 'values': {'low': 0.01, 'medium': 0.05, 'high': 0.1}},
    'salt_pepper': {'label': 'Sal-e-Pimenta', 'param': 'amount', 'values': {'low': 0.02, 'medium': 0.05, 'high': 0.1}},
    'speckle': {'label': 'Speckle', 'param': 'σ', 'values': {'low': 0.01, 'medium': 0.05, 'high': 0.1}}
}
LEVEL_LABELS = {'low': 'Baixo', 'medium': 'Médio', 'high': 'Alto'}
LEVEL_ORDER = ['low', 'medium', 'high']
IMAGE_SIZE = (224, 224)
SEED = 42


def create_dirs():
    for directory in [SYNTH_DIR, NOISE_DIR, GRADCAM_DIR, METRICS_DIR, STATS_DIR, PUB_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_metrics():
    csv_path = BASE_DIR / 'metrics.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f'Carregando métricas existentes: {csv_path}')
    else:
        df = pd.DataFrame()
        print('Métricas não encontradas. Será gerado novo conjunto a partir das imagens sintéticas.')
    return df


def create_sample_images(num_images=5, size=IMAGE_SIZE, seed=SEED):
    rng = np.random.default_rng(seed)
    images = []
    for idx in range(num_images):
        background = np.full(size + (3,), 35, dtype=np.uint8)
        circle = np.full(size + (3,), 240, dtype=np.uint8)
        y, x = np.ogrid[-1:1:size[0]*1j, -1:1:size[1]*1j]
        mask = x**2 + y**2 <= 0.25
        img = background.copy()
        img[mask] = circle[mask]
        images.append(img)
    return images


def save_image(array, path):
    Image.fromarray(array).save(path)


def load_existing_synthetic_images():
    paths = sorted(SYNTH_DIR.glob('original_*.png'))
    images = []
    for path in paths:
        images.append(np.array(Image.open(path).convert('RGB')))
    return images


def ensure_synthetic_images(num_images=5):
    existing = load_existing_synthetic_images()
    if len(existing) >= num_images:
        images = existing[:num_images]
    else:
        images = create_sample_images(num_images)
        for idx, img in enumerate(images):
            save_image(img, SYNTH_DIR / f'original_{idx}.png')
    names = [f'original_{idx}' for idx in range(len(images))]
    return images, names


def is_image_file(path):
    return Path(path).suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}


def load_image_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Arquivo não encontrado: {path}')
    img = Image.open(path).convert('RGB')
    img = img.resize(IMAGE_SIZE, Image.LANCZOS)
    return np.array(img)


def load_images_from_dir(dir_path, max_images=5):
    dir_path = Path(dir_path)
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f'Diretório não encontrado: {dir_path}')
    files = sorted([p for p in dir_path.iterdir() if p.is_file() and is_image_file(p)])
    images = []
    names = []
    for path in files[:max_images]:
        images.append(load_image_file(path))
        names.append(path.stem)
    return images, names


def add_gaussian_noise(img, sigma, rng):
    noisy = img.astype(np.float32) / 255.0
    noisy = noisy + rng.normal(0, sigma, noisy.shape)
    noisy = np.clip(noisy, 0, 1)
    return (noisy * 255).astype(np.uint8)


def add_salt_pepper_noise(img, amount, rng):
    noisy = img.copy()

    h, w, _ = noisy.shape
    total_pixels = h * w

    num_salt = int(amount * total_pixels)
    num_pepper = int(amount * total_pixels)

    # ==========================
    # SALT (branco)
    # ==========================
    ys = rng.integers(0, h, num_salt)
    xs = rng.integers(0, w, num_salt)

    noisy[ys, xs] = 255

    # ==========================
    # PEPPER (preto)
    # ==========================
    ys = rng.integers(0, h, num_pepper)
    xs = rng.integers(0, w, num_pepper)

    noisy[ys, xs] = 0

    return noisy


def add_speckle_noise(img, sigma, rng):
    noisy = img.astype(np.float32) / 255.0
    noise = rng.normal(0, sigma, noisy.shape)
    noisy = noisy * (1 + noise)
    noisy = np.clip(noisy, 0, 1)
    return (noisy * 255).astype(np.uint8)


def get_noise_image(image, noise_type, level, rng):
    param = NOISE_CONFIGS[noise_type]['values'][level]
    if noise_type == 'gaussian':
        return add_gaussian_noise(image, param, rng)
    if noise_type == 'salt_pepper':
        return add_salt_pepper_noise(image, param, rng)
    if noise_type == 'speckle':
        return add_speckle_noise(image, param, rng)
    raise ValueError(f'Ruído desconhecido: {noise_type}')


def prepare_model(device):
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()
    model.to(device)
    return model


def image_to_tensor(image, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(Image.fromarray(image)).unsqueeze(0).to(device)
    return tensor


def normalize_image(image):
    image_float = image.astype(np.float32) / 255.0
    return np.clip(image_float, 0, 1)


def compute_cam(model, input_tensor, target_layer, target_class=None):
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        if target_class is None:
            output = model(input_tensor)
            target_class = output.argmax(dim=1).item()
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_class)])[0]
    return grayscale_cam


def render_heatmap(cam, image):
    image_norm = normalize_image(image)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = np.clip(heatmap * 0.5 + image_norm * 0.5, 0, 1)
    return heatmap, overlay


def cosine_similarity(map1, map2):
    a = map1.flatten()
    b = map2.flatten()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def intersection_over_union(map1, map2):
    threshold1 = np.percentile(map1, 50)
    threshold2 = np.percentile(map2, 50)
    bin1 = map1 > threshold1
    bin2 = map2 > threshold2
    intersection = np.logical_and(bin1, bin2).sum()
    union = np.logical_or(bin1, bin2).sum()
    return float(intersection / (union + 1e-9))


def spatial_entropy(map_arr):
    flat = np.abs(map_arr).flatten().astype(np.float64)
    total = flat.sum()
    if total <= 0:
        return 0.0
    prob = flat / total
    prob = prob[prob > 0]
    return float(-(prob * np.log(prob)).sum())


def save_fig(fig, path_root, dpi=300):
    png_path = f'{path_root}.png'
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return png_path


def plot_image_panel(images, names, title='Painel de Imagens Originais'):
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4), constrained_layout=True)
    if len(images) == 1:
        axes = [axes]
    for img, ax, name in zip(images, axes, names):
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(name, fontsize=12, weight='bold')
    fig.suptitle(title, fontsize=16, weight='bold')
    return save_fig(fig, SYNTH_DIR / 'original_gallery')


def plot_noise_panel(image, image_idx, image_label, noise_images, noise_params):
    n_types = len(noise_images)
    fig, axes = plt.subplots(n_types, 4, figsize=(20, 5 * n_types), constrained_layout=True)
    for row_idx, noise_type in enumerate(NOISE_CONFIGS):
        variant_images = [noise_images[noise_type]['original']] + [noise_images[noise_type][lvl] for lvl in LEVEL_ORDER]
        for col_idx, (level_key, img) in enumerate(zip(['original'] + LEVEL_ORDER, variant_images)):
            ax = axes[row_idx][col_idx] if n_types > 1 else axes[col_idx]
            ax.imshow(img)
            ax.axis('off')
            if level_key == 'original':
                title = 'Original'
            else:
                label = LEVEL_LABELS[level_key]
                param = noise_params[noise_type]['values'][level_key]
                title = f'{label}\n{NOISE_CONFIGS[noise_type]["label"]}\n{NOISE_CONFIGS[noise_type]["param"]}={param}'
            ax.set_title(title, fontsize=11)
    fig.suptitle(f'Aplicação de Ruídos na Imagem {image_label}', fontsize=16, weight='bold')
    return save_fig(fig, NOISE_DIR / f'noise_gallery_{image_label}')


def plot_gradcam_comparison(image_idx, image_label, image, original_cam, noisy_image, noisy_cam, metrics, noise_type, level):
    fig, axes = plt.subplots(1, 5, figsize=(22, 5), constrained_layout=True)
    titles = [
        'Imagem Original',
        'Grad-CAM Original',
        f'Imagem com Ruído\n{NOISE_CONFIGS[noise_type]["label"]} ({LEVEL_LABELS[level]})',
        'Grad-CAM com Ruído',
        'Overlay Comparativo'
    ]
    overlay_heatmap, overlay_image = render_heatmap(noisy_cam, noisy_image)
    panels = [image, original_cam, noisy_image, noisy_cam, overlay_image]
    for ax, panel, title in zip(axes, panels, titles):
        if panel.ndim == 2:
            ax.imshow(panel, cmap='inferno')
        else:
            ax.imshow(panel)
        ax.set_title(title, fontsize=11, weight='bold')
        ax.axis('off')
    metric_text = (
        f'Cosine Similarity: {metrics["cosine_similarity"]:.4f}\n'
        f'IoU: {metrics["iou"]:.4f}\n'
        f'Entropia Original: {metrics["entropy_original"]:.4f}\n'
        f'Entropia Ruído: {metrics["entropy_perturbed"]:.4f}'
    )
    fig.text(0.5, 0.01, metric_text, fontsize=11, ha='center', va='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    fig.suptitle(f'Comparação Grad-CAM - Imagem {image_label} / {NOISE_CONFIGS[noise_type]["label"]} / {LEVEL_LABELS[level]}', fontsize=16, weight='bold')
    path_root = GRADCAM_DIR / f'gradcam_{image_label}_{noise_type}_{level}'
    return save_fig(fig, path_root)


def plot_global_statistics(df):
    if df.empty:
        return []
    saved_paths = []

    fig, ax = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    sns.boxplot(data=df, x='noise_type', y='cosine_similarity', hue='noise_level', ax=ax[0])
    ax[0].set_title('Similaridade de Cosseno por Ruído e Nível')
    ax[0].set_xlabel('Ruído')
    ax[0].set_ylabel('Cosine Similarity')
    sns.boxplot(data=df, x='noise_type', y='iou', hue='noise_level', ax=ax[1])
    ax[1].set_title('IoU por Ruído e Nível')
    ax[1].set_xlabel('Ruído')
    ax[1].set_ylabel('IoU')
    sns.boxplot(data=df, x='noise_type', y='entropy_perturbed', hue='noise_level', ax=ax[2])
    ax[2].set_title('Entropia na Imagem Perturbada')
    ax[2].set_xlabel('Ruído')
    ax[2].set_ylabel('Entropia')
    saved_paths.append(save_fig(fig, STATS_DIR / 'boxplots_metrics'))

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    order = LEVEL_ORDER
    for noise_type in df['noise_type'].unique():
        subset = df[df['noise_type'] == noise_type].groupby('noise_level')[['cosine_similarity', 'iou']].mean().reindex(order)
        ax.plot(order, subset['cosine_similarity'], marker='o', label=f'{NOISE_CONFIGS[noise_type]["label"]} - Cosine')
        ax.plot(order, subset['iou'], marker='s', linestyle='--', label=f'{NOISE_CONFIGS[noise_type]["label"]} - IoU')
    ax.set_title('Tendência de Métricas por Nível de Ruído')
    ax.set_xlabel('Nível de Ruído')
    ax.set_ylabel('Valor Médio')
    ax.set_xticks(order)
    ax.set_xticklabels([LEVEL_LABELS[k] for k in order])
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    saved_paths.append(save_fig(fig, STATS_DIR / 'lineplot_metrics'))

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)
    sns.barplot(data=df, x='noise_type', y='cosine_similarity', hue='noise_level', ci=95, ax=axes[0])
    axes[0].set_title('Média de Cosine Similarity com IC 95%')
    axes[0].set_xlabel('Ruído')
    axes[0].set_ylabel('Cosine Similarity')
    sns.barplot(data=df, x='noise_type', y='iou', hue='noise_level', ci=95, ax=axes[1])
    axes[1].set_title('Média de IoU com IC 95%')
    axes[1].set_xlabel('Ruído')
    axes[1].set_ylabel('IoU')
    saved_paths.append(save_fig(fig, STATS_DIR / 'barplot_metrics'))
    sns.violinplot(data=df, x='noise_type', y='cosine_similarity', hue='noise_level', split=True, ax=axes[0])
    axes[0].set_title('Distribuição de Cosine Similarity')
    sns.violinplot(data=df, x='noise_type', y='iou', hue='noise_level', split=True, ax=axes[1])
    axes[1].set_title('Distribuição de IoU')
    saved_paths.append(save_fig(fig, STATS_DIR / 'violin_metrics'))

    return saved_paths


def plot_publication_figure(images, names, df):
    image = images[0]
    image_label = names[0]
    rng = np.random.default_rng(SEED)
    noise_variant = {}
    cams = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = prepare_model(device)
    target_layer = model.layer4[-1]
    original_tensor = image_to_tensor(image, device)
    original_cam = compute_cam(model, original_tensor, target_layer)
    original_cam = cv2.resize(original_cam, IMAGE_SIZE)
    for noise_type in NOISE_CONFIGS:
        noisy = get_noise_image(image, noise_type, 'medium', rng)
        noisy_tensor = image_to_tensor(noisy, device)
        noisy_cam = compute_cam(model, noisy_tensor, target_layer)
        noisy_cam = cv2.resize(noisy_cam, IMAGE_SIZE)
        noise_variant[noise_type] = noisy
        cams[noise_type] = noisy_cam
    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 4)
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(image)
    ax_orig.set_title('Imagem Original', fontsize=12, weight='bold')
    ax_orig.axis('off')
    ax_orig_cam = fig.add_subplot(gs[1, 0])
    ax_orig_cam.imshow(original_cam, cmap='inferno')
    ax_orig_cam.set_title('Grad-CAM Original', fontsize=12, weight='bold')
    ax_orig_cam.axis('off')
    column = 1
    for noise_type in NOISE_CONFIGS:
        ax_img = fig.add_subplot(gs[0, column])
        ax_img.imshow(noise_variant[noise_type])
        ax_img.set_title(f'{NOISE_CONFIGS[noise_type]["label"]} - Médio', fontsize=12, weight='bold')
        ax_img.axis('off')
        ax_cam = fig.add_subplot(gs[1, column])
        ax_cam.imshow(cams[noise_type], cmap='inferno')
        ax_cam.set_title(f'Grad-CAM {NOISE_CONFIGS[noise_type]["label"]}', fontsize=12, weight='bold')
        ax_cam.axis('off')
        over = render_heatmap(cams[noise_type], noise_variant[noise_type])[1]
        ax_ov = fig.add_subplot(gs[2, column])
        ax_ov.imshow(over)
        ax_ov.set_title(f'Overlay {NOISE_CONFIGS[noise_type]["label"]}', fontsize=12, weight='bold')
        ax_ov.axis('off')
        column += 1
    avg_metrics = df.groupby('noise_type')[['cosine_similarity', 'iou', 'entropy_perturbed']].mean().reset_index()
    metrics_text = 'Resumo de Métricas por Tipo de Ruído:\n'
    for _, row in avg_metrics.iterrows():
        metrics_text += f"{NOISE_CONFIGS[row['noise_type']]['label']}: Cos={row['cosine_similarity']:.4f}, IoU={row['iou']:.4f}, Entropia={row['entropy_perturbed']:.4f}\n"
    fig.text(0.02, 0.02, metrics_text, fontsize=12, va='bottom', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    fig.suptitle(f'Figura de Publicação: Robustez Grad-CAM - {image_label}', fontsize=18, weight='bold')
    return save_fig(fig, PUB_DIR / f'publication_main_{image_label}')


def main(num_images=5, input_image=None, input_dir=None):
    create_dirs()
    df = load_metrics()

    # ==========================================================
    # DEFINIÇÃO DA IMAGEM DE ENTRADA
    # ==========================================================
    if input_image:
        images = [load_image_file(input_image)]
        names = [Path(input_image).stem]

    elif input_dir:
        images, names = load_images_from_dir(
            input_dir,
            max_images=num_images
        )

    else:
        # usa automaticamente a imagem real da raiz do projeto
        input_image = 'test_image.jpeg'
        images = [load_image_file(input_image)]
        names = [Path(input_image).stem]

    plot_image_panel(
        images,
        names,
        title='Painel de Imagens Originais'
    )

    rng = np.random.default_rng(SEED)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    model = prepare_model(device)
    target_layer = model.layer4[-1]

    if df.empty:
        df = pd.DataFrame(columns=[
            'image_idx',
            'noise_type',
            'noise_level',
            'cosine_similarity',
            'iou',
            'entropy_original',
            'entropy_perturbed'
        ])

    # ==========================================================
    # LOOP PRINCIPAL
    # ==========================================================
    for image_idx, (image, image_label) in enumerate(
        zip(images[:num_images], names)
    ):

        # ==========================================
        # Grad-CAM original calculado APENAS 1x
        # ==========================================
        img_tensor = image_to_tensor(
            image,
            device
        )

        original_cam = compute_cam(
            model,
            img_tensor,
            target_layer
        )

        original_cam = cv2.resize(
            original_cam,
            IMAGE_SIZE
        )

        noise_images = {
            'gaussian': {'original': image},
            'salt_pepper': {'original': image},
            'speckle': {'original': image}
        }

        current_metrics = []

        # ==========================================
        # Aplicação dos ruídos
        # ==========================================
        for noise_type in NOISE_CONFIGS:

            for level in LEVEL_ORDER:

                noisy_image = get_noise_image(
                    image,
                    noise_type,
                    level,
                    rng
                )

                noise_images[noise_type][level] = noisy_image

                # Grad-CAM apenas da imagem perturbada
                noisy_tensor = image_to_tensor(
                    noisy_image,
                    device
                )

                noisy_cam = compute_cam(
                    model,
                    noisy_tensor,
                    target_layer
                )

                noisy_cam = cv2.resize(
                    noisy_cam,
                    IMAGE_SIZE
                )

                metrics = {
                    'image_idx': image_idx,
                    'image_label': image_label,
                    'noise_type': noise_type,
                    'noise_level': level,
                    'cosine_similarity': cosine_similarity(
                        original_cam,
                        noisy_cam
                    ),
                    'iou': intersection_over_union(
                        original_cam,
                        noisy_cam
                    ),
                    'entropy_original': spatial_entropy(
                        original_cam
                    ),
                    'entropy_perturbed': spatial_entropy(
                        noisy_cam
                    ),
                }

                current_metrics.append(
                    metrics
                )

                plot_gradcam_comparison(
                    image_idx,
                    image_label,
                    image,
                    original_cam,
                    noisy_image,
                    noisy_cam,
                    metrics,
                    noise_type,
                    level
                )

        plot_noise_panel(
            image,
            image_idx,
            image_label,
            noise_images,
            NOISE_CONFIGS
        )

        df = pd.concat(
            [
                df,
                pd.DataFrame(current_metrics)
            ],
            ignore_index=True
        )

    # ==========================================================
    # SALVAMENTO
    # ==========================================================
    metrics_path = (
        VIS_DIR /
        'metrics' /
        'visualization_metrics.csv'
    )

    df.to_csv(
        metrics_path,
        index=False
    )

    print(
        f'✓ Métricas de visualização salvas em: {metrics_path}'
    )

    stats_paths = plot_global_statistics(df)

    if stats_paths:
        for p in stats_paths:
            print(
                f'✓ Estatística salva em: {p}'
            )

    plot_publication_figure(
        images[:num_images],
        names[:num_images],
        df
    )

    print(
        '✓ Visualizações concluídas em:',
        VIS_DIR
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gerar visualizações científicas de Grad-CAM e ruído.')
    parser.add_argument('--max-images', type=int, default=5, help='Número máximo de imagens a processar')
    parser.add_argument('--input-image', type=str, default=None, help='Caminho para uma imagem real a ser analisada')
    parser.add_argument('--input-dir', type=str, default=None, help='Diretório de imagens reais para análise')
    args = parser.parse_args()
    if args.input_image and args.input_dir:
        raise ValueError('Use apenas --input-image ou --input-dir, não ambos.')
    main(num_images=args.max_images, input_image=args.input_image, input_dir=args.input_dir)
