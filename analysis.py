"""
Análise Detalhada dos Resultados Experimentais
Gera relatório em texto e visualizações adicionais
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Carregar dados
df = pd.read_csv('results/metrics.csv')

print("=" * 70)
print("ANÁLISE DETALHADA - Impacto do Ruído Estocástico em Grad-CAM")
print("=" * 70)

# 1. Estatísticas Descritivas
print("\n1. ESTATÍSTICAS DESCRITIVAS")
print("-" * 70)

print("\nSimilaridade de Cosseno:")
print(f"  Média: {df['cosine_similarity'].mean():.6f}")
print(f"  Mediana: {df['cosine_similarity'].median():.6f}")
print(f"  Desvio Padrão: {df['cosine_similarity'].std():.6f}")
print(f"  Mín-Máx: [{df['cosine_similarity'].min():.6f}, {df['cosine_similarity'].max():.6f}]")
print(f"  Q1-Q3: [{df['cosine_similarity'].quantile(0.25):.6f}, {df['cosine_similarity'].quantile(0.75):.6f}]")

print("\nIntersection over Union (IoU):")
print(f"  Média: {df['iou'].mean():.6f}")
print(f"  Mediana: {df['iou'].median():.6f}")
print(f"  Desvio Padrão: {df['iou'].std():.6f}")
print(f"  Mín-Máx: [{df['iou'].min():.6f}, {df['iou'].max():.6f}]")
print(f"  Q1-Q3: [{df['iou'].quantile(0.25):.6f}, {df['iou'].quantile(0.75):.6f}]")

print("\nEntropia Espacial (Diferença):")
print(f"  Média: {df['entropy_diff'].mean():.6f}")
print(f"  Mediana: {df['entropy_diff'].median():.6f}")
print(f"  Desvio Padrão: {df['entropy_diff'].std():.6f}")
print(f"  Mín-Máx: [{df['entropy_diff'].min():.6f}, {df['entropy_diff'].max():.6f}]")

# 2. Análise por Tipo de Ruído
print("\n2. ANÁLISE POR TIPO DE RUÍDO")
print("-" * 70)

for noise_type in sorted(df['noise_type'].unique()):
    subset = df[df['noise_type'] == noise_type]
    print(f"\n{noise_type.upper()}:")
    print(f"  Amostras: {len(subset)}")
    print(f"  Similaridade: {subset['cosine_similarity'].mean():.6f} ± {subset['cosine_similarity'].std():.6f}")
    print(f"  IoU: {subset['iou'].mean():.6f} ± {subset['iou'].std():.6f}")
    print(f"  Degradação IoU: {(1 - subset['iou'].mean())*100:.2f}%")

# 3. Análise por Nível de Ruído
print("\n3. ANÁLISE POR NÍVEL DE RUÍDO")
print("-" * 70)

for level in ['low', 'medium', 'high']:
    subset = df[df['noise_level'] == level]
    print(f"\n{level.upper()}:")
    print(f"  Amostras: {len(subset)}")
    print(f"  Similaridade: {subset['cosine_similarity'].mean():.6f} ± {subset['cosine_similarity'].std():.6f}")
    print(f"  IoU: {subset['iou'].mean():.6f} ± {subset['iou'].std():.6f}")

# 4. Teste estatístico (ANOVA)
print("\n4. ANÁLISE ESTATÍSTICA (ANOVA)")
print("-" * 70)

# ANOVA para tipos de ruído
groups_noise = [df[df['noise_type'] == nt]['cosine_similarity'].values 
                for nt in df['noise_type'].unique()]
f_stat_noise, p_value_noise = stats.f_oneway(*groups_noise)

print(f"\nTipo de Ruído (Similaridade de Cosseno):")
print(f"  F-statistic: {f_stat_noise:.4f}")
print(f"  p-value: {p_value_noise:.4f}")
print(f"  Significância: {'SIM (p<0.05)' if p_value_noise < 0.05 else 'NÃO'}")

# ANOVA para níveis
groups_level = [df[df['noise_level'] == lv]['cosine_similarity'].values 
                for lv in df['noise_level'].unique()]
f_stat_level, p_value_level = stats.f_oneway(*groups_level)

print(f"\nNível de Ruído (Similaridade de Cosseno):")
print(f"  F-statistic: {f_stat_level:.4f}")
print(f"  p-value: {p_value_level:.4f}")
print(f"  Significância: {'SIM (p<0.05)' if p_value_level < 0.05 else 'NÃO'}")

# 5. Matriz de Correlação
print("\n5. CORRELAÇÃO ENTRE MÉTRICAS")
print("-" * 70)

corr_matrix = df[['cosine_similarity', 'iou', 'entropy_diff']].corr()
print("\n", corr_matrix)

# 6. Ranking de Robustez
print("\n6. RANKING DE ROBUSTEZ")
print("-" * 70)

rankings = df.groupby('noise_type').agg({
    'cosine_similarity': 'mean',
    'iou': 'mean'
}).sort_values('cosine_similarity', ascending=False)

for idx, (noise_type, row) in enumerate(rankings.iterrows(), 1):
    print(f"{idx}. {noise_type}: {row['cosine_similarity']:.4f} (cosine), {row['iou']:.4f} (IoU)")

# 7. Degradação por Nível (em %)
print("\n7. DEGRADAÇÃO PERCENTUAL POR NÍVEL")
print("-" * 70)

baseline = df[df['noise_level'] == 'low'].groupby('noise_type')['cosine_similarity'].mean()

for level in ['medium', 'high']:
    degraded = df[df['noise_level'] == level].groupby('noise_type')['cosine_similarity'].mean()
    print(f"\n{level.upper()} vs LOW:")
    for noise_type in degraded.index:
        pct = ((baseline[noise_type] - degraded[noise_type]) / baseline[noise_type]) * 100
        print(f"  {noise_type}: {pct:.2f}% degradação")

# 8. Casos Extremos
print("\n8. CASOS EXTREMOS")
print("-" * 70)

best_idx = df['cosine_similarity'].idxmax()
worst_idx = df['cosine_similarity'].idxmin()

print(f"\nMelhor Caso (maior similaridade):")
print(f"  Tipo: {df.loc[best_idx, 'noise_type']}, Nível: {df.loc[best_idx, 'noise_level']}")
print(f"  Similaridade: {df.loc[best_idx, 'cosine_similarity']:.6f}")
print(f"  IoU: {df.loc[best_idx, 'iou']:.6f}")

print(f"\nPior Caso (menor similaridade):")
print(f"  Tipo: {df.loc[worst_idx, 'noise_type']}, Nível: {df.loc[worst_idx, 'noise_level']}")
print(f"  Similaridade: {df.loc[worst_idx, 'cosine_similarity']:.6f}")
print(f"  IoU: {df.loc[worst_idx, 'iou']:.6f}")

# Gráfico adicional: Heatmap
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap de Similaridade
pivot_sim = df.pivot_table(
    values='cosine_similarity', 
    index='noise_type', 
    columns='noise_level',
    aggfunc='mean'
)
sns.heatmap(pivot_sim, annot=True, fmt='.4f', cmap='RdYlGn', ax=axes[0], 
            vmin=0.97, vmax=1.0, cbar_kws={'label': 'Similaridade'})
axes[0].set_title('Similaridade de Cosseno por Tipo e Nível')

# Heatmap de IoU
pivot_iou = df.pivot_table(
    values='iou', 
    index='noise_type', 
    columns='noise_level',
    aggfunc='mean'
)
sns.heatmap(pivot_iou, annot=True, fmt='.4f', cmap='RdYlGn', ax=axes[1],
            vmin=0.8, vmax=1.0, cbar_kws={'label': 'IoU'})
axes[1].set_title('IoU por Tipo e Nível')

plt.tight_layout()
plt.savefig('results/plots/heatmap_analysis.png', dpi=150, bbox_inches='tight')
print("\n✓ Heatmap salvo em: results/plots/heatmap_analysis.png")
plt.close()

# Gráfico: Violinplot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.violinplot(data=df, x='noise_type', y='cosine_similarity', hue='noise_level', ax=axes[0])
axes[0].set_title('Distribuição de Similaridade (Violin Plot)')
axes[0].set_ylabel('Similaridade de Cosseno')

sns.violinplot(data=df, x='noise_type', y='iou', hue='noise_level', ax=axes[1])
axes[1].set_title('Distribuição de IoU (Violin Plot)')
axes[1].set_ylabel('IoU')

plt.tight_layout()
plt.savefig('results/plots/violin_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Violin plot salvo em: results/plots/violin_analysis.png")
plt.close()

print("\n" + "=" * 70)
print("ANÁLISE CONCLUÍDA COM SUCESSO")
print("=" * 70)
