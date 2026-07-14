"""
STEP 4 — VISUALISATIONS
========================
What this file does:
  - Produces 4 publication-quality charts:
    1. Training curve (loss over epochs)
    2. Accuracy comparison: GNN vs Random Forest
    3. Feature importance (from the RF baseline)
    4. t-SNE embedding — the "hero" visualisation that shows
       how the GNN has learned to cluster similar customers

  These go straight into your GitHub README and CV portfolio.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE

print("=" * 55)
print("  BehavioralGraph — Step 4: Visualisations")
print("=" * 55)

# ── Load everything ───────────────────────────────────────────
print("\n[1/5] Loading data...")

# Load graph
graph = torch.load(r'C:\Users\Sejal\OneDrive\Desktop\AI\BehavioralGraph\behavioralgraph\data\customer_graph.pt', weights_only=False)
# Load results
with open(r'C:\Users\Sejal\OneDrive\Desktop\AI\BehavioralGraph\behavioralgraph\data\results.json') as f:
    results = json.load(f)

FEATURE_NAMES = [
    'Num Passengers', 'Purchase Lead', 'Length of Stay',
    'Flight Hour', 'Flight Duration', 'Total Add-ons',
    'Sales Channel', 'Trip Type', 'Booking Origin',
    'Gender', 'Age', 'Salary'
]

# Colour palette — navy/teal professional scheme
C1 = '#1F3864'   # dark navy
C2 = '#2874A6'   # mid blue
C3 = '#48C9B0'   # teal accent
C4 = '#E74C3C'   # red for contrast
GREY = '#95A5A6'

# ── Chart 1: Training curve ────────────────────────────────────
print("[2/5] Chart 1 — Training curve...")
fig, ax = plt.subplots(figsize=(8, 4.5))

epochs = range(1, len(results['train_losses']) + 1)
ax.plot(epochs, results['train_losses'], color=C1, linewidth=2, label='Training Loss')
ax2 = ax.twinx()
ax2.plot(epochs, [a*100 for a in results['test_accs']],
         color=C3, linewidth=2, linestyle='--', label='Test Accuracy %')

ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Loss', color=C1, fontsize=11)
ax2.set_ylabel('Test Accuracy (%)', color=C3, fontsize=11)
ax.set_title('GNN Training Curve\nBehavioralGraph — Customer Booking Prediction',
             fontsize=12, fontweight='bold', pad=12)
ax.tick_params(axis='y', labelcolor=C1)
ax2.tick_params(axis='y', labelcolor=C3)
ax.spines['top'].set_visible(False)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=9)

fig.tight_layout()
fig.savefig(r'C:\Users\Sejal\OneDrive\Desktop\AI\BehavioralGraph\behavioralgraph\outputs\1_training_curve.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("      Saved: 1_training_curve.png")

# ── Chart 2: GNN vs RF accuracy comparison ────────────────────
print("[3/5] Chart 2 — Model comparison...")
fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

models  = ['Random\nForest', 'Graph Neural\nNetwork']
accs    = [results['rf_accuracy'], results['gnn_accuracy']]
aucs    = [results['rf_auc'], results['gnn_auc']]
colors  = [GREY, C2]

for ax, vals, title, fmt in zip(
    axes,
    [accs, aucs],
    ['Test Accuracy (%)', 'ROC-AUC Score'],
    ['{:.1f}%', '{:.4f}']
):
    bars = ax.bar(models, vals, color=colors, width=0.45, edgecolor='none')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ymin = min(vals) * 0.96
    ax.set_ylim(ymin, max(vals) * 1.06)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (max(vals)-min(vals))*0.01,
                fmt.format(val),
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax.tick_params(axis='x', labelsize=10)

fig.suptitle('BehavioralGraph — GNN vs Random Forest Baseline',
             fontsize=12, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(r'C:\Users\Sejal\OneDrive\Desktop\AI\BehavioralGraph\behavioralgraph\outputs\2_model_comparison.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("      Saved: 2_model_comparison.png")

# ── Chart 3: Feature importance ────────────────────────────────
print("[4/5] Chart 3 — Feature importance...")
fi = np.array(results['rf_importances'])
order = np.argsort(fi)

fig, ax = plt.subplots(figsize=(8, 5.5))
bar_colors = [C1 if v > np.percentile(fi, 75) else
              C2 if v > np.percentile(fi, 50) else GREY
              for v in fi[order]]
bars = ax.barh([FEATURE_NAMES[i] for i in order], fi[order],
               color=bar_colors, edgecolor='none', height=0.6)
ax.set_xlabel('Importance Score', fontsize=10)
ax.set_title('Feature Importance — What Drives Booking Completion?\n'
             'BehavioralGraph Project', fontsize=11, fontweight='bold', pad=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.xaxis.grid(True, color='#ececec', linewidth=0.8)
ax.set_axisbelow(True)
for bar, val in zip(bars, fi[order]):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=8)

legend_patches = [
    mpatches.Patch(color=C1, label='High importance (top 25%)'),
    mpatches.Patch(color=C2, label='Medium importance'),
    mpatches.Patch(color=GREY, label='Lower importance'),
]
ax.legend(handles=legend_patches, fontsize=8, loc='lower right')
fig.tight_layout()
fig.savefig(r'C:\Users\Sejal\OneDrive\Desktop\AI\BehavioralGraph\behavioralgraph\outputs\3_feature_importance.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("      Saved: 3_feature_importance.png")

# ── Chart 4: t-SNE embedding (the hero visualisation) ─────────
print("[5/5] Chart 4 — t-SNE embedding (takes ~30 seconds)...")

# Get the learned node embeddings from the GNN's hidden layer
# These are 64-dimensional vectors — what the GNN "thinks" each customer is
# t-SNE compresses these 64 dimensions down to 2 so we can plot them

# We need to re-run the model to extract hidden representations
from torch_geometric.nn import GCNConv

class CustomerGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(12, 64)
        self.conv2 = GCNConv(64, 2)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return x   # return hidden layer, not final output

model = CustomerGNN()
# Load model
model.load_state_dict(
    torch.load(r'C:\Users\Sejal\OneDrive\Desktop\AI\BehavioralGraph\behavioralgraph\data\gnn_model.pt', weights_only=True),
    strict=False
)
model.eval()
with torch.no_grad():
    embeddings = model(graph.x, graph.edge_index).numpy()

# Sample 3000 points for faster t-SNE
idx    = np.random.choice(len(embeddings), 3000, replace=False)
emb_s  = embeddings[idx]
labels = graph.y.numpy()[idx]

tsne   = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1000)
coords = tsne.fit_transform(emb_s)

fig, ax = plt.subplots(figsize=(9, 6.5))
colors_tsne = [C4 if l == 1 else C2 for l in labels]
alphas = [0.85 if l == 1 else 0.35 for l in labels]

for xi, yi, ci, ai in zip(coords[:,0], coords[:,1], colors_tsne, alphas):
    ax.scatter(xi, yi, c=ci, alpha=ai, s=12, linewidths=0)

patch0 = mpatches.Patch(color=C2, alpha=0.5, label='Booking NOT completed')
patch1 = mpatches.Patch(color=C4, alpha=0.9, label='Booking completed ✓')
ax.legend(handles=[patch0, patch1], fontsize=10, loc='upper right',
          framealpha=0.9)
ax.set_title('t-SNE Visualisation of GNN Customer Embeddings\n'
             'Each point = one customer  |  Colour = booking outcome',
             fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel('t-SNE Dimension 1', fontsize=10)
ax.set_ylabel('t-SNE Dimension 2', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
fig.savefig(r'C:\Users\Sejal\OneDrive\Desktop\AI\BehavioralGraph\behavioralgraph\outputs\4_tsne_embeddings.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("      Saved: 4_tsne_embeddings.png")

print("\n✅ All 4 visualisations saved to outputs/")
