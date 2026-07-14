"""
STEP 2 — BUILDING THE CUSTOMER GRAPH
=====================================
What this file does:
  - Loads the clean features from Step 1
  - Computes similarity between every pair of customers
  - Draws an edge (connection) between customers who are similar enough
  - Saves the graph in a format PyTorch Geometric can use

The key idea:
  Normal ML treats each customer as an ISOLATED row in a table.
  A Graph Neural Network treats customers as NODES in a network,
  where similar customers are CONNECTED to each other.

  When the GNN predicts whether customer A will complete a booking,
  it looks at A's own features AND the features of A's neighbours.
  This is how it captures group patterns — e.g., "customers who
  behave like A tend to book when they have long purchase lead times."
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 55)
print("  BehavioralGraph — Step 2: Building the Graph")
print("=" * 55)

# ── 1. Load data ──────────────────────────────────────────────
print("\n[1/4] Loading features and labels...")
X = pd.read_csv(r'C:\Users\Sejal\OneDrive\Desktop\AI\BehavioralGraph\behavioralgraph\data\features.csv').values
y = pd.read_csv(r'C:\Users\Sejal\OneDrive\Desktop\AI\BehavioralGraph\behavioralgraph\data\labels.csv').values.flatten()
n = len(X)
print(f"      {n:,} customers loaded, {X.shape[1]} features each")

# ── 2. Compute similarity ─────────────────────────────────────
print("[2/4] Computing customer similarity...")
print("      (This compares every customer to every other customer)")

# Cosine similarity measures the angle between two feature vectors.
# Result is a matrix where sim[i][j] = how similar customer i is to customer j
# Value ranges from 0 (completely different) to 1 (identical behaviour)
sim_matrix = cosine_similarity(X)

# ── 3. Build edges ────────────────────────────────────────────
print("[3/4] Drawing edges between similar customers...")

# THRESHOLD: only connect customers with similarity > 0.97
# Why 0.97? It keeps the graph manageable (not too many edges)
# while still connecting genuinely similar customers
# Think of it as: "only be friends with people you're 97% similar to"
THRESHOLD = 0.97

# Find all pairs (i, j) where similarity exceeds the threshold
# np.where returns the row and column indices where condition is True
rows, cols = np.where((sim_matrix > THRESHOLD) & (np.eye(n) == 0))

# edge_index is the standard PyTorch Geometric format for edges
# Shape: [2, num_edges] — first row = source nodes, second row = target nodes
edge_index = torch.tensor([rows, cols], dtype=torch.long)

print(f"      Threshold: {THRESHOLD}")
print(f"      Edges created: {edge_index.shape[1]:,}")
print(f"      Average connections per customer: {edge_index.shape[1]/n:.1f}")

# ── 4. Package into PyTorch Geometric Data object ─────────────
print("[4/4] Packaging graph...")

# x = node features (what each customer looks like)
# edge_index = which customers are connected
# y = labels (did they complete a booking?)
graph = Data(
    x          = torch.tensor(X, dtype=torch.float),
    edge_index = edge_index,
    y          = torch.tensor(y, dtype=torch.long)
)

# Create train/test split masks
# mask = a list of True/False for each node
# 80% of nodes for training, 20% for testing
torch.manual_seed(42)
perm = torch.randperm(n)
train_size = int(0.8 * n)

train_mask = torch.zeros(n, dtype=torch.bool)
test_mask  = torch.zeros(n, dtype=torch.bool)
train_mask[perm[:train_size]] = True
test_mask[perm[train_size:]]  = True

graph.train_mask = train_mask
graph.test_mask  = test_mask

# Save the graph
torch.save(graph, r'C:\Users\Sejal\OneDrive\Desktop\AI\BehavioralGraph\behavioralgraph\data\customer_graph.pt')


print("\n✅ Graph saved to data/customer_graph.pt")
print(f"\n   Nodes (customers): {graph.num_nodes:,}")
print(f"   Edges (connections): {graph.num_edges:,}")
print(f"   Features per node: {graph.num_node_features}")
print(f"   Training nodes: {train_mask.sum().item():,}")
print(f"   Test nodes: {test_mask.sum().item():,}")
