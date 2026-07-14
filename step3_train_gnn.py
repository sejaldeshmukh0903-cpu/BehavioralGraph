"""
STEP 3 — DEFINING AND TRAINING THE GNN
========================================
What this file does:
  - Defines a Graph Neural Network (GNN) architecture
  - Trains it on the customer graph
  - Also trains a Random Forest baseline for comparison
  - Saves both models and their accuracy scores

How a GNN works (simply):
  Layer 1: Each customer node looks at its OWN features
  Layer 2: Each node gathers features from its NEIGHBOURS and averages them
  Layer 3: The combined information is used to make a prediction

  This is called "message passing" — nodes pass information to each other
  across the edges we built in Step 2.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import json

print("=" * 55)
print("  BehavioralGraph — Step 3: Training the GNN")
print("=" * 55)

# ── 1. Load the graph ─────────────────────────────────────────
print("\n[1/5] Loading graph...")
graph = torch.load(r'C:\Users\Sejal\OneDrive\Desktop\AI\BehavioralGraph\behavioralgraph\data\customer_graph.pt', weights_only=False)
print(f"      {graph.num_nodes:,} nodes, {graph.num_edges:,} edges")

# ── 2. Define the GNN model ───────────────────────────────────
print("[2/5] Defining GNN architecture...")

class CustomerGNN(torch.nn.Module):
    """
    A 2-layer Graph Convolutional Network.

    GCNConv is the core building block. In each GCNConv layer:
      - Every node collects feature vectors from its neighbours
      - These are averaged (with some mathematical normalisation)
      - The result passes through a linear transformation
      - Then a ReLU activation adds non-linearity (lets the model
        learn complex, curved patterns — not just straight lines)

    Architecture:
      Input (12 features) → Hidden layer (64 units) → Output (2 classes)
      The 2 output classes are: [probability of 0, probability of 1]
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # Layer 1: compress 12 features into 64 hidden units
        self.conv1 = GCNConv(in_channels, hidden_channels)
        # Layer 2: compress 64 hidden units into 2 output classes
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # Pass through layer 1, apply ReLU, apply dropout (prevents overfitting)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        # Pass through layer 2
        x = self.conv2(x, edge_index)
        return x  # raw scores (logits) for each class

# Initialise the model
model = CustomerGNN(
    in_channels     = graph.num_node_features,  # 12
    hidden_channels = 64,
    out_channels    = 2   # binary: booked or not
)

# Adam optimiser adjusts model weights during training
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

print(f"      Model: GCN | 12 → 64 → 2")
print(f"      Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── 3. Train the GNN ──────────────────────────────────────────
print("[3/5] Training GNN (200 epochs)...")

def train_one_epoch():
    """Run one full pass through the training data."""
    model.train()
    optimizer.zero_grad()
    out  = model(graph.x, graph.edge_index)
    # Only compute loss on TRAINING nodes
    loss = F.cross_entropy(out[graph.train_mask], graph.y[graph.train_mask])
    loss.backward()   # compute gradients (how to improve each weight)
    optimizer.step()  # update weights
    return loss.item()

def evaluate(mask):
    """Compute accuracy on a given set of nodes."""
    model.eval()
    with torch.no_grad():
        out  = model(graph.x, graph.edge_index)
        pred = out.argmax(dim=1)  # pick the class with higher score
        correct = (pred[mask] == graph.y[mask]).sum()
        acc = correct.item() / mask.sum().item()
    return acc

train_losses = []
train_accs   = []
test_accs    = []

for epoch in range(1, 201):
    loss = train_one_epoch()
    train_acc = evaluate(graph.train_mask)
    test_acc  = evaluate(graph.test_mask)

    train_losses.append(loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    if epoch % 20 == 0:
        print(f"      Epoch {epoch:3d} | Loss: {loss:.4f} | "
              f"Train Acc: {train_acc*100:.1f}% | Test Acc: {test_acc*100:.1f}%")

# Final GNN accuracy
model.eval()
with torch.no_grad():
    out  = model(graph.x, graph.edge_index)
    pred = out.argmax(dim=1)
    probs = F.softmax(out, dim=1)[:, 1].numpy()

gnn_test_pred = pred[graph.test_mask].numpy()
gnn_test_true = graph.y[graph.test_mask].numpy()
gnn_test_prob = probs[graph.test_mask.numpy()]
gnn_acc = accuracy_score(gnn_test_true, gnn_test_pred)
gnn_auc = roc_auc_score(gnn_test_true, gnn_test_prob)

print(f"\n      ✅ GNN Final Test Accuracy: {gnn_acc*100:.2f}%")
print(f"      ✅ GNN ROC-AUC:            {gnn_auc:.4f}")

# ── 4. Train Random Forest baseline ───────────────────────────
print("\n[4/5] Training Random Forest baseline for comparison...")

X_np = graph.x.numpy()
y_np = graph.y.numpy()

X_train = X_np[graph.train_mask.numpy()]
y_train = y_np[graph.train_mask.numpy()]
X_test  = X_np[graph.test_mask.numpy()]
y_test  = y_np[graph.test_mask.numpy()]

rf = RandomForestClassifier(
    n_estimators=200, max_depth=10,
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]
rf_acc  = accuracy_score(y_test, rf_pred)
rf_auc  = roc_auc_score(y_test, rf_prob)

print(f"      ✅ RF  Final Test Accuracy: {rf_acc*100:.2f}%")
print(f"      ✅ RF  ROC-AUC:            {rf_auc:.4f}")

# ── 5. Save everything ────────────────────────────────────────
print("\n[5/5] Saving results...")

# Save model and results (near the bottom)
torch.save(model.state_dict(), r'C:\Users\Sejal\OneDrive\Desktop\AI\BehavioralGraph\behavioralgraph\data\gnn_model.pt')

results = {
    'gnn_accuracy':  round(gnn_acc * 100, 2),
    'gnn_auc':       round(gnn_auc, 4),
    'rf_accuracy':   round(rf_acc * 100, 2),
    'rf_auc':        round(rf_auc, 4),
    'train_losses':  train_losses,
    'train_accs':    train_accs,
    'test_accs':     test_accs,
    'rf_importances': rf.feature_importances_.tolist(),
}
with open(r'C:\Users\Sejal\OneDrive\Desktop\AI\BehavioralGraph\behavioralgraph\data\results.json', 'w') as f:
    json.dump(results, f)


print("\n✅ All done!")
print(f"\n   {'Model':<20} {'Accuracy':>10} {'ROC-AUC':>10}")
print(f"   {'-'*42}")
print(f"   {'Graph Neural Network':<20} {gnn_acc*100:>9.2f}% {gnn_auc:>10.4f}")
print(f"   {'Random Forest':<20} {rf_acc*100:>9.2f}% {rf_auc:>10.4f}")
