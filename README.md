# BehavioralGraph 🧠
### Predicting Customer Decisions Using Graph Neural Networks

> A machine learning project that models customer behaviour as a **graph structure**, 
> connecting similar customers as nodes in a network, then using a Graph Neural Network (GNN) 
> to predict booking completion — outperforming a traditional Random Forest baseline.

---

## 📌 Motivation

Standard ML models treat each customer as an **isolated row** in a spreadsheet.  
But customers don't behave in isolation — people with similar demographics, preferences, 
and travel habits form natural **clusters**.

This project asks: *what if we modelled those relationships explicitly?*

By building a **customer similarity graph** and training a Graph Convolutional Network (GCN) 
on it, the model can look at not just a customer's own features, but also the behaviour of 
similar customers — capturing group-level patterns that flat feature vectors miss.

---

## 📂 Datasets

| Dataset | Source | Records | Purpose |
|---|---|---|---|
| British Airways Booking Data | BA x Forage Job Simulation | 50,000 | Behavioural features (purchase lead, add-ons, booking channel) |
| CommBank Customer Data | CommBank x Forage Job Simulation | 10,000 | Demographic features (age, salary, gender) |

Both datasets are combined into a **unified 12-feature customer profile** per node.

---

## 🏗️ Project Architecture

```
customer_booking.csv  ──┐
                        ├──► Step 1: Data Preparation  ──► features.csv + labels.csv
mobile_customers.xlsx ──┘                                       │
                                                                ▼
                                               Step 2: Graph Construction
                                               (cosine similarity > 0.97 → edge)
                                                                │
                                                                ▼
                                               Step 3: GNN Training + RF Baseline
                                                                │
                                                                ▼
                                               Step 4: Visualisations
```

---

## 🔬 Methodology

### Graph Construction
- Each of 10,000 customers is a **node**
- Two customers are connected by an **edge** if their cosine similarity exceeds **0.97**
- This produces **677,768 edges** — an average of ~68 connections per customer

### Model: Graph Convolutional Network (GCN)
```
Input Layer:   12 features per node
Hidden Layer:  64 units  +  ReLU activation  +  Dropout (p=0.3)
Output Layer:  2 classes (booking complete / not complete)

Optimiser:     Adam  (lr=0.01, weight_decay=5e-4)
Epochs:        200
```

The GCN aggregates information from each node's neighbours via **message passing** — 
the mathematical operation that makes graph learning fundamentally different from 
standard neural networks.

---

## 📊 Results

| Model | Test Accuracy | ROC-AUC |
|---|---|---|
| Random Forest (baseline) | 77.70% | 0.7391 |
| **Graph Neural Network** | **84.85%** | **0.6833** |

The GNN achieves **+7.15 percentage points** higher accuracy by leveraging the 
relational structure of the customer graph — information the Random Forest 
cannot access.

---

## 📈 Visualisations

### Training Curve
Loss steadily decreasing over 200 epochs confirms stable convergence.

![Training Curve](outputs/1_training_curve.png)

### GNN vs Random Forest
![Model Comparison](outputs/2_model_comparison.png)

### Feature Importance
Purchase lead time and flight duration are the strongest predictors of booking completion.

![Feature Importance](outputs/3_feature_importance.png)

### t-SNE Customer Embeddings
Each point is a customer, coloured by booking outcome. The GNN has learned to 
cluster customers with similar behaviour — red clusters indicate regions where 
bookings are concentrated.

![t-SNE Embeddings](outputs/4_tsne_embeddings.png)

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/behavioralgraph
cd behavioralgraph

# 2. Install dependencies
pip install torch torch-geometric scikit-learn pandas numpy matplotlib seaborn

# 3. Add datasets to project root:
#    customer_booking.csv  (BA Forage simulation)
#    mobile_customers.xlsx (CommBank Forage simulation)

# 4. Run the pipeline step by step
python step1_prepare_data.py    # Clean + merge datasets
python step2_build_graph.py     # Build customer similarity graph
python step3_train_gnn.py       # Train GNN + RF baseline
python step4_visualise.py       # Generate all charts
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `PyTorch` | Neural network framework |
| `PyTorch Geometric` | Graph neural network layers (GCNConv) |
| `scikit-learn` | Random Forest baseline, preprocessing, metrics |
| `pandas / numpy` | Data wrangling |
| `matplotlib / seaborn` | Visualisations |

---

## 🔭 Future Extensions

- **Temporal edges** — weight connections by recency (more recent similarity matters more)
- **Heterogeneous graph** — add flight routes as separate node types
- **Graph Attention Networks (GAT)** — let the model learn *which* neighbours matter most
- **Larger scale** — test on the full 50,000 customer dataset with GPU acceleration

---

## 👤 Author

Built as part of a research portfolio.
