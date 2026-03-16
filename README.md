# Store Sales Clustering — End-to-End Unsupervised ML Pipeline

An end-to-end unsupervised machine learning pipeline that clusters stores by sales behaviour, enabling data-driven store categorisation and targeted business strategy. The trained model, scaler, and cluster labels are fully serialized and ready for deployment.

---

## Business Problem

Retail businesses and enterprises with multiple outlets often treat all stores the same — applying identical pricing, inventory, and marketing strategies regardless of performance differences. This project solves that by:

- Automatically grouping stores into distinct clusters based on sales patterns
- Identifying high-performing, average, and underperforming store categories
- Enabling targeted interventions: different stock levels, promotions, and staffing per cluster

**Direct enterprise applications at Sybyl's clients:**
- Retail banks clustering branches by transaction volume and product uptake
- Telcos grouping outlets by airtime sales, data bundle uptake, and mobile money activity
- Enterprise clients optimising territory management and regional strategy

---

## Project Structure

```
store-sales-clustering/
│
├── store_category.ipynb          # Full analysis and modelling notebook
├── sales.csv                     # Raw sales dataset
├── main.py                       # Production pipeline script
├── store_cluster_model.joblib    # Saved trained clustering model
├── store_cluster_scaler.joblib   # Saved StandardScaler for new data
├── cluster_lables.joblib         # Saved cluster label assignments
├── .gitignore
└── README.md
```

---

## Why This Project Is Production-Ready

Most ML projects stop at the notebook. This one goes further:

| File | What it means |
|---|---|
| `store_cluster_model.joblib` | Trained KMeans model saved — can classify new stores without retraining |
| `store_cluster_scaler.joblib` | Scaler saved separately — ensures new data is transformed identically to training data |
| `cluster_lables.joblib` | Cluster assignments saved — enables downstream reporting without re-running the model |
| `main.py` | Pipeline script — can be called from an API or automated workflow |

This architecture mirrors how ML models are deployed in production systems.

---

## Pipeline Overview

```
Raw sales data (sales.csv)
        ↓
Data cleaning and preprocessing
        ↓
Feature engineering — sales metrics per store
        ↓
StandardScaler — normalize features
        ↓
KMeans clustering — find optimal number of clusters
        ↓
Cluster analysis — interpret what each group means
        ↓
Save model + scaler + labels (joblib)
        ↓
Deploy via main.py — classify new stores instantly
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| scikit-learn | KMeans clustering, StandardScaler, evaluation |
| pandas | Data loading, cleaning, feature engineering |
| matplotlib / seaborn | Cluster visualisation |
| joblib | Model serialization for deployment |
| Jupyter Notebook | Exploratory analysis in `store_category.ipynb` |

---

## Key Results

- Successfully segmented stores into distinct sales behaviour clusters
- Saved fully deployable pipeline — new stores can be classified in real time
- Separated scaler from model (best practice) — prevents data leakage on new inputs
- `main.py` enables integration into a REST API or automated reporting workflow

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/ImpactByMercy/End-to-End-Sales-Prediction-Project-unsupervised-
cd End-to-End-Sales-Prediction-Project-unsupervised-

# 2. Install dependencies
pip install scikit-learn pandas matplotlib seaborn joblib jupyter

# 3. Explore the full analysis
jupyter notebook store_category.ipynb

# 4. Run the production pipeline
python main.py

# 5. Load the saved model for inference on new data
import joblib
model = joblib.load('store_cluster_model.joblib')
scaler = joblib.load('store_cluster_scaler.joblib')
# Transform and predict cluster for new store data
new_store_scaled = scaler.transform(new_store_data)
cluster = model.predict(new_store_scaled)
```

---

## Concepts Demonstrated

- **Unsupervised learning** — finding patterns without labelled data
- **K-Means clustering** — grouping stores by feature similarity
- **The Elbow Method** — determining the optimal number of clusters
- **StandardScaler** — normalising features so distance-based algorithms work correctly
- **Model serialization** — saving and loading trained models with joblib
- **Production pipeline design** — separating notebook exploration from deployable code

---

## Author

**Mercy Musyoka** — Data Scientist | AI Engineer  
[LinkedIn](https://www.linkedin.com/in/mercymawiamusyoka) | [GitHub](https://github.com/ImpactByMercy)
