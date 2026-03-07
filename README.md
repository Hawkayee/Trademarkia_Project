Below is your **entire README content in pure Markdown input format** (no extra formatting blocks), ready to paste directly into a `README.md` file in your repository.

---

# Trademarkia AI/ML Engineer Task: Semantic Search Cache

This repository contains a lightweight semantic search system with a custom semantic cache built from first principles using the **20 Newsgroups dataset**.

The system integrates:

* SentenceTransformer embeddings
* FAISS vector similarity search
* Gaussian Mixture Model fuzzy clustering
* A custom semantic cache
* FastAPI service for live API access

---

# Project Structure

```
.
├── main.py
├── cache.py
├── requirements.txt
├── Dockerfile
├── README.md
└── data/
    ├── corpus_index.faiss
    ├── corpus_metadata.pkl
    └── gmm_model.pkl
```

---

# Setup & Installation

## Option 1: Docker Deployment (Recommended)

The project includes a Dockerfile for containerized deployment.

### Build the Docker Image

```bash
docker build -t semantic-cache-api .
```

### Run the Container

```bash
docker run -p 8000:8000 semantic-cache-api
```

### Access the API

Open in browser:

```
http://localhost:8000/docs
```

---

## Option 2: Local Python Environment

### Clone the Repository

```bash
git clone <your-repo-link>
cd Trademarkia_Project
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Server

```bash
uvicorn main:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

# API Endpoints

## POST /query

Accepts a single query and checks the semantic cache.

Example request:

```json
{
  "query": "What is the best orbit for satellites?"
}
```

Example response:

```json
{
  "query": "...",
  "cache_hit": true,
  "matched_query": "...",
  "similarity_score": 0.91,
  "result": "...",
  "dominant_cluster": 3
}
```

---

## POST /batch_query

Processes multiple queries in one request.

Example request:

```json
{
  "queries": [
    "What is the best orbit for satellites?",
    "Explain geostationary orbit"
  ]
}
```

---

## GET /cache/stats

Returns cache statistics.

Example response:

```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}
```

---

## DELETE /cache

Flushes the cache entirely.

---

# Architectural Design & Justifications

## Data Preparation & Embeddings

The **20 Newsgroups dataset** contains noisy artifacts such as:

* email headers
* routing information
* signature blocks

A preprocessing step removes these artifacts while retaining the **Subject line**, which carries important semantic information.

### Embedding Model

The system uses:

`all-MiniLM-L6-v2`

Reasons:

* 384 dimensional embeddings
* strong semantic representation
* fast inference on CPU
* lightweight for real-time systems

---

## Vector Database

The project uses **FAISS (IndexFlatL2)** for vector similarity search.

Advantages:

* extremely fast nearest neighbor search
* lightweight
* avoids heavy external databases
* ideal for in-memory semantic retrieval

---

## Fuzzy Clustering (Gaussian Mixture Model)

Documents often belong to multiple semantic categories.

Example:

A document discussing **computer hardware sales** may belong to:

* Hardware
* Marketplace

To support soft clustering the system uses **Gaussian Mixture Models (GMM)**.

### Cluster Selection

Cluster count was determined using **Bayesian Information Criterion (BIC)** analysis.

Range evaluated:

```
k = 10 → 40
```

Using a curvature detection algorithm, the optimal number of clusters was determined to be:

```
22 clusters
```

---

## Custom Semantic Cache

Traditional caching fails when queries are phrased differently.

Example:

```
"What is the best orbit for satellites?"
"Which orbital path is optimal for satellites?"
```

The semantic cache solves this problem by comparing query embeddings.

### Cache Mechanism

1. Embed incoming query
2. Predict dominant cluster via GMM
3. Search cache entries within that cluster
4. Compute cosine similarity

This reduces lookup complexity from:

```
O(N) → O(N/k)
```

where `k` is the number of clusters.

---

### Tunable Decision: Similarity Threshold

The similarity threshold controls cache strictness.

Initial experiments:

```
0.85 → Too strict (no cache hits)
```

Final threshold selected:

```
0.75
```

This allows detection of semantically equivalent queries while preventing false matches.

---

# Future Improvements for Production

## Cluster-Adaptive Thresholds

Currently the system uses a static threshold:

```
threshold = 0.75
```

In a production system this could be replaced with **cluster-adaptive thresholds**.

Using variance metrics from the Gaussian Mixture Model:

* dense clusters → stricter similarity thresholds
* broad clusters → relaxed thresholds

This removes manual tuning and enables automatic optimization.

---

# Technologies Used

* Python
* FastAPI
* SentenceTransformers
* FAISS
* Scikit-learn
* NumPy
* Docker

---

# Running the Project

Local execution:

```bash
uvicorn main:app --reload
```

Docker execution:

```bash
docker build -t semantic-cache-api .
docker run -p 8000:8000 semantic-cache-api
```

---

# Author

Ranjith K
