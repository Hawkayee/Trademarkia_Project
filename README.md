# Trademarkia AI/ML Engineer Task: Semantic Search Cache

This repository contains a lightweight, fully functional semantic search system and custom cache built from first principles using the 20 Newsgroups dataset.

## Setup & Installation

1. **Clone the repository:** `git clone <your-repo-link>`
2. **Create the virtual environment:** `python -m venv venv`
3. **Activate it:** * Windows: `venv\Scripts\activate`
   * macOS/Linux: `source venv/bin/activate`
4. **Install dependencies:** `pip install -r requirements.txt`
5. **Run the server:** `uvicorn main:app --reload`

*Note: A `Dockerfile` is also included for containerized deployment.*

---

## Architectural Design & Justifications

### 1. Data Preparation & Embeddings
The 20 Newsgroups dataset contains heavy noise (email headers, routing info, signature blocks). I implemented a custom cleaning function to strip these artifacts while explicitly keeping the "Subject" line. 
* **Justification:** Retaining standard email headers would cause the model to artificially cluster documents by author or university network rather than semantic intent. 
* **Embedding Model:** I chose `all-MiniLM-L6-v2`. It strikes the ideal balance between deep semantic capture (384-dimensional vectors) and the lightweight, CPU-friendly performance required for a rapid-response caching system.
* **Vector Store:** I utilized **FAISS** (IndexFlatL2) for persistent, in-memory exact nearest-neighbor search, avoiding the latency and overhead of heavy standalone databases.

### 2. Fuzzy Clustering (Gaussian Mixture Models)
Documents rarely fit into a single hard category. To achieve soft assignments, I used a Gaussian Mixture Model (GMM).
* **Cluster Count Justification:** I did not blindly choose 20 clusters. I ran a Bayesian Information Criterion (BIC) analysis across $k=10$ to $k=40$. Using an automated point-to-line distance algorithm to find the point of maximum curvature (the "elbow"), the system mathematically locked in **22 clusters** as the optimal structure.
* **Boundary Case Uncertainty:** The model successfully identified semantic overlaps. For example, boundary documents discussing the sale of computer hardware showed near-equal probability distributions between "Hardware/Tech" and "Forsale/Classifieds", proving the clusters represent true semantic meaning rather than rigid labels.

### 3. The Custom Semantic Cache
Instead of an $O(N)$ lookup that degrades as the cache grows, this system leverages the GMM cluster structure to achieve highly scalable efficiency.
* **Mechanism:** The cache is a dictionary partitioned by cluster IDs. When a new query arrives, it is embedded, its dominant cluster is predicted by the GMM, and its cosine similarity is computed *only* against previously cached queries in that specific cluster. This drops the search space from $O(N)$ to roughly $O(N/k)$.
* **The Tunable Decision (Threshold = 0.75):** The similarity threshold dictates the cache's strictness. Initial testing at `0.85` resulted in a 0% hit rate for semantically identical but differently worded queries (e.g., "What is the best orbit..." vs "Which orbital path..."). Lowering the threshold to `0.75` allowed the cache to accurately recognize semantic equivalents while remaining strict enough to reject unrelated keyword traps.

### 4. API Endpoints (FastAPI)
The service manages the cache state in memory and exposes the following endpoints:
* `POST /query`: Accepts a single query, checks the cache, and computes/stores the result on a miss.
* `POST /batch_query`: Processes an array of queries in a single request, dynamically caching and retrieving results mid-loop.
* `GET /cache/stats`: Returns current hit/miss rates and total cache size.
* `DELETE /cache`: Flushes the cache entirely.

---

## Future Improvements for Production

**Automating the Tunable Decision (Cluster-Adaptive Thresholds)**
Currently, the cache uses a statically tuned threshold of 0.75. For a fully automated production environment, I would implement Cluster-Adaptive Thresholding. By utilizing the variance metrics already calculated by the Gaussian Mixture Model, the system could dynamically assign stricter similarity thresholds to dense semantic clusters (e.g., Cryptography) and more forgiving thresholds to broad clusters (e.g., General Chat), entirely removing the need for human-tuned magic numbers.
