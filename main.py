from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import joblib
import numpy as np
from cache import SemanticCache

app = FastAPI(title="Trademarkia Semantic Search Cache")

# --- 1. STARTUP LOADING ---
print("Starting server up... Loading models.")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
gmm_model = joblib.load("data/gmm_model.pkl")
index = faiss.read_index("data/corpus_index.faiss")

with open("data/corpus_metadata.pkl", "rb") as f:
    corpus_metadata = pickle.load(f)

# Initialize semantic cache
semantic_cache = SemanticCache(embedding_model, gmm_model, threshold=0.75)


# --- 2. API SCHEMAS ---

class QueryRequest(BaseModel):
    query: str


class BatchQueryRequest(BaseModel):
    queries: List[str]


# --- 3. ENDPOINTS ---

# ---------------------------------
# 1️ SINGLE QUERY ENDPOINT
# ---------------------------------
@app.post("/query")
async def process_query(request: QueryRequest):
    user_query = request.query

    # Step 1: Check cache
    is_hit, matched_query, score, cached_result, dominant_cluster = semantic_cache.check_cache(user_query)

    if is_hit:
        return {
            "query": user_query,
            "cache_hit": True,
            "matched_query": matched_query,
            "similarity_score": round(score, 3),
            "result": cached_result,
            "dominant_cluster": dominant_cluster
        }

    # Step 2: Cache MISS → compute embedding
    query_embedding = embedding_model.encode([user_query])[0]

    # Step 3: Find dominant cluster
    dominant_cluster = int(gmm_model.predict([query_embedding])[0])

    # Step 4: Search FAISS
    distances, indices = index.search(np.array([query_embedding]), k=1)

    best_match_idx = indices[0][0]
    real_result = corpus_metadata["documents"][best_match_idx]

    # Step 5: Store in cache
    semantic_cache.add_to_cache(user_query, query_embedding, real_result, dominant_cluster)

    return {
        "query": user_query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": 0.0,
        "result": real_result,
        "dominant_cluster": dominant_cluster
    }


# ---------------------------------
# 2️ MULTIPLE QUERY ENDPOINT
# ---------------------------------
@app.post("/batch_query")
async def batch_query(request: BatchQueryRequest):

    results = []

    # Encode all queries at once (faster)
    embeddings = embedding_model.encode(request.queries)

    for i, user_query in enumerate(request.queries):

        # Check cache
        is_hit, matched_query, score, cached_result, dominant_cluster = semantic_cache.check_cache(user_query)

        if is_hit:
            results.append({
                "query": user_query,
                "cache_hit": True,
                "matched_query": matched_query,
                "similarity_score": round(score, 3),
                "result": cached_result,
                "dominant_cluster": dominant_cluster
            })
            continue

        query_embedding = embeddings[i]

        # Cluster prediction
        dominant_cluster = int(gmm_model.predict([query_embedding])[0])

        # FAISS search
        distances, indices = index.search(np.array([query_embedding]), k=1)

        best_match_idx = indices[0][0]
        real_result = corpus_metadata["documents"][best_match_idx]

        # Store in cache
        semantic_cache.add_to_cache(user_query, query_embedding, real_result, dominant_cluster)

        results.append({
            "query": user_query,
            "cache_hit": False,
            "matched_query": None,
            "similarity_score": 0.0,
            "result": real_result,
            "dominant_cluster": dominant_cluster
        })

    return {"results": results}


# ---------------------------------
# 3️ CACHE STATS
# ---------------------------------
@app.get("/cache/stats")
async def get_cache_stats():
    return semantic_cache.get_stats()


# ---------------------------------
# 4️ CLEAR CACHE
# ---------------------------------
@app.delete("/cache")
async def clear_cache():
    semantic_cache.flush()
    return {"message": "Cache successfully flushed."}