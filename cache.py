import numpy as np
class SemanticCache:
    def __init__(self, embedding_model, gmm_model,threshold=0.85):
        """
        Initializes the Semantic Cache.
        
        JUSTIFICATION FOR TUNABLE DECISION (Threshold):
        The 'threshold' (default 0.85) dictates the strictness of the cache. 
        - A high value (e.g., 0.95) acts almost like an exact-match cache: high accuracy, 
          but very low hit rate. 
        - A low value (e.g., 0.70) yields a high hit rate, but risks returning an answer 
          about "baseball" when the user asked about "hockey" because they share a semantic 
          cluster. 0.85 offers a balanced sweet spot for conversational NLP."""

        self.embedding_model =embedding_model
        self.gmm_model = gmm_model
        self.threshold = threshold

        # The core data Structure : A dictionary partitioned cluster ID
        self.cache_store = {} 

        # State management for API
        self.total_entries = 0;
        self.hit_count = 0;
        self.miss_count =0;
        
    def _cosine_similarity(self,vec1, vec2):
        # calculates cosine similarity between 2 vectors 
        dot_product =np.dot(vec1,vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        if norm_a == 0 or norm_b ==0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def check_cache(self,query: str):
        """
        Checks if a semantically similar query exists in the cache.
        Returns (is_hit, matched_query, score, cached_result, dominant_cluster).
        """

        # 1 Embed the incomming query
        query_embedding  = self.embedding_model.encode([query])[0]

        # 2 Find the dominent cluster using GMM
        cluster_probs  = self.gmm_model.predict_proba([query_embedding])[0]
        dominant_cluster = int(np.argmax(cluster_probs))

        # 3 Cluster bin doesn't exits it's automatic miss
        if dominant_cluster not in self.cache_store or not self.cache_store[dominant_cluster]:
            return False, None, 0.0, None, dominant_cluster

        # 4. Search ONLY within the dominant cluster
        best_score = -1.0
        best_match = None

        for cached_item in self.cache_store[dominant_cluster]:
            score =self._cosine_similarity(query_embedding, cached_item['embedding'])
            if score > best_score:
                best_score = score
                best_match= cached_item
        # 5. Evaluate against our tunable threshold

        if best_score >= self.threshold:
            self.hit_count +=1
            return True, best_match['query'], float(best_score), best_match['result'], dominant_cluster

        return False, None,0.0, None, dominant_cluster

    def add_to_cache(self, query: str, query_embedding, result: str, dominant_cluster: int):
        """Stores a new query and its result in the appropriate cluster bin."""
        if dominant_cluster not in self.cache_store:
            self.cache_store[dominant_cluster] = []
            
        self.cache_store[dominant_cluster].append({
            'query': query,
            'embedding': query_embedding,
            'result': result
        })
        
        self.total_entries += 1
        self.miss_count += 1 # If we are adding it, it means it was a miss

    def get_stats(self):
        """Returns the current state of the cache for the API."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests) if total_requests > 0 else 0.0
        
        return {
            "total_entries": self.total_entries,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(hit_rate, 3)
        }
        
    def flush(self):
        """Clears the cache entirely."""
        self.cache_store = {}
        self.total_entries = 0
        self.hit_count = 0
        self.miss_count = 0