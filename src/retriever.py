import os
import json
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

class InMemoryRetriever:
    def __init__(self, dataset_path="data/dataset.json"):
        with open(dataset_path, "r") as f:
            self.dataset = json.load(f)
        
        # Load a lightweight, fast local embedding model
        print("Loading local embedding model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.embeddings = []
        self._embed_dataset()

    def _embed_dataset(self):
        """Pre-computes embeddings for the dataset locally."""
        print("Embedding dataset locally...")
        texts_to_embed = []
        for item in self.dataset:
            # Embed the combination of the email and the policy
            text = f"Email: {item['incoming_email']}\nPolicy: {item['internal_policy']}"
            texts_to_embed.append(text)
            
        # Batch encode is much faster
        self.embeddings_matrix = self.model.encode(texts_to_embed, convert_to_tensor=True)
        print(f"Embedded {len(self.dataset)} items.")

    def search(self, query_email, top_k=2):
        """Finds the top_k most similar past emails to the query."""
        query_emb = self.model.encode(query_email, convert_to_tensor=True)
        
        # Compute cosine similarity using sentence-transformers utility
        cos_scores = util.cos_sim(query_emb, self.embeddings_matrix)[0]
        
        # Get top_k indices
        top_results = np.argpartition(-cos_scores.cpu().numpy(), range(top_k))[:top_k]
        
        # Sort them by score descending
        top_results = top_results[np.argsort(-cos_scores.cpu().numpy()[top_results])]
        
        # Return the actual dataset items
        return [self.dataset[i] for i in top_results]

# Quick test if run directly
if __name__ == "__main__":
    retriever = InMemoryRetriever()
    test_query = "I was charged twice for my Pro plan subscription, please fix!"
    results = retriever.search(test_query)
    for r in results:
        print(f"Match: {r['incoming_email'][:50]}... | Policy: {r['internal_policy'][:50]}...")