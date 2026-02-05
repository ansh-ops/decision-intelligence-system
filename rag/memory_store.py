import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class AnalysisMemory:
    def __init__(self, embedder):
        self.embedder = embedder
        self.entries = []

    def add(self, dataset_signature, summary_text, metadata):
        emb = self.embedder(summary_text)
        self.entries.append({
            "signature": dataset_signature,
            "embedding": emb,
            "summary": summary_text,
            "metadata": metadata
        })

    def query(self, query_text, top_k=3):
        q_emb = self.embedder(query_text)
        sims = [
            cosine_similarity([q_emb], [e["embedding"]])[0][0]
            for e in self.entries
        ]

        ranked = sorted(
            zip(self.entries, sims),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked[:top_k]
