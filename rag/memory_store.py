import json
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class FAISSAnalysisMemory:
    def __init__(self, index_dir: str = "memory_index", model_name: str = "all-MiniLM-L6-v2"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.model = SentenceTransformer(model_name)
        self.index_path = self.index_dir / "analysis.index"
        self.meta_path = self.index_dir / "metadata.json"

        self.dimension = 384
        self.index = self._load_or_create_index()
        self.metadata = self._load_metadata()

    def _load_or_create_index(self):
        if self.index_path.exists():
            return faiss.read_index(str(self.index_path))
        return faiss.IndexFlatL2(self.dimension)

    def _load_metadata(self):
        if self.meta_path.exists():
            with open(self.meta_path, "r") as f:
                return json.load(f)
        return []

    def _save(self):
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def embed(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.astype("float32")

    def add(self, summary_text: str, metadata: Dict[str, Any]):
        vec = self.embed([summary_text])
        self.index.add(vec)
        self.metadata.append({
            "summary": summary_text,
            "metadata": metadata,
        })
        self._save()

    def query(self, query_text: str, top_k: int = 3):
        if len(self.metadata) == 0:
            return []

        q = self.embed([query_text])
        distances, indices = self.index.search(q, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            item = self.metadata[idx]
            results.append({
                "score": float(dist),
                "summary": item["summary"],
                "metadata": item["metadata"],
            })
        return results