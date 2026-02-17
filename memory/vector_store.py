"""
Apex Predator — Vector Memory Store
=====================================
Pattern memory using FAISS or numpy fallback.
Stores market state embeddings with trade outcomes
for similarity-based recall of past patterns.
"""

import os
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

import numpy as np

from config.settings import MemoryConfig

logger = logging.getLogger("apex_predator.memory")


@dataclass
class MemoryEntry:
    """A stored market state with outcome."""
    embedding: np.ndarray     # State embedding from perception
    action: int               # Action taken (0=BUY, 1=SELL, 2=HOLD)
    outcome: float            # P&L result
    regime: str               # Market regime at the time
    timestamp: float          # Unix timestamp


class VectorMemory:
    """
    Long-term pattern memory using vector similarity search.
    Stores embeddings from profitable/unprofitable trades
    and recalls similar historical situations.

    Uses FAISS for fast similarity search, with numpy cosine fallback.
    """

    def __init__(self, config: MemoryConfig):
        self.config = config
        self._embeddings: List[np.ndarray] = []
        self._metadata: List[dict] = []
        self._faiss_index = None
        self._use_faiss = False
        self._stores_since_save: int = 0

        os.makedirs(config.persist_path, exist_ok=True)
        self._init_faiss()

    def _init_faiss(self):
        """Try to initialise FAISS index."""
        try:
            import faiss
            self._faiss_index = faiss.IndexFlatIP(self.config.embedding_dim)
            self._use_faiss = True
            logger.info("VectorMemory using FAISS (dim=%d)", self.config.embedding_dim)
        except ImportError:
            logger.info("FAISS not available — using numpy cosine similarity fallback")
            self._use_faiss = False

    # ── Store ───────────────────────────────────

    def store(
        self,
        embedding: np.ndarray,
        action: int,
        outcome: float,
        regime: str = "",
        timestamp: float = 0.0,
    ):
        """
        Store a market state embedding with its trade outcome.

        Args:
            embedding: State embedding from perception module (embedding_dim,)
            action: Action taken (0, 1, 2)
            outcome: P&L result (positive = profit)
            regime: Market regime string
            timestamp: Unix timestamp
        """
        # Normalize embedding for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        embedding = embedding.astype(np.float32)

        self._embeddings.append(embedding)
        self._metadata.append({
            "action": action,
            "outcome": outcome,
            "regime": regime,
            "timestamp": timestamp,
        })

        if self._use_faiss and self._faiss_index is not None:
            self._faiss_index.add(embedding.reshape(1, -1))

        # Trim if exceeded max capacity
        if len(self._embeddings) > self.config.max_memories:
            self._trim()

        # Auto-save every N stores
        self._stores_since_save += 1
        if self._stores_since_save >= self.config.auto_save_interval:
            self.save()
            self._stores_since_save = 0

    # ── Recall

    def recall(
        self,
        query_embedding: np.ndarray,
        top_k: Optional[int] = None,
    ) -> List[Tuple[dict, float]]:
        """
        Find similar historical patterns.

        Args:
            query_embedding: Current state embedding
            top_k: Number of similar patterns to retrieve

        Returns:
            List of (metadata_dict, similarity_score) tuples,
            sorted by similarity descending
        """
        if len(self._embeddings) == 0:
            return []

        k = top_k or self.config.recall_top_k

        # Normalize query
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        query = query_embedding.astype(np.float32)

        if self._use_faiss and self._faiss_index is not None:
            return self._recall_faiss(query, k)
        else:
            return self._recall_numpy(query, k)

    def _recall_faiss(
        self, query: np.ndarray, top_k: int
    ) -> List[Tuple[dict, float]]:
        """FAISS-based recall."""
        k = min(top_k, self._faiss_index.ntotal)
        if k == 0:
            return []

        scores, indices = self._faiss_index.search(query.reshape(1, -1), k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            if score >= self.config.similarity_threshold:
                results.append((self._metadata[idx], float(score)))
        return results

    def _recall_numpy(
        self, query: np.ndarray, top_k: int
    ) -> List[Tuple[dict, float]]:
        """Numpy cosine similarity fallback."""
        if not self._embeddings:
            return []

        matrix = np.array(self._embeddings)
        similarities = matrix @ query  # Cosine similarity (vectors are normalised)

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = []
        for idx in top_indices:
            sim = float(similarities[idx])
            if sim >= self.config.similarity_threshold:
                results.append((self._metadata[idx], sim))
        return results

    # ── Analysis ────────────────────────────────

    def get_pattern_stats(
        self, query_embedding: np.ndarray
    ) -> dict:
        """
        Analyse recall results to provide trade guidance.

        Returns:
            {
                "similar_count": int,
                "avg_outcome": float,
                "win_rate": float,
                "dominant_action": int,
                "confidence": float,
            }
        """
        recalls = self.recall(query_embedding)
        if not recalls:
            return {
                "similar_count": 0,
                "avg_outcome": 0.0,
                "win_rate": 0.0,
                "dominant_action": 2,  # HOLD
                "confidence": 0.0,
            }

        outcomes = [r[0]["outcome"] for r in recalls]
        actions = [r[0]["action"] for r in recalls]
        similarities = [r[1] for r in recalls]

        # Weighted outcomes by similarity
        weights = np.array(similarities)
        weighted_outcomes = np.array(outcomes) * weights

        action_counts = {0: 0, 1: 0, 2: 0}
        for a in actions:
            action_counts[a] = action_counts.get(a, 0) + 1
        dominant = max(action_counts, key=action_counts.get)

        return {
            "similar_count": len(recalls),
            "avg_outcome": float(np.mean(weighted_outcomes)),
            "win_rate": float(np.mean(np.array(outcomes) > 0)),
            "dominant_action": dominant,
            "confidence": float(np.mean(similarities)),
        }

    # ── Persistence ─────────────────────────────

    def save(self):
        """Save memory to disk."""
        path = os.path.join(self.config.persist_path, "vector_memory.npz")
        if self._embeddings:
            np.savez_compressed(
                path,
                embeddings=np.array(self._embeddings),
            )
            # Save metadata separately
            import json
            meta_path = os.path.join(self.config.persist_path, "memory_metadata.json")
            with open(meta_path, "w") as f:
                json.dump(self._metadata, f)
            logger.info("VectorMemory saved: %d entries", len(self._embeddings))

    def load(self):
        """Load memory from disk."""
        emb_path = os.path.join(self.config.persist_path, "vector_memory.npz")
        meta_path = os.path.join(self.config.persist_path, "memory_metadata.json")

        if not os.path.exists(emb_path):
            return

        try:
            data = np.load(emb_path)
            embeddings = data["embeddings"]

            import json
            with open(meta_path, "r") as f:
                metadata = json.load(f)

            self._embeddings = list(embeddings)
            self._metadata = metadata

            if self._use_faiss and self._faiss_index is not None:
                import faiss
                self._faiss_index = faiss.IndexFlatIP(self.config.embedding_dim)
                self._faiss_index.add(embeddings.astype(np.float32))

            logger.info("VectorMemory loaded: %d entries", len(self._embeddings))
        except Exception as e:
            logger.error("Failed to load VectorMemory: %s", e)

    # ── Maintenance ─────────────────────────────

    def _trim(self):
        """Remove oldest entries when exceeding max capacity."""
        excess = len(self._embeddings) - self.config.max_memories
        if excess > 0:
            self._embeddings = self._embeddings[excess:]
            self._metadata = self._metadata[excess:]
            # Rebuild FAISS index
            if self._use_faiss:
                self._rebuild_faiss()

    def _rebuild_faiss(self):
        """Rebuild FAISS index from current embeddings."""
        try:
            import faiss
            self._faiss_index = faiss.IndexFlatIP(self.config.embedding_dim)
            if self._embeddings:
                matrix = np.array(self._embeddings, dtype=np.float32)
                self._faiss_index.add(matrix)
        except ImportError:
            pass

    def size(self) -> int:
        """Get number of stored memories."""
        return len(self._embeddings)
