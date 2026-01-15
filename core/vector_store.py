import faiss
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import json
import os


class VectorStore:
    """FAISS-based vector store with Hybrid Search (Dense + Sparse) and Reranking"""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        """Initialize vector store with sentence transformer and cross encoder models"""
        print(
            f"🔄 Initializing VectorStore with {model_name} and {cross_encoder_model}..."
        )
        self.model = SentenceTransformer(model_name)
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        self.index = faiss.IndexFlatIP(
            self.dimension
        )  # Inner product for cosine similarity

        self.chunks = []  # Store actual text chunks
        self.metadata = []  # Store metadata

        # BM25 attributes
        self.bm25 = None
        self.tokenized_corpus = []

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the vector store and build BM25 index"""
        all_chunks = []
        all_metadata = []
        new_tokenized_chunks = []

        for doc in documents:
            filename = doc["filename"]
            chunks = doc["chunks"]
            doc_metadata = doc["metadata"]

            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata = {
                    "filename": filename,
                    "chunk_id": i,
                    "doc_metadata": doc_metadata,
                    "chunk_text": chunk[:100] + "..." if len(chunk) > 100 else chunk,
                }
                all_metadata.append(chunk_metadata)
                new_tokenized_chunks.append(chunk.lower().split())

        if all_chunks:
            # 1. Dense Vector Indexing
            print(f"🔄 Generating embeddings for {len(all_chunks)} chunks...")
            embeddings = self.model.encode(all_chunks, convert_to_tensor=False)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            self.index.add(embeddings.astype("float32"))

            # 2. Sparse BM25 Indexing
            self.chunks.extend(all_chunks)
            self.metadata.extend(all_metadata)
            self.tokenized_corpus.extend(new_tokenized_chunks)

            print(f"🔄 Building BM25 index for {len(self.tokenized_corpus)} chunks...")
            self.bm25 = BM25Okapi(self.tokenized_corpus)

            print(f"✅ Added {len(all_chunks)} chunks to hybrid store")

    def search(
        self, query: str, top_k: int = 5, use_reranker: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform Hybrid Search (Dense + Sparse) and Reranking
        1. Retrieve top_k * 2 candidates from Dense Search
        2. Retrieve top_k * 2 candidates from Sparse Search (BM25)
        3. Merge unique candidates
        4. Rerank using CrossEncoder
        5. Return top_k results
        """
        if self.index.ntotal == 0:
            return []

        # Increase candidates for hybrid retrieval
        initial_k = top_k * 3

        # --- 1. Dense Search ---
        query_embedding = self.model.encode([query], convert_to_tensor=False)
        query_embedding = query_embedding / np.linalg.norm(
            query_embedding, axis=1, keepdims=True
        )
        scores, indices = self.index.search(
            query_embedding.astype("float32"), initial_k
        )

        dense_results = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks) and idx >= 0:
                dense_results[idx] = float(score)

        # --- 2. Sparse Search (BM25) ---
        tokenized_query = query.lower().split()
        # BM25 returns top n docs, we need indices.
        # We can just get all scores and sort, or use get_top_n equivalent if we had mapping.
        # simple way: get scores for all docs, argsort top k
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[-initial_k:][::-1]

        sparse_results = {}
        for idx in top_bm25_indices:
            sparse_results[idx] = float(bm25_scores[idx])

        # --- 3. Merge Candidates ---
        all_candidate_indices = set(dense_results.keys()) | set(sparse_results.keys())
        candidates = []

        for idx in all_candidate_indices:
            candidates.append(
                {
                    "chunk": self.chunks[idx],
                    "metadata": self.metadata[idx],
                    "original_index": idx,
                }
            )

        print(
            f"🔍 Hybrid Search: Found {len(candidates)} unique candidates from Dense & Sparse search."
        )

        # --- 4. Reranking ---
        if use_reranker and candidates:
            print("🔄 Reranking candidates...")
            sentence_pairs = [[query, c["chunk"]] for c in candidates]
            rerank_scores = self.cross_encoder.predict(sentence_pairs)

            # Attach new scores
            for i, score in enumerate(rerank_scores):
                candidates[i]["score"] = float(score)

            # Sort by reranker score
            candidates.sort(key=lambda x: x["score"], reverse=True)
        else:
            # Fallback (simple weighted fusion or just dense score preference)
            # giving dense score priority if no reranker
            for c in candidates:
                idx = c["original_index"]
                c["score"] = dense_results.get(idx, 0) * 0.7 + (
                    sparse_results.get(idx, 0) / 100.0
                )
            candidates.sort(key=lambda x: x["score"], reverse=True)

        return candidates[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        return {
            "total_chunks": self.index.ntotal,
            "dimension": self.dimension,
            "dense_model": self.model.get_sentence_embedding_dimension(),
            "has_bm25": self.bm25 is not None,
        }

    def clear(self):
        """Clear the vector store"""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = []
        self.metadata = []
        self.bm25 = None
        self.tokenized_corpus = []
        print("🗑️ Vector store cleared")
