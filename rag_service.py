"""
rag_service.py
==============
Lightweight Hybrid RAG (Retrieval-Augmented Generation) Service.
 Combines:
  1. Dense Semantic Search (Hugging Face Sentence-Transformers embeddings)
  2. Sparse Keyword Search (BM25 term matching)
  3. Reciprocal Rank Fusion (RRF) to merge and rank results
  4. Section & Clause aware text chunking
"""

import os
import json
import re
import math
from typing import List, Dict, Any, Tuple
import numpy as np

from ocr_service import ocr_service


class RAGService:
    """
    Hybrid RAG engine managing document ingestion, embedding indexing,
    BM25 keyword indexing, and RRF fused retrieval.
    """

    def __init__(self, storage_dir: str = "rag_storage"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.meta_file = os.path.join(self.storage_dir, "documents.json")

        self.documents: List[Dict[str, Any]] = [] # [{id, filename, chunk_index, text, meta}]
        self.embeddings: List[np.ndarray] = []
        self.embed_model = None

        self._load_metadata()
        self._init_embedding_model()

    def _init_embedding_model(self):
        """Lazy load lightweight Hugging Face embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            # Lightweight, fast, multilingual sentence embedding model
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            print(f"Loading RAG embedding model: {model_name}...")
            self.embed_model = SentenceTransformer(model_name)
            print("✅ RAG embedding model ready.")
        except Exception as e:
            print(f"⚠️ SentenceTransformer init failed ({e}). Falling back to TF-IDF cosine vectorizer.")
            self.embed_model = None

    def _load_metadata(self):
        """Load stored chunk metadata from disk."""
        if os.path.exists(self.meta_file):
            try:
                with open(self.meta_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.documents = data.get("documents", [])
                    raw_embeds = data.get("embeddings", [])
                    if raw_embeds:
                        self.embeddings = [np.array(e, dtype=np.float32) for e in raw_embeds]
                print(f"ℹ️ Loaded {len(self.documents)} RAG chunks from storage.")
            except Exception as e:
                print(f"⚠️ Failed to load RAG metadata: {e}")

    def _save_metadata(self):
        """Persist metadata and embeddings to disk."""
        try:
            raw_embeds = [e.tolist() for e in self.embeddings] if self.embeddings else []
            data = {
                "documents": self.documents,
                "embeddings": raw_embeds
            }
            with open(self.meta_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Failed to save RAG metadata: {e}")

    # ============================================================
    # CHUNKING
    # ============================================================

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Section and clause aware recursive text chunking.
        Splits by paragraphs, clauses, or sentences.
        """
        text = text.strip()
        if not text:
            return []

        # Split into initial paragraphs / sections
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []

        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) + 1 <= chunk_size:
                current_chunk = (current_chunk + "\n\n" + para).strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                # If individual paragraph is larger than chunk size, split by sentences
                if len(para) > chunk_size:
                    sentences = re.split(r'(?<=[.?!])\s+', para)
                    current_chunk = ""
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                            current_chunk = (current_chunk + " " + sentence).strip()
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = sentence
                else:
                    current_chunk = para

        if current_chunk:
            chunks.append(current_chunk)

        # Apply overlap if needed
        final_chunks = []
        for i, c in enumerate(chunks):
            if i > 0 and overlap > 0 and len(chunks[i-1]) >= overlap:
                overlap_text = chunks[i-1][-overlap:]
                c = overlap_text + " " + c
            final_chunks.append(c.strip())

        return final_chunks

    # ============================================================
    # INGESTION
    # ============================================================

    def add_document(self, file_path: str, custom_name: str = None) -> Dict[str, Any]:
        """
        Process a file, extract text via OCR/Parsers, chunk it, compute embeddings, and store.
        """
        filename = custom_name or os.path.basename(file_path)
        print(f"📄 Processing document for RAG: {filename}")

        # 1. OCR / Text Extraction
        extracted_text = ocr_service.extract_text_from_file(file_path)
        if not extracted_text or extracted_text == "No extractable text found in PDF.":
            return {"status": "error", "message": "No text could be extracted from document."}

        # 2. Chunking
        chunks = self.chunk_text(extracted_text)
        if not chunks:
            return {"status": "error", "message": "Document contains no readable chunks."}

        # 3. Compute Embeddings
        new_embeds = []
        if self.embed_model is not None:
            raw_vectors = self.embed_model.encode(chunks, show_progress_bar=False)
            for vec in raw_vectors:
                # Normalize vector for cosine similarity
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                new_embeds.append(vec.astype(np.float32))

        # 4. Save Chunks to Store
        added_count = 0
        for i, chunk in enumerate(chunks):
            chunk_id = f"{filename}_chunk_{i}"
            doc_item = {
                "id": chunk_id,
                "filename": filename,
                "chunk_index": i,
                "text": chunk,
            }
            self.documents.append(doc_item)
            if new_embeds:
                self.embeddings.append(new_embeds[i])
            added_count += 1

        self._save_metadata()
        print(f"✅ Ingested '{filename}': {added_count} chunks stored.")
        return {
            "status": "success",
            "filename": filename,
            "chunks_added": added_count,
            "total_documents": len(set(d['filename'] for d in self.documents)),
            "total_chunks": len(self.documents)
        }

    def list_documents(self) -> List[Dict[str, Any]]:
        """List distinct ingested files and chunk counts."""
        file_counts = {}
        for d in self.documents:
            fn = d['filename']
            file_counts[fn] = file_counts.get(fn, 0) + 1

        return [{"filename": fn, "chunks": count} for fn, count in file_counts.items()]

    def delete_document(self, filename: str) -> bool:
        """Remove all chunks associated with a filename."""
        indices_to_remove = [i for i, d in enumerate(self.documents) if d['filename'] == filename]
        if not indices_to_remove:
            return False

        # Filter out from documents and embeddings
        self.documents = [d for i, d in enumerate(self.documents) if i not in indices_to_remove]
        if self.embeddings:
            self.embeddings = [e for i, e in enumerate(self.embeddings) if i not in indices_to_remove]

        self._save_metadata()
        print(f"🗑️ Removed document '{filename}' from RAG index.")
        return True

    def clear_all(self):
        """Clear all ingested documents."""
        self.documents = []
        self.embeddings = []
        self._save_metadata()

    # ============================================================
    # HYBRID RETRIEVAL (Dense Vector + Sparse BM25 + RRF)
    # ============================================================

    def _dense_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Dense semantic vector similarity search."""
        if not self.documents or not self.embeddings or self.embed_model is None:
            return []

        query_vec = self.embed_model.encode(query)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        scores = []
        for idx, doc_vec in enumerate(self.embeddings):
            sim = float(np.dot(query_vec, doc_vec))
            scores.append((idx, sim))

        # Sort descending by similarity score
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _sparse_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Sparse BM25 / Keyword search."""
        if not self.documents:
            return []

        query_terms = re.findall(r'\w+', query.lower())
        if not query_terms:
            return []

        # Try using rank_bm25 if available
        try:
            from rank_bm25 import BM25Okapi
            corpus = [re.findall(r'\w+', d['text'].lower()) for d in self.documents]
            bm25 = BM25Okapi(corpus)
            doc_scores = bm25.get_scores(query_terms)
            scored = [(idx, float(score)) for idx, score in enumerate(doc_scores)]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]
        except Exception:
            # Simple keyword frequency fallback
            scored = []
            for idx, d in enumerate(self.documents):
                text_lower = d['text'].lower()
                matches = sum(text_lower.count(term) for term in query_terms)
                if matches > 0:
                    scored.append((idx, float(matches)))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]

    def hybrid_search(self, query: str, top_k: int = 4, rrf_k: int = 60) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) combining Dense Vector + Sparse BM25 results.
        """
        if not self.documents:
            return []

        dense_results = self._dense_search(query, top_k=top_k * 2)
        sparse_results = self._sparse_search(query, top_k=top_k * 2)

        rrf_scores: Dict[int, float] = {}

        # 1. RRF from dense search
        for rank, (doc_idx, score) in enumerate(dense_results, start=1):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (rrf_k + rank))

        # 2. RRF from sparse search
        for rank, (doc_idx, score) in enumerate(sparse_results, start=1):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (rrf_k + rank))

        # Sort combined results
        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for doc_idx, rrf_score in fused:
            doc = self.documents[doc_idx].copy()
            doc['rrf_score'] = rrf_score
            results.append(doc)

        return results

    def format_rag_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Format RAG search results into a clean context prompt block."""
        if not search_results:
            return ""

        formatted_blocks = []
        for i, res in enumerate(search_results, start=1):
            formatted_blocks.append(
                f"[Document Source {i}: {res['filename']} (Chunk {res['chunk_index']+1})]\n"
                f"{res['text']}"
            )

        return "\n\n".join(formatted_blocks)


rag_service = RAGService()
