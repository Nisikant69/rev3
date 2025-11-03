# backend/context_indexer.py
import os
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import networkx as nx

from backend.config import INDEX_DIR, CHUNK_SIZE, CHUNK_OVERLAP, REPO_CACHE_DIR
from backend.utils import detect_language_from_filename, get_function_boundaries

MODEL_NAME = "all-MiniLM-L6-v2"  # or any local sentence-transformers model
_model = None

def _get_model():
    """Lazily loads the SentenceTransformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def chunk_text(path, max_chunk_size=2000):
    """
    Reads a file line by line and splits into manageable text chunks.
    """
    chunks = []
    current_chunk = []
    current_length = 0

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                # Add a small buffer to prevent overshooting the max_chunk_size
                if current_length + len(line) > max_chunk_size and current_chunk:
                    chunks.append("".join(current_chunk))
                    current_chunk = [line]
                    current_length = len(line)
                else:
                    current_chunk.append(line)
                    current_length += len(line)

            if current_chunk:  # flush last chunk
                chunks.append("".join(current_chunk))

    except Exception as e:
        print(f"⚠️ Skipping file {path} due to error: {e}")

    return chunks

def index_repo(repo_dir: str, repo_name: str, commit_sha: str) -> Tuple[faiss.Index, List[dict]]:
    """
    Walks through the repo, processes files, chunks them, and builds a FAISS index.
    Returns the index and the metadata for each chunk.
    """
    # Check if a pre-existing index exists for this commit
    index_path = INDEX_DIR / f"{repo_name.replace('/', '__')}_{commit_sha}.faiss"
    metadata_path = INDEX_DIR / f"{repo_name.replace('/', '__')}_{commit_sha}_meta.json"
    
    if index_path.exists() and metadata_path.exists():
        print(f"✅ Loading existing index for {repo_name}@{commit_sha}")
        index = faiss.read_index(str(index_path))
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        return index, metadata

    print(f"⏳ Building new index for {repo_name}@{commit_sha}")
    all_chunks = []
    metadatas = []

    for root, _, files in os.walk(repo_dir):
        for file in files:
            file_path = os.path.join(root, file)
            # Make the file path relative to the repo root for clean metadata
            relative_path = os.path.relpath(file_path, repo_dir)
            
            # Skip binary or large non-text files
            if file.endswith((".png", ".jpg", ".jpeg", ".gif", ".exe", ".dll", ".zip")):
                continue

            file_chunks = chunk_text(file_path)

            for chunk in file_chunks:
                metadatas.append({
                    "repo": repo_name,
                    "commit": commit_sha,
                    "file": relative_path,
                    "content": chunk
                })
                all_chunks.append(chunk)

    if not all_chunks:
        return None, []

    # Create embeddings
    model = _get_model()
    embeddings = model.encode(all_chunks).astype("float32")

    # Create and populate FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save the index and metadata for future use
    faiss.write_index(index, str(index_path))
    with open(metadata_path, "w") as f:
        json.dump(metadatas, f)

    return index, metadatas