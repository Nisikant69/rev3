# backend/semantic_search.py
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from backend.config import TOP_K
from backend.context_indexer import _get_model, load_dependency_graph, get_related_files


def semantic_search(query: str, index: faiss.Index, metadata: List[dict],
                   repo_name: str = None, commit_sha: str = None,
                   file_path: str = None, enhanced: bool = True) -> List[dict]:
    """
    Enhanced semantic search with dependency-aware context retrieval.

    Args:
        query: Search query
        index: FAISS index
        metadata: List of chunk metadata
        repo_name: Repository name (for dependency graph loading)
        commit_sha: Commit SHA (for dependency graph loading)
        file_path: Current file being analyzed (for dependency context)
        enhanced: Whether to use enhanced search features

    Returns:
        List of relevant context chunks
    """
    if index is None or not metadata:
        return []

    model = _get_model()
    q_emb = model.encode([query])[0].astype("float32")

    # Perform initial semantic search
    D, I = index.search(np.array([q_emb]), TOP_K)

    results = []
    for idx in I[0]:
        if idx < 0 or idx >= len(metadata):
            continue
        results.append(metadata[idx])

    # Enhanced search: add dependency-aware context
    if enhanced and file_path and repo_name and commit_sha:
        dependency_graph = load_dependency_graph(repo_name, commit_sha)
        if dependency_graph:
            related_files = get_related_files(file_path, dependency_graph)
            dependency_context = get_dependency_context(metadata, related_files)
            results.extend(dependency_context)

    # Deduplicate and rank results
    unique_results = deduplicate_results(results)
    ranked_results = rank_results_by_relevance(unique_results, query, file_path)

    return ranked_results[:TOP_K]


def deduplicate_results(results: List[dict]) -> List[dict]:
    """
    Remove duplicate results while preserving order.
    """
    seen = set()
    unique_results = []

    for result in results:
        # Create a unique identifier based on file and content hash
        identifier = f"{result['file']}_{hash(result['content']) % 10000}"
        if identifier not in seen:
            seen.add(identifier)
            unique_results.append(result)

    return unique_results


def rank_results_by_relevance(results: List[dict], query: str, current_file: str = None) -> List[dict]:
    """
    Re-rank search results based on multiple relevance factors.
    """
    if not results:
        return results

    model = _get_model()
    query_emb = model.encode([query])[0]

    scored_results = []
    for result in results:
        score = 0.0

        # Semantic similarity score
        content_emb = model.encode([result["content"]])[0]
        semantic_score = np.dot(query_emb, content_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(content_emb))
        score += semantic_score * 0.6

        # File type relevance
        if current_file:
            # Boost same file
            if result["file"] == current_file:
                score += 0.3
            # Boost related files (same directory)
            elif (result["file"].split("/")[0] if "/" in result["file"] else result["file"]) == \
                 (current_file.split("/")[0] if "/" in current_file else current_file):
                score += 0.2

        # Code type relevance
        content_type = result.get("type", "text_chunk")
        if content_type in ["function", "class", "method"]:
            score += 0.1  # Boost functions/classes
        elif content_type == "global":
            score += 0.05  # Slight boost for global code

        # Language matching
        if current_file and result.get("language"):
            current_lang = detect_language_from_filename(current_file)
            if result.get("language") == current_lang:
                score += 0.1

        scored_results.append((score, result))

    # Sort by score (descending) and return just the results
    scored_results.sort(key=lambda x: x[0], reverse=True)
    return [result for _, result in scored_results]


def detect_language_from_filename(filename: str) -> str:
    """
    Simple language detection from filename (moved from utils to avoid circular imports).
    """
    ext_map = {
        ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
        ".java": "Java", ".cpp": "C++", ".c": "C", ".cs": "C#",
        ".go": "Go", ".rb": "Ruby", ".php": "PHP", ".rs": "Rust",
        ".swift": "Swift", ".kt": "Kotlin", ".scala": "Scala",
        ".html": "HTML", ".css": "CSS", ".sql": "SQL"
    }
    for ext, lang in ext_map.items():
        if filename.endswith(ext):
            return lang
    return "Unknown"


def get_dependency_context(metadata: List[dict], related_files: List[str]) -> List[dict]:
    """
    Get context chunks from files that are related through dependencies.
    """
    dependency_context = []

    for chunk in metadata:
        if chunk["file"] in related_files:
            # Boost priority for dependency chunks
            chunk["dependency_boost"] = True
            dependency_context.append(chunk)

    return dependency_context


def hybrid_search(query: str, index: faiss.Index, metadata: List[dict],
                 repo_name: str = None, commit_sha: str = None,
                 file_path: str = None) -> List[dict]:
    """
    Hybrid search combining semantic and keyword-based search.
    """
    # Semantic search results
    semantic_results = semantic_search(query, index, metadata, repo_name, commit_sha, file_path)

    # Keyword-based search results
    keyword_results = keyword_search(query, metadata)

    # Combine and re-rank
    combined_results = semantic_results + keyword_results
    unique_results = deduplicate_results(combined_results)
    ranked_results = rank_results_by_relevance(unique_results, query, file_path)

    return ranked_results[:TOP_K]


def keyword_search(query: str, metadata: List[dict]) -> List[dict]:
    """
    Simple keyword-based search as fallback or supplement to semantic search.
    """
    query_terms = query.lower().split()
    results = []

    for chunk in metadata:
        content_lower = chunk["content"].lower()
        file_lower = chunk["file"].lower()

        # Count keyword matches
        matches = 0
        for term in query_terms:
            matches += content_lower.count(term)
            matches += file_lower.count(term)

        if matches > 0:
            chunk["keyword_score"] = matches
            results.append(chunk)

    # Sort by keyword matches
    results.sort(key=lambda x: x["keyword_score"], reverse=True)
    return results


def get_function_context(metadata: List[dict], function_name: str, file_path: str = None) -> List[dict]:
    """
    Get context specifically for a given function name.
    """
    function_results = []

    for chunk in metadata:
        # Check if chunk contains the function
        if chunk.get("function_name") == function_name:
            if not file_path or chunk["file"] == file_path:
                function_results.append(chunk)

        # Also search for function mentions in content
        elif function_name.lower() in chunk["content"].lower():
            if not file_path or chunk["file"] == file_path:
                chunk["contains_function_mention"] = True
                function_results.append(chunk)

    return function_results


def get_class_context(metadata: List[dict], class_name: str, file_path: str = None) -> List[dict]:
    """
    Get context specifically for a given class name.
    """
    class_results = []

    for chunk in metadata:
        # Check if chunk contains the class
        if chunk.get("function_name") == class_name and chunk.get("function_type") == "class":
            if not file_path or chunk["file"] == file_path:
                class_results.append(chunk)

        # Also search for class mentions in content
        elif class_name.lower() in chunk["content"].lower():
            if not file_path or chunk["file"] == file_path:
                chunk["contains_class_mention"] = True
                class_results.append(chunk)

    return class_results


def search_by_file_type(metadata: List[dict], file_type: str) -> List[dict]:
    """
    Search for chunks by file type or extension.
    """
    results = []

    for chunk in metadata:
        file_ext = chunk["file"].split(".")[-1] if "." in chunk["file"] else ""
        language = chunk.get("language", "").lower()

        if file_type.lower() in file_ext.lower() or file_type.lower() in language:
            results.append(chunk)

    return results


def search_recent_changes(metadata: List[dict], commit_sha: str) -> List[dict]:
    """
    Search for chunks from a specific commit (for recent changes).
    """
    return [chunk for chunk in metadata if chunk.get("commit") == commit_sha]