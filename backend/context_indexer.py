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

def chunk_text(path, max_chunk_size=2000, use_function_boundaries=True):
    """
    Enhanced text chunking that respects function/class boundaries when possible.

    Args:
        path: File path to read
        max_chunk_size: Maximum chunk size in characters
        use_function_boundaries: Whether to try chunking by function boundaries

    Returns:
        List of text chunks with metadata
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        print(f"⚠️ Skipping file {path} due to error: {e}")
        return []

    # Try function-based chunking first
    if use_function_boundaries:
        language = detect_language_from_filename(path)
        if language != "Unknown":
            try:
                return chunk_code_by_functions(content, language, max_chunk_size, path)
            except Exception:
                # Fall back to simple chunking if function-based fails
                pass

    # Simple line-based chunking as fallback
    return chunk_text_by_lines(content, max_chunk_size, path)


def chunk_code_by_functions(content: str, language: str, max_chunk_size: int, file_path: str) -> List[Dict[str, Any]]:
    """
    Chunk code by function/class boundaries with enhanced metadata.
    """
    chunks = []
    boundaries = get_function_boundaries(content, language)

    if not boundaries:
        return chunk_text_by_lines(content, max_chunk_size, file_path)

    # Sort boundaries and create chunks
    boundaries.sort(key=lambda x: x['start_line'])
    lines = content.split('\n')

    for boundary in boundaries:
        start_idx = boundary['start_line'] - 1
        end_idx = min(boundary['end_line'], len(lines))

        if start_idx >= len(lines):
            continue

        function_lines = lines[start_idx:end_idx]
        function_content = '\n'.join(function_lines)

        # If function is too large, split it further
        if len(function_content) > max_chunk_size:
            sub_chunks = chunk_text_by_lines(function_content, max_chunk_size, file_path)
            for sub_chunk in sub_chunks:
                chunks.append({
                    "content": sub_chunk["content"],
                    "file": sub_chunk["file"],
                    "type": "function_subchunk",
                    "function_name": boundary['name'],
                    "function_type": boundary['type'],
                    "start_line": boundary['start_line'],
                    "end_line": boundary['end_line'],
                    "language": language
                })
        else:
            chunks.append({
                "content": function_content,
                "file": file_path,
                "type": boundary['type'],
                "function_name": boundary['name'],
                "function_type": boundary['type'],
                "start_line": boundary['start_line'],
                "end_line": boundary['end_line'],
                "language": language
            })

    # Handle code outside of functions (imports, global variables, etc.)
    all_function_lines = set()
    for boundary in boundaries:
        all_function_lines.update(range(boundary['start_line'] - 1, boundary['end_line']))

    other_lines = []
    for i, line in enumerate(lines):
        if i not in all_function_lines:
            other_lines.append(line)

    if other_lines:
        other_content = '\n'.join(other_lines)
        if len(other_content) > max_chunk_size:
            sub_chunks = chunk_text_by_lines(other_content, max_chunk_size, file_path)
            chunks.extend(sub_chunks)
        else:
            chunks.append({
                "content": other_content,
                "file": file_path,
                "type": "global",
                "language": language
            })

    return chunks


def chunk_text_by_lines(content: str, max_chunk_size: int, file_path: str) -> List[Dict[str, Any]]:
    """
    Simple line-based text chunking with basic metadata.
    """
    chunks = []
    lines = content.split('\n')
    current_chunk = []
    current_length = 0

    for line in lines:
        if current_length + len(line) + 1 > max_chunk_size and current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunks.append({
                "content": chunk_content,
                "file": file_path,
                "type": "text_chunk"
            })
            current_chunk = [line]
            current_length = len(line)
        else:
            current_chunk.append(line)
            current_length += len(line) + 1  # +1 for newline

    if current_chunk:
        chunk_content = '\n'.join(current_chunk)
        chunks.append({
            "content": chunk_content,
            "file": file_path,
            "type": "text_chunk"
        })

    return chunks


def extract_dependencies(repo_dir: str) -> Dict[str, List[str]]:
    """
    Extract dependency relationships between files (imports, function calls, etc.).
    """
    dependencies = {}

    for root, _, files in os.walk(repo_dir):
        for file in files:
            if not file.endswith(('.py', '.js', '.ts', '.java')):
                continue

            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, repo_dir)

            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                file_deps = extract_file_dependencies(content, file)
                dependencies[relative_path] = file_deps

            except Exception as e:
                print(f"⚠️ Could not extract dependencies from {relative_path}: {e}")

    return dependencies


def extract_file_dependencies(content: str, file_path: str) -> List[str]:
    """
    Extract dependencies from a single file.
    """
    deps = []

    if file_path.endswith('.py'):
        # Python imports
        import_patterns = [
            r'from\s+([^\s]+)\s+import',
            r'import\s+([^\s]+)',
            r'from\s+\.([^\s]+)\s+import'  # Relative imports
        ]

        for pattern in import_patterns:
            matches = re.findall(pattern, content)
            deps.extend(matches)

    elif file_path.endswith(('.js', '.ts')):
        # JavaScript/TypeScript imports
        import_patterns = [
            r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]',
            r'require\([\'"]([^\'"]+)[\'"]\)',
            r'import\s+[\'"]([^\'"]+)[\'"]'
        ]

        for pattern in import_patterns:
            matches = re.findall(pattern, content)
            deps.extend(matches)

    elif file_path.endswith('.java'):
        # Java imports
        import_pattern = r'import\s+([^\s;]+);'
        matches = re.findall(import_pattern, content)
        deps.extend(matches)

    return list(set(deps))  # Remove duplicates


def build_dependency_graph(dependencies: Dict[str, List[str]]) -> nx.DiGraph:
    """
    Build a NetworkX dependency graph from file dependencies.
    """
    G = nx.DiGraph()

    # Add nodes
    for file_path in dependencies.keys():
        G.add_node(file_path)

    # Add edges (dependencies)
    for file_path, deps in dependencies.items():
        for dep in deps:
            # Try to find the actual file that corresponds to this dependency
            dep_file = find_dependency_file(dep, dependencies.keys())
            if dep_file and dep_file != file_path:
                G.add_edge(file_path, dep_file)

    return G


def find_dependency_file(dep: str, all_files: List[str]) -> Optional[str]:
    """
    Find the actual file that corresponds to a dependency string.
    """
    # Direct match
    if dep in all_files:
        return dep

    # Try common patterns
    for file_path in all_files:
        if dep.replace('.', '/') in file_path or dep.replace('\\', '/') in file_path:
            return file_path

        # Check filename matches
        if file_path.endswith(f"/{dep}.py") or file_path.endswith(f"/{dep}.js") or file_path.endswith(f"/{dep}.ts"):
            return file_path

    return None


def get_related_files(file_path: str, dependency_graph: nx.DiGraph, max_depth: int = 2) -> List[str]:
    """
    Get files related to the given file through dependency relationships.
    """
    related = set()

    try:
        # Files that this file depends on
        related.update(dependency_graph.successors(file_path))

        # Files that depend on this file
        related.update(dependency_graph.predecessors(file_path))

        # Go deeper if requested
        if max_depth > 1:
            current_related = related.copy()
            for _ in range(max_depth - 1):
                next_related = set()
                for related_file in current_related:
                    next_related.update(dependency_graph.successors(related_file))
                    next_related.update(dependency_graph.predecessors(related_file))
                related.update(next_related)
                current_related = next_related

    except Exception as e:
        print(f"⚠️ Error getting related files for {file_path}: {e}")

    return list(related)

def index_repo(repo_dir: str, repo_name: str, commit_sha: str) -> Tuple[faiss.Index, List[dict]]:
    """
    Enhanced repository indexing with function-level chunking and dependency analysis.
    Returns the index and the metadata for each chunk.
    """
    # Check if a pre-existing index exists for this commit
    index_path = INDEX_DIR / f"{repo_name.replace('/', '__')}_{commit_sha}.faiss"
    metadata_path = INDEX_DIR / f"{repo_name.replace('/', '__')}_{commit_sha}_meta.json"
    dependency_path = INDEX_DIR / f"{repo_name.replace('/', '__')}_{commit_sha}_deps.json"

    if index_path.exists() and metadata_path.exists():
        print(f"✅ Loading existing index for {repo_name}@{commit_sha}")
        index = faiss.read_index(str(index_path))
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        return index, metadata

    print(f"⏳ Building enhanced index for {repo_name}@{commit_sha}")
    all_chunks = []
    metadatas = []

    # Extract dependencies first
    print("🔍 Analyzing file dependencies...")
    dependencies = extract_dependencies(repo_dir)
    dependency_graph = build_dependency_graph(dependencies)

    # Save dependency graph
    with open(dependency_path, "w") as f:
        # Convert NetworkX graph to serializable format
        serializable_deps = {
            "nodes": list(dependency_graph.nodes()),
            "edges": list(dependency_graph.edges()),
            "dependencies": dependencies
        }
        json.dump(serializable_deps, f)

    # Process files with enhanced chunking
    for root, _, files in os.walk(repo_dir):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, repo_dir)

            # Skip binary or large non-text files
            if file.endswith((".png", ".jpg", ".jpeg", ".gif", ".exe", ".dll", ".zip", ".pdf")):
                continue

            # Get related files for enhanced context
            related_files = get_related_files(relative_path, dependency_graph)

            # Enhanced chunking
            file_chunks = chunk_text(file_path, use_function_boundaries=True)

            for chunk_data in file_chunks:
                chunk_content = chunk_data["content"]

                # Enhanced metadata
                metadata = {
                    "repo": repo_name,
                    "commit": commit_sha,
                    "file": relative_path,
                    "content": chunk_content,
                    "type": chunk_data.get("type", "text_chunk"),
                    "language": chunk_data.get("language", detect_language_from_filename(relative_path)),
                    "dependencies": dependencies.get(relative_path, []),
                    "related_files": related_files
                }

                # Add function-specific metadata
                if "function_name" in chunk_data:
                    metadata["function_name"] = chunk_data["function_name"]
                    metadata["function_type"] = chunk_data["function_type"]
                    metadata["start_line"] = chunk_data["start_line"]
                    metadata["end_line"] = chunk_data["end_line"]

                metadatas.append(metadata)
                all_chunks.append(chunk_content)

    if not all_chunks:
        return None, []

    # Create embeddings
    print("🧠 Creating embeddings...")
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

    print(f"✅ Indexed {len(all_chunks)} chunks from {len(set(m['file'] for m in metadatas))} files")
    return index, metadatas


def load_dependency_graph(repo_name: str, commit_sha: str) -> Optional[nx.DiGraph]:
    """
    Load the dependency graph for a repository if it exists.
    """
    dependency_path = INDEX_DIR / f"{repo_name.replace('/', '__')}_{commit_sha}_deps.json"

    if not dependency_path.exists():
        return None

    try:
        with open(dependency_path, "r") as f:
            data = json.load(f)

        G = nx.DiGraph()
        G.add_nodes_from(data["nodes"])
        G.add_edges_from(data["edges"])

        return G
    except Exception as e:
        print(f"⚠️ Could not load dependency graph: {e}")
        return None