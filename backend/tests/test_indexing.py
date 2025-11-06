import os
from context_indexer import index_repo
from semantic_search import semantic_search

def test_index_and_search(tmp_path):
    # Create fake repo dir
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    f = repo_dir / "test.py"
    f.write_text("def hello_world():\n    return 'hi'\n")

    repo = "demo/repo"
    sha = "abc123"

    index_repo(repo, sha, str(repo_dir))
    results = semantic_search("hello_world", repo, sha, top_k=1)
    assert any("test.py" in r["path"] for r in results)
