import os
import logging
from github import Repository
from config import REPO_CACHE_DIR, MAX_FILE_CHARS

logger = logging.getLogger(__name__)


def save_repo_snapshot(repo: Repository, ref: str) -> str:
    """Fetch all repo files at ref (commit/branch) and store under cache path.
    Returns path to snapshot root.
    """
    safe_name = repo.full_name.replace('/', '__')
    snapshot_dir = REPO_CACHE_DIR / f"{safe_name}_{ref}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # GitHub API: recursively list contents
    stack = [""]
    while stack:
        path = stack.pop()
        try:
            contents = repo.get_contents(path, ref=ref)
        except Exception as e:
            logger.warning(f"Failed to list {path}@{ref}: {e}")
            continue

        contents = contents if isinstance(contents, list) else [contents]
        for c in contents:
            if c.type == "dir":
                stack.append(c.path)
                continue
            if c.type != "file":
                continue

            try:
                raw = c.decoded_content
                # enforce char limit (decode -> slice -> re-encode)
                raw_text = raw.decode("utf-8", errors="ignore")
                if len(raw_text) > MAX_FILE_CHARS:
                    raw_text = raw_text[:MAX_FILE_CHARS]
                raw = raw_text.encode("utf-8")

                target_path = snapshot_dir / c.path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with open(target_path, "wb") as f:
                    f.write(raw)
            except Exception as e:
                logger.warning(f"Failed to fetch file {c.path}: {e}")
                continue

    return str(snapshot_dir)
