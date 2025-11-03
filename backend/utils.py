import re
from typing import List, Dict, Optional
from backend.diff_parser import parse_diff_hunks, map_comment_to_diff_position, format_ai_response_for_line_comments


def detect_language_from_filename(filename: str) -> str:
    ext_map = {
    ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
    ".java": "Java", ".cpp": "C++", ".c": "C", ".cs": "C#",
    ".go": "Go", ".rb": "Ruby", ".php": "PHP", ".rs": "Rust",
    ".swift": "Swift", ".kt": "Kotlin", ".scala": "Scala",
    ".html": "HTML", ".css": "CSS", ".sql": "SQL", ".ipynb": "Jupyter Notebook"
    }
    for ext, lang in ext_map.items():
        if filename.endswith(ext):
            return lang
    return "Unknown"




def trim_diff(patch: str, context_window: int = 3) -> str:
    lines = patch.splitlines()
    keep_ranges = []
    for i, line in enumerate(lines):
        if line.startswith(('+', '-')):
            start = max(i - context_window, 0)
            end = min(i + context_window + 1, len(lines))
            keep_ranges.append((start, end))

    # merge overlapping ranges
    merged = []
    for start, end in sorted(keep_ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    # collect lines
    kept = []
    for start, end in merged:
        kept.extend(lines[start:end])

    return "\n".join(kept)





def extract_symbols_from_patch(patch: str) -> List[str]:
    # generic regex for def/class and common function patterns
    syms = re.findall(r'def\s+(\w+)|class\s+(\w+)|function\s+(\w+)', patch)
    flat = [s for t in syms for s in t if s]
    return list(dict.fromkeys(flat))