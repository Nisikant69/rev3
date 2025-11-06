import re
from typing import List, Dict, Optional
from diff_parser import parse_diff_hunks as diff_parser_parse_hunks, map_comment_to_diff_position, format_ai_response_for_line_comments


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


def parse_diff_hunks(patch: str):
    """
    Parse diff into hunks for line-level comment positioning.

    Args:
        patch: The GitHub diff patch string

    Returns:
        List of DiffHunk objects
    """
    return diff_parser_parse_hunks(patch)


def map_comment_to_position(comment: str, hunks, filename: str):
    """
    Map an AI comment to the appropriate position in the diff.

    Args:
        comment: The AI-generated comment text
        hunks: List of DiffHunk objects
        filename: The file path

    Returns:
        CommentPosition object or None
    """
    return map_comment_to_diff_position(comment, hunks, filename)


def validate_diff_position(position: int, hunks) -> bool:
    """
    Validate that a diff position is valid.

    Args:
        position: The diff position (1-based)
        hunks: List of DiffHunk objects

    Returns:
        True if position is valid
    """
    from backend.diff_parser import validate_diff_position
    return validate_diff_position(position, hunks)


def format_ai_comments(ai_response: str) -> List[str]:
    """
    Format AI response into individual line-specific comments.

    Args:
        ai_response: The raw AI review response

    Returns:
        List of individual comment strings
    """
    return format_ai_response_for_line_comments(ai_response)


def get_function_boundaries(code: str, language: str) -> List[Dict]:
    """
    Extract function/class boundaries from code for intelligent chunking.

    Args:
        code: The source code string
        language: The programming language

    Returns:
        List of dictionaries with function boundaries
    """
    boundaries = []

    if language.lower() == 'python':
        # Python functions and classes
        pattern = r'^(\s*)(def|class)\s+(\w+)\s*\([^)]*\)\s*:|^(\s*)(def|class)\s+(\w+)\s*:'
        matches = re.finditer(pattern, code, re.MULTILINE)

        for match in matches:
            start_line = code[:match.start()].count('\n') + 1
            indent = len(match.group(1) or match.group(4) or '')
            name = match.group(3) or match.group(6)

            # Find the end of the function/class
            remaining_code = code[match.start():]
            lines = remaining_code.split('\n')
            end_line = start_line

            base_indent = indent
            for i, line in enumerate(lines[1:], 1):
                if line.strip() == '':
                    continue
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= base_indent and line.strip():
                    break
                end_line = start_line + i

            boundaries.append({
                'name': name,
                'type': match.group(2) or match.group(5),
                'start_line': start_line,
                'end_line': end_line,
                'indent': indent
            })

    elif language.lower() in ['javascript', 'typescript']:
        # JavaScript/TypeScript functions and classes
        patterns = [
            r'^\s*(function\s+\w+\s*\([^)]*\)\s*\{|const\s+\w+\s*=\s*(?:function\s*\([^)]*\)\s*\{|\([^)]*\)\s*=>)|class\s+\w+)',
            r'^\s*\w+\s*\([^)]*\)\s*\{',  # Method definitions
            r'^\s*\w+\s*:\s*function\s*\([^)]*\)\s*\{',  # Object methods
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, code, re.MULTILINE)
            for match in matches:
                start_line = code[:match.start()].count('\n') + 1

                # Find matching brace
                brace_count = 0
                in_function = False
                remaining_code = code[match.start():]
                end_line = start_line

                for i, char in enumerate(remaining_code):
                    if char == '{':
                        brace_count += 1
                        in_function = True
                    elif char == '}':
                        brace_count -= 1
                        if in_function and brace_count == 0:
                            end_line = start_line + remaining_code[:i].count('\n')
                            break

                boundaries.append({
                    'name': match.group().split('(')[0].strip().replace('function ', '').replace('const ', '').replace('= ', ''),
                    'type': 'function',
                    'start_line': start_line,
                    'end_line': end_line
                })

    return boundaries


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for a text string.
    Rough approximation: ~4 characters per token for English text.

    Args:
        text: The text to estimate tokens for

    Returns:
        Estimated token count
    """
    # Simple approximation: ~4 characters per token
    return len(text) // 4


def should_chunk_file(content: str, max_tokens: int = 8000) -> bool:
    """
    Determine if a file should be chunked based on token count.

    Args:
        content: The file content
        max_tokens: Maximum tokens per chunk

    Returns:
        True if file should be chunked
    """
    return estimate_tokens(content) > max_tokens


def chunk_code_by_functions(code: str, language: str, max_tokens: int = 8000) -> List[str]:
    """
    Chunk code by function/class boundaries for better context preservation.

    Args:
        code: The source code
        language: The programming language
        max_tokens: Maximum tokens per chunk

    Returns:
        List of code chunks
    """
    if not should_chunk_file(code, max_tokens):
        return [code]

    boundaries = get_function_boundaries(code, language)
    if not boundaries:
        # Fallback to line-based chunking
        return chunk_code_by_lines(code, max_tokens)

    chunks = []
    lines = code.split('\n')

    # Sort boundaries by start line
    boundaries.sort(key=lambda x: x['start_line'])

    current_chunk = []
    current_tokens = 0

    for boundary in boundaries:
        function_lines = lines[boundary['start_line']-1:boundary['end_line']]
        function_code = '\n'.join(function_lines)
        function_tokens = estimate_tokens(function_code)

        # If adding this function would exceed the limit, start a new chunk
        if current_tokens + function_tokens > max_tokens and current_chunk:
            chunks.append('\n'.join(current_chunk))
            current_chunk = []
            current_tokens = 0

        current_chunk.extend(function_lines)
        current_tokens += function_tokens

    # Add the last chunk
    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    return chunks if chunks else [code]


def chunk_code_by_lines(code: str, max_tokens: int = 8000, min_lines: int = 10) -> List[str]:
    """
    Chunk code by line count when function-based chunking isn't possible.

    Args:
        code: The source code
        max_tokens: Maximum tokens per chunk
        min_lines: Minimum lines per chunk

    Returns:
        List of code chunks
    """
    lines = code.split('\n')
    if len(lines) <= min_lines:
        return [code]

    chunks = []
    current_chunk = []
    current_tokens = 0

    for line in lines:
        line_tokens = estimate_tokens(line + '\n')

        if current_tokens + line_tokens > max_tokens and current_chunk:
            chunks.append('\n'.join(current_chunk))
            current_chunk = []
            current_tokens = 0

        current_chunk.append(line)
        current_tokens += line_tokens

    # Add the last chunk
    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    return chunks if chunks else [code]


def enhance_trim_diff(patch: str, context_window: int = 3, preserve_function_context: bool = True) -> str:
    """
    Enhanced version of trim_diff that preserves function context when possible.

    Args:
        patch: The diff patch
        context_window: Number of context lines to preserve
        preserve_function_context: Whether to preserve entire function context

    Returns:
        Trimmed diff with enhanced context
    """
    if preserve_function_context:
        # Try to extract function context around changes
        lines = patch.splitlines()
        enhanced_lines = []

        for i, line in enumerate(lines):
            if line.startswith(('+', '-')):
                # Find function definition above this change
                start = max(i - context_window * 2, 0)
                for j in range(i - 1, start - 1, -1):
                    if j < len(lines):
                        enhanced_lines.insert(0, lines[j])
                        if any(keyword in lines[j] for keyword in ['def ', 'class ', 'function ', 'class ']):
                            break

                enhanced_lines.append(line)

                # Look for more context after the change
                end = min(i + context_window * 2, len(lines))
                for j in range(i + 1, end):
                    if j < len(lines):
                        enhanced_lines.append(lines[j])

        return '\n'.join(list(dict.fromkeys(enhanced_lines)))  # Remove duplicates
    else:
        # Fall back to original trim_diff
        return trim_diff(patch, context_window)