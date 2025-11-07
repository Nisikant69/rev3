# backend/diff_parser.py
"""
Utilities for parsing GitHub diff patches and mapping AI review comments
to exact line positions in pull requests.
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DiffHunk:
    """Represents a single hunk in a GitHub diff."""
    old_start: int
    old_lines: int
    new_start: int
    new_lines: int
    content: str
    lines: List[str]

    @property
    def diff_header(self) -> str:
        """Returns the hunk header line."""
        return f"@@ -{self.old_start},{self.old_lines} +{self.new_start},{self.new_lines} @@"


@dataclass
class CommentPosition:
    """Represents a comment position in a diff."""
    path: str
    body: str
    position: int  # Position in the diff (1-based)
    line: Optional[int] = None  # Optional line number in the new file


def parse_diff_hunks(patch: str) -> List[DiffHunk]:
    """
    Parse a GitHub diff patch into individual hunks with line number mapping.

    Args:
        patch: The GitHub diff patch string

    Returns:
        List of DiffHunk objects representing each hunk in the diff
    """
    if not patch:
        return []

    lines = patch.splitlines()
    hunks = []
    current_hunk_lines = []

    # Pattern to match hunk headers: @@ -old_start,old_lines +new_start,new_lines @@
    hunk_header_pattern = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')

    for line in lines:
        if hunk_header_pattern.match(line):
            # Save previous hunk if it exists
            if current_hunk_lines:
                hunk = _create_hunk_from_lines(current_hunk_lines)
                if hunk:
                    hunks.append(hunk)
                current_hunk_lines = []

            # Start new hunk
            current_hunk_lines.append(line)
        elif current_hunk_lines:
            # Continue collecting lines for current hunk
            current_hunk_lines.append(line)

    # Don't forget the last hunk
    if current_hunk_lines:
        hunk = _create_hunk_from_lines(current_hunk_lines)
        if hunk:
            hunks.append(hunk)

    return hunks


def _create_hunk_from_lines(lines: List[str]) -> Optional[DiffHunk]:
    """Create a DiffHunk from a list of lines."""
    if not lines:
        return None

    # Parse header
    header_line = lines[0]
    match = re.match(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', header_line)
    if not match:
        return None

    old_start = int(match.group(1))
    old_lines = int(match.group(2) or 1)
    new_start = int(match.group(3))
    new_lines = int(match.group(4) or 1)

    content = "\n".join(lines[1:])  # Exclude header from content

    return DiffHunk(
        old_start=old_start,
        old_lines=old_lines,
        new_start=new_start,
        new_lines=new_lines,
        content=content,
        lines=lines[1:]  # Exclude header
    )


def map_comment_to_diff_position(comment: str, hunks: List[DiffHunk],
                                filename: str) -> Optional[CommentPosition]:
    """
    Map an AI comment to the appropriate position in the diff.

    Args:
        comment: The AI-generated comment text
        hunks: List of DiffHunk objects from the diff
        filename: The file path the comment applies to

    Returns:
        CommentPosition object with the mapped position, or None if no mapping found
    """
    # Try to extract line numbers from the comment
    line_references = extract_line_references(comment)

    if line_references:
        # Use explicit line references from the comment
        for line_num in line_references:
            position = find_position_for_line(line_num, hunks)
            if position:
                return CommentPosition(
                    path=filename,
                    body=comment,
                    position=position,
                    line=line_num  # Also set the line number for clarity
                )

    # Fallback: place comment at the first relevant hunk
    if hunks:
        # Find the first hunk with additions (lines starting with '+')
        for i, hunk in enumerate(hunks):
            if any(line.startswith('+') and not line.startswith('++') for line in hunk.lines):
                # Position is the line number in the diff (1-based)
                # Calculate the position as the cumulative line count up to this hunk
                position = 1  # Start after the hunk header
                for prev_hunk in hunks[:i]:
                    position += len(prev_hunk.lines) + 1  # +1 for header
                position += 1  # +1 for this hunk's header

                # Find the line number in the new file for this position
                new_line = find_line_for_position(position, hunks)
                if new_line:
                    return CommentPosition(
                        path=filename,
                        body=comment,
                        position=position,
                        line=new_line
                    )
                else:
                    # If we can't find the exact line, just use position
                    return CommentPosition(
                        path=filename,
                        body=comment,
                        position=position,
                        line=None  # Line will be omitted
                    )

        # If no additions found, use the first hunk
        return CommentPosition(
            path=filename,
            body=comment,
            position=1,
            line=hunks[0].new_start if hunks else 1
        )

    return None


def find_line_for_position(position: int, hunks: List[DiffHunk]) -> Optional[int]:
    """
    Find the line number in the new file for a given diff position.

    Args:
        position: The diff position (1-based)
        hunks: List of DiffHunk objects

    Returns:
        Line number in the new file, or None if not found
    """
    current_new_line = 1
    diff_position = 1  # Start after first hunk header

    for hunk in hunks:
        diff_position += 1  # Account for hunk header

        # Skip to the start of this hunk in the new file
        while current_new_line < hunk.new_start:
            current_new_line += 1
            diff_position += 1

        # Check if the target position is within this hunk's range
        hunk_start_position = diff_position
        hunk_end_position = diff_position + len(hunk.lines) - 1

        if hunk_start_position <= position <= hunk_end_position:
            # Calculate the line offset within this hunk
            position_offset = position - hunk_start_position

            # Count new file lines until we reach the target position
            for line in hunk.lines:
                if line.startswith('+') and not line.startswith('++'):
                    if position_offset == 0:
                        return current_new_line
                    current_new_line += 1
                    position_offset -= 1
                elif line.startswith('-'):
                    # Skip deleted lines for position counting
                    continue
                else:
                    # Context lines
                    current_new_line += 1
                    position_offset -= 1

        # Skip to the end of this hunk
        diff_position += len(hunk.lines)

    return None


def extract_line_references(comment: str) -> List[int]:
    """
    Extract line number references from AI comment text.

    Args:
        comment: The AI-generated comment text

    Returns:
        List of line numbers mentioned in the comment
    """
    # Pattern to match line references like "line 42", "L42", "line 42-45", etc.
    patterns = [
        r'line\s+(\d+)',
        r'L(\d+)',
        r'at\s+line\s+(\d+)',
        r'on\s+line\s+(\d+)',
        r'(\d+)(?:-\d+)?\s*[:\)]',  # Like "42:" or "42)"
    ]

    line_numbers = []
    for pattern in patterns:
        matches = re.findall(pattern, comment, re.IGNORECASE)
        line_numbers.extend([int(match) for match in matches])

    # Also look for line ranges like "lines 42-45"
    range_pattern = r'lines?\s+(\d+)\s*-\s*(\d+)'
    range_matches = re.findall(range_pattern, comment, re.IGNORECASE)
    for start, end in range_matches:
        line_numbers.extend(range(int(start), int(end) + 1))

    return sorted(list(set(line_numbers)))


def find_position_for_line(line_num: int, hunks: List[DiffHunk]) -> Optional[int]:
    """
    Find the diff position for a specific line number in the new file.

    Args:
        line_num: The line number in the new file
        hunks: List of DiffHunk objects

    Returns:
        Position in the diff (1-based), or None if not found
    """
    current_new_line = 1
    diff_position = 1  # Start after first hunk header

    for hunk in hunks:
        diff_position += 1  # Account for hunk header

        # Skip to the start of this hunk in the new file
        while current_new_line < hunk.new_start:
            current_new_line += 1
            diff_position += 1

        # Check if the target line is within this hunk's range
        if hunk.new_start <= line_num < hunk.new_start + hunk.new_lines:
            # Calculate position within this hunk
            line_offset = line_num - hunk.new_start

            # Count lines in hunk until we reach the target
            hunk_line_count = 0
            for line in hunk.lines:
                hunk_line_count += 1
                if not line.startswith('-'):  # Skip deleted lines
                    if current_new_line == line_num:
                        return diff_position
                    current_new_line += 1
                diff_position += 1

        # Skip to the end of this hunk
        diff_position += len(hunk.lines) - hunk_line_count

    return None


def validate_diff_position(position: int, hunks: List[DiffHunk]) -> bool:
    """
    Validate that a diff position is valid for the given hunks.

    Args:
        position: The diff position (1-based)
        hunks: List of DiffHunk objects

    Returns:
        True if position is valid, False otherwise
    """
    if position < 1:
        return False

    total_lines = sum(len(hunk.lines) + 1 for hunk in hunks)  # +1 for each header

    return position <= total_lines


def split_comments_by_hunk(comments: List[CommentPosition], hunks: List[DiffHunk]) -> Dict[int, List[CommentPosition]]:
    """
    Organize comments by which hunk they belong to.

    Args:
        comments: List of CommentPosition objects
        hunks: List of DiffHunk objects

    Returns:
        Dictionary mapping hunk index to list of comments
    """
    hunk_comments = {i: [] for i in range(len(hunks))}

    # Calculate the start position for each hunk
    hunk_positions = []
    current_position = 1
    for hunk in hunks:
        hunk_positions.append(current_position + 1)  # +1 for header
        current_position += len(hunk.lines) + 1  # +1 for header

    for comment in comments:
        # Find which hunk this comment belongs to
        for i, hunk_start in enumerate(hunk_positions):
            hunk_end = hunk_start + len(hunks[i].lines) - 1
            if hunk_start <= comment.position <= hunk_end:
                hunk_comments[i].append(comment)
                break

    return hunk_comments


def format_ai_response_for_line_comments(ai_response: str) -> List[str]:
    """
    Format AI response into individual line-specific comments.

    Args:
        ai_response: The raw AI review response

    Returns:
        List of individual comment strings
    """
    # Split by common delimiters that indicate separate issues
    delimiters = [
        r'\n\d+\.\s+',  # Numbered lists: "1. ", "2. ", etc.
        r'\n-\s+',      # Bullet points: "- "
        r'\n\*\s+',     # Bullet points: "* "
        r'\n---',       # Horizontal rules
        r'\n\n',        # Double newlines (separate paragraphs)
    ]

    comments = [ai_response.strip()]

    for delimiter in delimiters:
        new_comments = []
        for comment in comments:
            parts = re.split(delimiter, comment)
            if len(parts) > 1 and all(part.strip() for part in parts):
                new_comments.extend([part.strip() for part in parts if part.strip()])
            else:
                new_comments.append(comment)
        comments = new_comments

    # Filter out very short comments and comments that don't look actionable
    actionable_comments = []
    for comment in comments:
        if len(comment) > 10 and any(keyword in comment.lower() for keyword in [
            'consider', 'suggest', 'recommend', 'fix', 'issue', 'problem',
            'bug', 'error', 'improve', 'optimize', 'refactor', 'security'
        ]):
            actionable_comments.append(comment)

    return actionable_comments if actionable_comments else [ai_response.strip()]