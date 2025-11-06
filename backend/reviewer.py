# backend/reviewer.py
import json
import google.generativeai as genai
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    trim_diff, extract_symbols_from_patch, detect_language_from_filename,
    parse_diff_hunks, map_comment_to_position, format_ai_comments,
    enhance_trim_diff, estimate_tokens, should_chunk_file, chunk_code_by_functions
)
from semantic_search import semantic_search
from config import GEMINI_API_KEY, TOP_K, MAX_TOKENS_PER_REQUEST
import faiss
from typing import List, Dict, Any
import time
from google.api_core.exceptions import ResourceExhausted

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# A rough estimate for a safe token limit for a single API call.
# This might need to be adjusted based on the specific model and its limits.
SAFE_TOKEN_LIMIT = 5000

def review_patch_line_level(patch: str, filename: str, repo_full_name: str, ref: str,
                          commit_sha: str, index: faiss.Index, metadata: List[dict]) -> Dict[str, Any]:
    """
    Generate line-specific AI review comments for a given patch.

    Returns structured data with individual comments that can be mapped to exact line positions.
    """
    language = detect_language_from_filename(str(filename))
    symbols = extract_symbols_from_patch(patch)
    hunks = parse_diff_hunks(patch)

    if not hunks:
        return {
            "file": filename,
            "language": language,
            "symbols": symbols,
            "comments": [],
            "summary": "No changes to review.",
            "context_used": [],
            "hunks": []
        }

    # Get semantic context from repo index
    trimmed = enhance_trim_diff(patch)
    context_chunks = semantic_search(trimmed, index, metadata)
    context_chunks = context_chunks[:TOP_K]

    # Check if we need to chunk the patch
    if should_chunk_file(patch, MAX_TOKENS_PER_REQUEST):
        comments = review_large_patch_in_chunks(patch, filename, language, symbols, context_chunks, hunks)
    else:
        comments = review_single_patch(patch, filename, language, symbols, context_chunks, hunks)

    # Generate summary
    summary = generate_review_summary(comments, filename, language)

    return {
        "file": filename,
        "language": language,
        "symbols": symbols,
        "comments": comments,
        "summary": summary,
        "context_used": context_chunks,
        "hunks": [{"header": hunk.diff_header, "start": hunk.new_start, "lines": hunk.new_lines} for hunk in hunks]
    }


def review_single_patch(patch: str, filename: str, language: str, symbols: List[str],
                       context_chunks: List[dict], hunks: List) -> List[Dict[str, Any]]:
    """
    Review a patch that fits within token limits.
    """
    # Create line-specific review prompt
    prompt = create_line_specific_review_prompt(patch, filename, language, symbols, context_chunks, hunks)

    def make_api_call():
        model = genai.GenerativeModel("gemini-2.5-pro")
        return model.generate_content(prompt)

    try:
        # Use rate limiter for API call
        response = execute_with_rate_limit(make_api_call, priority=1)

        if response and response.text:
            # Parse the AI response into individual line-specific comments
            ai_comments = format_ai_comments(response.text)

            # Map each comment to a position in the diff
            mapped_comments = []
            for comment_text in ai_comments:
                comment_pos = map_comment_to_position(comment_text, hunks, filename)
                if comment_pos:
                    comment_data = {
                        "path": comment_pos.path,
                        "body": comment_pos.body,
                        "position": comment_pos.position
                    }
                    # Only include line if it's a valid number
                    if comment_pos.line is not None:
                        comment_data["line"] = comment_pos.line
                    mapped_comments.append(comment_data)

            return mapped_comments
    except Exception as e:
        print(f"Error generating review for {filename}: {e}")
        return []

    return []


def review_large_patch_in_chunks(patch: str, filename: str, language: str, symbols: List[str],
                               context_chunks: List[dict], hunks: List) -> List[Dict[str, Any]]:
    """
    Review a large patch by processing it in chunks.
    """
    # For large patches, we'll process hunk by hunk or group hunks
    all_comments = []

    # Group hunks to stay within token limits
    hunk_groups = group_hunks_by_size(hunks, MAX_TOKENS_PER_REQUEST // 2)  # Leave room for context

    for i, hunk_group in enumerate(hunk_groups):
        # Create a mini-patch for this group
        group_patch = create_patch_from_hunks(hunk_group)

        # Review this chunk
        prompt = create_line_specific_review_prompt(
            group_patch, filename, language, symbols, context_chunks, hunk_group
        )

        def make_api_call():
            model = genai.GenerativeModel("gemini-2.5-pro")
            return model.generate_content(prompt)

        try:
            # Use rate limiter for API call
            response = execute_with_rate_limit(make_api_call, priority=2)  # Lower priority for chunks
            if response and response.text:
                ai_comments = format_ai_comments(response.text)

                # Map comments to positions in the original diff
                for comment_text in ai_comments:
                    comment_pos = map_comment_to_position(comment_text, hunks, filename)
                    if comment_pos:
                        comment_data = {
                            "path": comment_pos.path,
                            "body": comment_pos.body,
                            "position": comment_pos.position
                        }
                        # Only include line if it's a valid number
                        if comment_pos.line is not None:
                            comment_data["line"] = comment_pos.line
                        all_comments.append(comment_data)
        except Exception as e:
            print(f"Error reviewing chunk {i+1} for {filename}: {e}")
            continue

        # Add delay between chunks (rate limiter will handle most of this)
        time.sleep(5)  # Increased delay to respect free tier limits

    return all_comments


def create_line_specific_review_prompt(patch: str, filename: str, language: str,
                                     symbols: List[str], context_chunks: List[dict],
                                     hunks: List) -> str:
    """
    Create a prompt that asks the AI to provide line-specific review comments.
    """
    # Create a simplified view of the hunks for line reference
    hunk_info = []
    line_num = hunks[0].new_start if hunks else 1

    for hunk in hunks:
        hunk_lines = []
        for i, line in enumerate(hunk.lines):
            if line.startswith('+') and not line.startswith('++'):
                hunk_lines.append(f"Line {line_num}: {line}")
                line_num += 1
            elif line.startswith('-'):
                continue  # Skip deleted lines for line numbering
            else:
                line_num += 1
        if hunk_lines:
            hunk_info.extend(hunk_lines)

    return f"""
You are an expert software reviewer providing line-specific feedback.

File: {filename}
Language: {language}
Symbols changed: {', '.join(symbols) if symbols else 'None'}

Relevant context from repository:
{json.dumps(context_chunks, indent=2)}

Diff to review:
```
{patch}
```

Line reference guide for added lines:
{chr(10).join(hunk_info[:20])}  # Limit to first 20 lines for brevity

Instructions:
1. Review the code changes carefully
2. For each issue you identify, specify the exact line number using "Line X:" format
3. Focus on: bugs, security issues, performance problems, code quality, and best practices
4. Provide actionable, specific feedback
5. Separate different issues with blank lines

Example format:
"Line 15: Consider adding input validation here to prevent potential security issues.

Line 23: This function could benefit from error handling for the case where the API call fails.

Line 42: The variable name 'data' is too generic - consider a more descriptive name like 'user_profile'."

Please provide your line-specific review:
"""


def generate_review_summary(comments: List[Dict[str, Any]], filename: str, language: str) -> str:
    """
    Generate a summary of the review comments.
    """
    if not comments:
        return f"No issues found in `{filename}`."

    # Count different types of issues
    issue_types = {
        "security": 0,
        "performance": 0,
        "bugs": 0,
        "style": 0,
        "best_practices": 0
    }

    for comment in comments:
        body = comment["body"].lower()
        if any(keyword in body for keyword in ["security", "vulnerability", "injection", "xss"]):
            issue_types["security"] += 1
        elif any(keyword in body for keyword in ["performance", "slow", "optimize", "efficient"]):
            issue_types["performance"] += 1
        elif any(keyword in body for keyword in ["bug", "error", "fail", "incorrect", "wrong"]):
            issue_types["bugs"] += 1
        elif any(keyword in body for keyword in ["style", "format", "naming", "convention"]):
            issue_types["style"] += 1
        else:
            issue_types["best_practices"] += 1

    # Create summary
    total_issues = sum(issue_types.values())
    summary_parts = [f"Found {total_issues} issue{'s' if total_issues != 1 else ''} in `{filename}`:"]

    for issue_type, count in issue_types.items():
        if count > 0:
            summary_parts.append(f"  • {issue_type.replace('_', ' ').title()}: {count}")

    return "\n".join(summary_parts)


def group_hunks_by_size(hunks: List, max_tokens: int) -> List[List]:
    """
    Group hunks together to stay within token limits.
    """
    groups = []
    current_group = []
    current_tokens = 0

    for hunk in hunks:
        hunk_tokens = estimate_tokens(hunk.content)

        if current_tokens + hunk_tokens > max_tokens and current_group:
            groups.append(current_group)
            current_group = []
            current_tokens = 0

        current_group.append(hunk)
        current_tokens += hunk_tokens

    if current_group:
        groups.append(current_group)

    return groups


def create_patch_from_hunks(hunks: List) -> str:
    """
    Create a patch string from a list of hunks.
    """
    patch_lines = []
    for hunk in hunks:
        patch_lines.append(hunk.diff_header)
        patch_lines.extend(hunk.lines)

    return "\n".join(patch_lines)


# Legacy function for backward compatibility
def review_patch(patch: str, filename: str, repo_full_name: str, ref: str, commit_sha: str,
                index: faiss.Index, metadata: List[dict]) -> Dict[str, Any]:
    """
    Legacy function that maintains backward compatibility.
    Uses the new line-level review system but formats output similarly to the old system.
    """
    result = review_patch_line_level(patch, filename, repo_full_name, ref, commit_sha, index, metadata)

    # Convert to old format for backward compatibility
    comments_text = "\n\n".join([f"Line {comment.get('line', '?')}: {comment['body']}"
                                for comment in result["comments"]])

    return {
        "file": result["file"],
        "language": result["language"],
        "symbols": result["symbols"],
        "review": comments_text if comments_text else "No issues found.",
        "context_used": result["context_used"],
        "trimmed_patch": enhance_trim_diff(patch),
        "line_comments": result["comments"]  # New field for line-specific comments
    }