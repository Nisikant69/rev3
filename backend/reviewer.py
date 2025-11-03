# backend/reviewer.py
import json
import google.generativeai as genai
from backend.utils import (
    trim_diff, extract_symbols_from_patch, detect_language_from_filename,
    parse_diff_hunks, map_comment_to_position, format_ai_comments,
    enhance_trim_diff, estimate_tokens, should_chunk_file, chunk_code_by_functions
)
from backend.semantic_search import semantic_search
from backend.config import GEMINI_API_KEY, TOP_K, MAX_TOKENS_PER_REQUEST
import faiss
from typing import List, Dict, Any
import time
from google.api_core.exceptions import ResourceExhausted

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# A rough estimate for a safe token limit for a single API call.
# This might need to be adjusted based on the specific model and its limits.
SAFE_TOKEN_LIMIT = 5000

def review_patch(patch: str, filename: str, repo_full_name: str, ref: str, commit_sha: str, index: faiss.Index, metadata: List[dict]):
    """
    Generate a structured AI review for a given patch using an iterative, chunking approach.
    """
    language = detect_language_from_filename(str(filename))
    symbols = extract_symbols_from_patch(patch)
    trimmed = trim_diff(patch)

    # Get semantic context from repo index
    context_chunks = semantic_search(trimmed, index, metadata)
    context_chunks = context_chunks[:TOP_K]

    # Combine context and general instructions into a base prompt
    base_prompt = f"""
You are an expert software reviewer.
A diff has been provided for a file in {language}.

File: {filename}
Symbols changed: {symbols}

Relevant context from repo:
{json.dumps(context_chunks, indent=2)}

Please provide a detailed review with:
1. Identification of potential bugs or inefficiencies.
2. Suggestions for code quality, readability, and performance.
3. Security or best practice considerations if relevant.
Format your answer as plain text.
"""

    model = genai.GenerativeModel("gemini-2.5-pro")
    full_review_text = ""
    patch_lines = trimmed.splitlines()
    chunk_size = 50  # Number of lines per chunk

    for i in range(0, len(patch_lines), chunk_size):
        patch_chunk = "\n".join(patch_lines[i:i + chunk_size])
        
        # Combine the base prompt with the current chunk of the patch
        prompt_with_chunk = f"{base_prompt}\n\nHere is the patch chunk:\n{patch_chunk}"
        
        retries = 3
        delay = 10
        for attempt in range(retries):
            try:
                response = model.generate_content(prompt_with_chunk)
                if response and response.text:
                    full_review_text += f"\n\n--- Review for Patch Chunk {i//chunk_size + 1} ---\n\n"
                    full_review_text += response.text
                break
            except ResourceExhausted as e:
                print(f"⚠️ Quota exceeded. Attempt {attempt + 1}/{retries}. Retrying in {delay} seconds...")
                if attempt < retries - 1:
                    time.sleep(delay)
                    delay *= 2
                else:
                    print("❌ Max retries reached for this chunk. Skipping.")
                    full_review_text += "\n\n--- Review Failed for this chunk due to API rate limits ---\n\n"
                    break
            except Exception as e:
                print(f"An unexpected error occurred for this chunk: {e}")
                full_review_text += "\n\n--- Review Failed for this chunk due to an unexpected error ---\n\n"
                break
        
        time.sleep(5)  # Add a small delay between chunks to avoid rate limit spikes

    return {
        "file": filename,
        "language": language,
        "symbols": symbols,
        "review": full_review_text if full_review_text else "AI review failed for all chunks.",
        "context_used": context_chunks,
        "trimmed_patch": trimmed,
    }