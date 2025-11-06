# backend/suggestions.py
"""
Code suggestion system that generates actionable code improvements
using GitHub's suggestion API for one-click fixes.
"""

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import google.generativeai as genai
from typing import List, Dict, Any, Optional, Tuple
from backend.utils import detect_language_from_filename
from backend.config import GEMINI_API_KEY
from backend.api_rate_limiter import execute_with_rate_limit

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)


class CodeSuggestionEngine:
    """Engine for generating code suggestions and fixes."""

    def __init__(self):
        self.model = genai.GenerativeModel("gemini-2.5-pro")

    def generate_suggestions(self, patch: str, filename: str, context: List[Dict] = None) -> Dict[str, Any]:
        """
        Generate code suggestions for a given patch.

        Args:
            patch: The diff patch to analyze
            filename: The file being analyzed
            context: Additional context from semantic search

        Returns:
            Dictionary containing suggestions and GitHub-formatted suggestions
        """
        prompt = self.create_suggestion_prompt(patch, filename, context)

        def make_api_call():
            return self.model.generate_content(prompt)

        try:
            # Use rate limiter for API call
            response = execute_with_rate_limit(make_api_call, priority=2)
            if response and response.text:
                return self.parse_suggestion_response(response.text, filename, patch)
        except Exception as e:
            print(f"Error generating suggestions: {e}")

        return {
            "filename": filename,
            "suggestions": [],
            "github_suggestions": [],
            "summary": "No suggestions generated."
        }

    def create_suggestion_prompt(self, patch: str, filename: str, context: List[Dict] = None) -> str:
        """Create a prompt for generating code suggestions."""
        language = detect_language_from_filename(filename)

        context_str = ""
        if context:
            context_str = f"\n\nContext from related code:\n{json.dumps(context[:2], indent=2)}"

        return f"""
You are an expert code reviewer who provides actionable code improvements with specific fix suggestions.

File: {filename}
Language: {language}

Diff to review:
```
{patch}
```
{context_str}

Your task is to identify code issues and provide specific, one-click fix suggestions using GitHub's suggestion format.

For each issue you identify:
1. Specify the exact line number (using "Line X:" format)
2. Describe the issue briefly
3. Provide the exact code fix using GitHub suggestion format

GitHub Suggestion Format:
```suggestion
- (your improved code here)
+ (your fix here)
```

Focus on:
1. **Bug Fixes**: Logic errors, null pointer exceptions, off-by-one errors
2. **Performance**: Inefficient algorithms, memory leaks, redundant operations
3. **Security**: Input validation, proper error handling, secure coding practices
4. **Code Quality**: Better variable names, clearer logic, improved readability
5. **Best Practices**: Language-specific idioms, proper error handling, design patterns

Guidelines:
- Keep suggestions minimal and focused
- Only suggest changes that clearly improve the code
- Ensure your suggestions compile and are syntactically correct
- Provide brief explanations for why the change is needed

Format your response as:
Line X: ISSUE_TYPE - Brief description
Explanation: [1-2 sentence explanation]
Suggestion:
```suggestion
- original_code_line
+ fixed_code_line
```

Only provide actual improvements. If no suggestions are needed, respond with "No code suggestions needed."
"""

    def parse_suggestion_response(self, response: str, filename: str, original_patch: str) -> Dict[str, Any]:
        """Parse the AI suggestion response into structured data."""
        lines = response.strip().split('\n')
        suggestions = []
        github_suggestions = []
        current_suggestion = {}

        for line in lines:
            line = line.strip()

            # Check for new suggestion (starts with "Line X:")
            if line.startswith("Line ") and ":" in line:
                if current_suggestion:
                    # Complete the previous suggestion
                    if "github_suggestion" in current_suggestion:
                        github_suggestions.append({
                            "path": filename,
                            "position": current_suggestion.get("line"),
                            "body": current_suggestion["explanation"],
                            "suggestion": current_suggestion["github_suggestion"]
                        })
                    suggestions.append(current_suggestion)

                # Parse the new suggestion
                line_num = line.split(":")[0].replace("Line ", "")
                issue_desc = line.split(" - ", 1)[-1] if " - " in line else line

                current_suggestion = {
                    "line": line_num,
                    "type": "suggestion",
                    "description": issue_desc,
                    "explanation": "",
                    "github_suggestion": ""
                }

            elif line.startswith("Explanation:"):
                current_suggestion["explanation"] = line.replace("Explanation:", "").strip()
            elif line.startswith("Suggestion:"):
                # Start collecting the suggestion code
                continue
            elif line.startswith("```suggestion"):
                # Start of suggestion block
                continue
            elif line.startswith("```"):
                # End of suggestion block
                continue
            elif current_suggestion and not any(line.startswith(prefix) for prefix in ["Line ", "Explanation:", "Suggestion:", "```"]):
                # This is part of the suggestion code
                if current_suggestion["github_suggestion"]:
                    current_suggestion["github_suggestion"] += "\n" + line
                else:
                    current_suggestion["github_suggestion"] = line

        # Add the last suggestion
        if current_suggestion and "github_suggestion" in current_suggestion:
            github_suggestions.append({
                "path": filename,
                "position": current_suggestion.get("line"),
                "body": current_suggestion["explanation"],
                "suggestion": current_suggestion["github_suggestion"]
            })
            suggestions.append(current_suggestion)

        return {
            "filename": filename,
            "suggestions": suggestions,
            "github_suggestions": github_suggestions,
            "summary": f"Generated {len(suggestions)} code suggestion{'s' if len(suggestions) != 1 else ''} for {filename}"
        }


def format_suggestion_for_github(suggestion: Dict[str, Any], filename: str) -> Dict[str, Any]:
    """
    Format a suggestion for GitHub's API.

    Args:
        suggestion: The parsed suggestion data
        filename: The file path

    Returns:
        GitHub API formatted suggestion
    """
    return {
        "path": filename,
        "position": suggestion.get("line"),
        "body": suggestion["explanation"],
        "suggestion": suggestion["github_suggestion"]
    }


def generate_quick_fixes(patch: str, filename: str, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate quick fixes for identified issues.

    Args:
        patch: The diff patch
        filename: The file path
        issues: List of identified issues

    Returns:
        List of quick fix suggestions
    """
    quick_fixes = []

    # Common fix patterns
    fix_patterns = {
        # Python fixes
        "missing_error_handling": {
            "pattern": r"(except\s*:)",
            "fix": "except Exception as e:\n    # Handle error appropriately"
        },
        "f_string_format": {
            "pattern": r'f".*{.*}.*"',
            "fix": "Use f-string formatting consistently"
        },
        "list_comprehension": {
            "pattern": r"\[\s*for\s+.*\s+in\s+.*:\s*\]",
            "fix": "Consider using list comprehension"
        }
    }

    # JavaScript/TypeScript fixes
    if filename.endswith(('.js', '.ts')):
        fix_patterns.update({
            "var_to_const": {
                "pattern": r"\bvar\s+\w+\s*=",
                "fix": "Use 'const' or 'let' instead of 'var'"
            },
            "missing_semicolon": {
                "pattern": r"[^;]\s*$",
                "fix": "Add semicolon at end of statement"
            }
        })

    # Apply fix patterns to the patch
    for issue_type, pattern_info in fix_patterns.items():
        import re
        if re.search(pattern_info["pattern"], patch):
            quick_fixes.append({
                "type": "quick_fix",
                "issue_type": issue_type,
                "description": pattern_info["fix"],
                "suggestion": pattern_info["fix"]
            })

    return quick_fixes


def create_suggestion_comment(suggestion: Dict[str, Any]) -> str:
    """
    Create a user-friendly comment for a suggestion.

    Args:
        suggestion: The suggestion data

    Returns:
        Formatted comment string
    """
    comment = f"💡 **Code Suggestion**\n\n"

    if suggestion.get("type"):
        comment += f"**Type:** {suggestion['type'].replace('_', ' ').title()}\n\n"

    if suggestion.get("description"):
        comment += f"**Issue:** {suggestion['description']}\n\n"

    if suggestion.get("explanation"):
        comment += f"**Why:** {suggestion['explanation']}\n\n"

    comment += "**Proposed Fix:**\n"
    comment += "```suggestion\n"
    comment += suggestion.get("github_suggestion", "No specific code change provided.")
    comment += "\n```\n\n"

    comment += "👆 Click the 'Apply suggestion' button above to apply this fix automatically."

    return comment


def generate_suggestions_for_file(patch: str, filename: str, context: List[Dict] = None) -> Dict[str, Any]:
    """
    Generate comprehensive suggestions for a single file.

    Args:
        patch: The diff patch
        filename: The file path
        context: Additional context

    Returns:
        Comprehensive suggestion analysis
    """
    engine = CodeSuggestionEngine()

    # Generate AI-powered suggestions
    ai_suggestions = engine.generate_suggestions(patch, filename, context)

    # Generate quick fixes for common patterns
    quick_fixes = generate_quick_fixes(patch, filename, [])

    # Combine all suggestions
    all_suggestions = ai_suggestions.get("suggestions", [])
    all_suggestions.extend([{"description": fix["description"], "type": "quick_fix"} for fix in quick_fixes])

    # Create GitHub-formatted suggestions
    github_suggestions = []
    for suggestion in all_suggestions:
        if "github_suggestion" in suggestion and suggestion["github_suggestion"]:
            github_suggestions.append(format_suggestion_for_github(suggestion, filename))

    return {
        "filename": filename,
        "total_suggestions": len(all_suggestions),
        "suggestions": all_suggestions,
        "github_suggestions": github_suggestions,
        "quick_fixes": quick_fixes,
        "summary": f"Generated {len(all_suggestions)} suggestions ({len(quick_fixes)} quick fixes) for {filename}"
    }


def apply_suggestions_to_pr(pr, suggestions: List[Dict[str, Any]]) -> bool:
    """
    Apply suggestions to a pull request using GitHub's API.

    Args:
        pr: GitHub PullRequest object
        suggestions: List of GitHub-formatted suggestions

    Returns:
        True if successful, False otherwise
    """
    if not suggestions:
        return True

    try:
        # Group suggestions by file
        file_suggestions = {}
        for suggestion in suggestions:
            path = suggestion.get("path", "")
            if path not in file_suggestions:
                file_suggestions[path] = []
            file_suggestions[path].append(suggestion)

        # Create review with suggestions for each file
        all_review_comments = []

        for file_path, file_sugs in file_suggestions.items():
            for suggestion in file_sugs:
                # Format suggestion as review comment
                comment_body = create_suggestion_comment(suggestion)

                all_review_comments.append({
                    "path": suggestion["path"],
                    "body": comment_body,
                    "line": suggestion.get("position")
                })

        if all_review_comments:
            pr.create_review(
                body=f"🤖 **AI Code Suggestions**\n\nFound {len(all_review_comments)} improvement{'s' if len(all_review_comments) != 1 else ''}. Review the suggestions below and apply them with one click!",
                event="COMMENT",
                comments=all_review_comments,
            )
            print(f"✅ Applied {len(all_review_comments)} code suggestions to PR")
            return True

    except Exception as e:
        print(f"⚠️ Error applying suggestions: {e}")
        return False

    return True


def get_suggestion_types() -> Dict[str, str]:
    """
    Get information about available suggestion types.

    Returns:
        Dictionary mapping suggestion types to descriptions
    """
    return {
        "bug_fix": "Fixes for bugs and logic errors",
        "performance": "Performance optimizations and improvements",
        "security": "Security vulnerability fixes",
        "code_quality": "Code quality and readability improvements",
        "best_practices": "Best practices and idiomatic code improvements",
        "quick_fix": "Common pattern fixes and improvements"
    }