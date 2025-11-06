# backend/summarizer.py
"""
Intelligent PR summarization system that generates meaningful descriptions
and analyses of pull request changes.
"""

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import google.generativeai as genai
from typing import List, Dict, Any, Optional
from backend.utils import detect_language_from_filename, extract_symbols_from_patch, estimate_tokens
from backend.config import GEMINI_API_KEY, ENABLE_SUMMARIZATION, MAX_TOKENS_PER_REQUEST
from backend.api_rate_limiter import execute_with_rate_limit

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)


def generate_pr_summary(files: List[Any], repo_name: str, pr_title: str, pr_description: str) -> Dict[str, Any]:
    """
    Generate an intelligent summary of the PR changes.

    Args:
        files: List of GitHub PR file objects
        repo_name: Repository name
        pr_title: PR title
        pr_description: PR description

    Returns:
        Dictionary containing summary components
    """
    if not ENABLE_SUMMARIZATION:
        return create_basic_summary(files, pr_title, pr_description)

    # Analyze changed files
    file_analysis = analyze_changed_files(files)

    # Generate AI-powered summary
    ai_summary = generate_ai_summary(file_analysis, repo_name, pr_title, pr_description)

    return {
        "overview": ai_summary.get("overview", "Code changes in this PR"),
        "changes": ai_summary.get("changes", []),
        "impact": ai_summary.get("impact", "Various improvements and fixes"),
        "testing": ai_summary.get("testing", "Review changes and test functionality"),
        "files_affected": len([f for f in files if f.status != "removed"]),
        "languages": list(set([detect_language_from_filename(f.filename) for f in files if f.patch])),
        "file_types": categorize_file_types(files),
        "complexity": assess_complexity(files)
    }


def analyze_changed_files(files: List[Any]) -> List[Dict[str, Any]]:
    """
    Analyze each changed file to extract meaningful information.
    """
    analyzed_files = []

    for file in files:
        if file.status == "removed" or not file.patch:
            continue

        language = detect_language_from_filename(file.filename)
        symbols = extract_symbols_from_patch(file.patch)
        change_type = categorize_change_type(file.patch, file.status)
        complexity = estimate_patch_complexity(file.patch)

        analyzed_files.append({
            "filename": file.filename,
            "status": file.status,
            "language": language,
            "symbols": symbols,
            "change_type": change_type,
            "complexity": complexity,
            "additions": file.additions if hasattr(file, 'additions') else 0,
            "deletions": file.deletions if hasattr(file, 'deletions') else 0,
            "patch_size": len(file.patch)
        })

    return analyzed_files


def generate_ai_summary(file_analysis: List[Dict[str, Any]], repo_name: str,
                       pr_title: str, pr_description: str) -> Dict[str, Any]:
    """
    Use AI to generate a comprehensive PR summary.
    """
    if not file_analysis:
        return {"overview": "No significant changes detected"}

    # Create summary prompt
    prompt = create_summary_prompt(file_analysis, repo_name, pr_title, pr_description)

    model = genai.GenerativeModel("gemini-2.5-pro")

    try:
        response = model.generate_content(prompt)
        if response and response.text:
            # Parse the AI response
            return parse_ai_summary_response(response.text)
    except Exception as e:
        print(f"Error generating AI summary: {e}")

    # Fallback to basic summary
    return create_basic_ai_summary(file_analysis)


def create_summary_prompt(file_analysis: List[Dict[str, Any]], repo_name: str,
                         pr_title: str, pr_description: str) -> str:
    """
    Create a prompt for AI-based PR summarization.
    """
    # Create a concise view of changes
    changes_summary = []
    for file in file_analysis[:10]:  # Limit to first 10 files
        changes_summary.append(
            f"- {file['filename']} ({file['status']}, {file['language']}): "
            f"{file['change_type']}, {file['complexity']} complexity"
        )

    return f"""
You are an expert software engineer analyzing a pull request.

Repository: {repo_name}
PR Title: {pr_title}
PR Description: {pr_description if pr_description else "No description provided"}

Files changed:
{chr(10).join(changes_summary)}

Total files: {len(file_analysis)}
Languages involved: {list(set([f['language'] for f in file_analysis]))}

Please provide a comprehensive analysis of this PR in the following JSON format:

{{
    "overview": "A concise 1-2 sentence summary of what this PR accomplishes",
    "changes": [
        "List of main changes or features being implemented",
        "Focus on the most important changes"
    ],
    "impact": "Description of how these changes affect the system/users",
    "testing": "Recommendations for testing these changes",
    "risk_level": "low/medium/high based on the complexity and scope of changes"
}}

Focus on:
1. The actual functionality being changed
2. The purpose and impact of these changes
3. Any potential risks or considerations
4. Testing recommendations

Provide your response in valid JSON format only.
"""


def parse_ai_summary_response(response_text: str) -> Dict[str, Any]:
    """
    Parse the AI summary response into structured data.
    """
    try:
        # Try to extract JSON from the response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1

        if start_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Fallback: parse text response manually
    lines = response_text.strip().split('\n')
    result = {
        "overview": "",
        "changes": [],
        "impact": "",
        "testing": "",
        "risk_level": "medium"
    }

    current_section = None
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if "overview" in line.lower():
            current_section = "overview"
        elif "changes" in line.lower():
            current_section = "changes"
        elif "impact" in line.lower():
            current_section = "impact"
        elif "testing" in line.lower():
            current_section = "testing"
        elif line.startswith('-') or line.startswith('*'):
            if current_section == "changes":
                result["changes"].append(line.lstrip('- *').strip())
        elif current_section and result[current_section]:
            if isinstance(result[current_section], list):
                result[current_section].append(line)
            else:
                result[current_section] += " " + line
        elif current_section:
            result[current_section] = line

    return result


def create_basic_ai_summary(file_analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create a basic summary without AI.
    """
    changes = []
    languages = set()
    complexity_scores = []

    for file in file_analysis:
        changes.append(f"{file['filename']}: {file['change_type']}")
        languages.add(file['language'])
        complexity_scores.append(file['complexity'])

    avg_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 1

    return {
        "overview": f"Changes to {len(file_analysis)} files across {len(languages)} programming languages",
        "changes": changes[:5],  # Limit to first 5 changes
        "impact": "Various code improvements and modifications",
        "testing": "Review changes and test affected functionality",
        "risk_level": "high" if avg_complexity > 7 else "medium" if avg_complexity > 4 else "low"
    }


def create_basic_summary(files: List[Any], pr_title: str, pr_description: str) -> Dict[str, Any]:
    """
    Create a basic summary without AI analysis.
    """
    changed_files = [f for f in files if f.status != "removed"]
    languages = set([detect_language_from_filename(f.filename) for f in changed_files if f.patch])

    return {
        "overview": pr_title if pr_title else "Code changes in pull request",
        "changes": [f.filename for f in changed_files[:5]],
        "impact": pr_description if pr_description else "Various code modifications",
        "testing": "Review and test the changes",
        "files_affected": len(changed_files),
        "languages": list(languages),
        "file_types": categorize_file_types(files),
        "complexity": "medium"
    }


def categorize_change_type(patch: str, status: str) -> str:
    """
    Categorize the type of change based on patch content.
    """
    if status == "added":
        return "New file"

    patch_lower = patch.lower()

    # Look for patterns indicating different types of changes
    if any(keyword in patch_lower for keyword in ["test", "spec"]):
        return "Test updates"
    elif any(keyword in patch_lower for keyword in ["import", "require", "include"]):
        return "Dependency changes"
    elif any(keyword in patch_lower for keyword in ["def ", "function", "class "]):
        return "Function/class changes"
    elif any(keyword in patch_lower for keyword in ["fix", "bug", "error"]):
        return "Bug fixes"
    elif any(keyword in patch_lower for keyword in ["refactor", "cleanup", "improve"]):
        return "Refactoring"
    elif any(keyword in patch_lower for keyword in ["config", "setting"]):
        return "Configuration changes"
    else:
        return "Code modifications"


def estimate_patch_complexity(patch: str) -> int:
    """
    Estimate the complexity of a patch on a scale of 1-10.
    """
    # Base complexity on various factors
    factors = {
        "line_count": min(patch.count('\n') / 10, 3),  # Max 3 points for line count
        "additions": min(patch.count('+') / 5, 2),      # Max 2 points for additions
        "deletions": min(patch.count('-') / 10, 1),     # Max 1 point for deletions
        "functions": min(patch.count('def ') + patch.count('function '), 2),  # Max 2 points
        "classes": min(patch.count('class '), 1),       # Max 1 point
        "loops": min(patch.lower().count('for ') + patch.lower().count('while '), 1),  # Max 1 point
    }

    complexity = sum(factors.values())
    return min(int(complexity), 10)  # Cap at 10


def categorize_file_types(files: List[Any]) -> Dict[str, int]:
    """
    Categorize files by type.
    """
    categories = {
        "code": 0,
        "tests": 0,
        "config": 0,
        "docs": 0,
        "other": 0
    }

    for file in files:
        filename = file.filename.lower()

        if any(test_indicator in filename for test_indicator in ["test", "spec", "mock"]):
            categories["tests"] += 1
        elif any(code_indicator in filename for code_indicator in [".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs"]):
            categories["code"] += 1
        elif any(config_indicator in filename for config_indicator in [".json", ".yaml", ".yml", ".toml", ".ini", ".env", "dockerfile"]):
            categories["config"] += 1
        elif any(doc_indicator in filename for doc_indicator in [".md", ".txt", ".rst", ".doc", ".pdf"]):
            categories["docs"] += 1
        else:
            categories["other"] += 1

    return categories


def assess_complexity(files: List[Any]) -> str:
    """
    Assess overall PR complexity.
    """
    total_additions = sum(f.additions for f in files if hasattr(f, 'additions'))
    total_deletions = sum(f.deletions for f in files if hasattr(f, 'deletions'))
    files_changed = len([f for f in files if f.status != "removed"])

    # Simple complexity assessment
    if total_additions > 500 or files_changed > 20:
        return "high"
    elif total_additions > 100 or files_changed > 10:
        return "medium"
    else:
        return "low"


def format_summary_for_comment(summary: Dict[str, Any]) -> str:
    """
    Format the summary for posting as a PR comment.
    """
    sections = []

    sections.append(f"## 📋 PR Summary")
    sections.append(f"**{summary['overview']}**")

    if summary.get("changes"):
        sections.append("\n### 🔧 Main Changes")
        for change in summary["changes"][:5]:  # Limit to 5 changes
            sections.append(f"- {change}")

    sections.append(f"\n### 📊 Impact")
    sections.append(summary.get("impact", "Code improvements and modifications"))

    if summary.get("files_affected"):
        sections.append(f"\n### 📁 Files Affected")
        sections.append(f"- **Files changed**: {summary['files_affected']}")
        if summary.get("languages"):
            sections.append(f"- **Languages**: {', '.join(summary['languages'])}")
        if summary.get("file_types"):
            types = [f"{k} ({v})" for k, v in summary['file_types'].items() if v > 0]
            if types:
                sections.append(f"- **Types**: {', '.join(types)}")

    if summary.get("testing"):
        sections.append(f"\n### 🧪 Testing Recommendations")
        sections.append(summary["testing"])

    if summary.get("risk_level"):
        risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}
        sections.append(f"\n### ⚠️ Risk Assessment")
        sections.append(f"**Risk Level**: {risk_emoji.get(summary['risk_level'], '⚪')} {summary['risk_level'].title()}")

    return "\n".join(sections)