# backend/changelog_generator.py
"""
Generate formatted changelog entries for pull requests following
conventional commit standards and categorized changes.
"""

import re
from typing import List, Dict, Any, Optional
from backend.utils import detect_language_from_filename, extract_symbols_from_patch


def generate_changelog(files: List[Any], pr_title: str, pr_description: str) -> Dict[str, Any]:
    """
    Generate a structured changelog from PR changes.

    Args:
        files: List of GitHub PR file objects
        pr_title: PR title
        pr_description: PR description

    Returns:
        Dictionary with categorized changelog entries
    """
    # Analyze changes
    changes = analyze_changelog_changes(files, pr_title, pr_description)

    # Categorize changes
    categorized = categorize_changelog_entries(changes)

    # Generate markdown
    markdown = format_changelog_markdown(categorized)

    return {
        "changes": categorized,
        "markdown": markdown,
        "summary": create_changelog_summary(categorized),
        "version": "next"  # Placeholder for version
    }


def analyze_changelog_changes(files: List[Any], pr_title: str, pr_description: str) -> List[Dict[str, Any]]:
    """
    Analyze files and PR content to extract changelog-relevant changes.
    """
    changes = []

    # Extract information from PR title and description
    title_info = parse_pr_title(pr_title)
    description_info = parse_pr_description(pr_description)

    # Combine with file analysis
    for file in files:
        if file.status == "removed" or not file.patch:
            continue

        file_changes = analyze_file_for_changelog(file, title_info, description_info)
        changes.extend(file_changes)

    # Add PR-level changes if no file-specific changes found
    if not changes and title_info:
        changes.append({
            "type": title_info.get("type", "changed"),
            "description": title_info.get("description", pr_title),
            "scope": "general",
            "breaking": title_info.get("breaking", False),
            "files": [f.filename for f in files if f.status != "removed"]
        })

    return changes


def parse_pr_title(pr_title: str) -> Dict[str, Any]:
    """
    Parse PR title to extract conventional commit information.
    """
    # Pattern for conventional commits: type(scope): description
    pattern = r'^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(?:\(([^)]+)\))?:\s*(.+?)(?:\s*!\s*)?$'

    match = re.match(pattern, pr_title, re.IGNORECASE)
    if match:
        commit_type = match.group(1).lower()
        scope = match.group(2)
        description = match.group(3)
        breaking = "!" in pr_title or "BREAKING CHANGE" in pr_title.upper()

        # Map conventional types to changelog categories
        type_mapping = {
            "feat": "added",
            "fix": "fixed",
            "docs": "changed",
            "style": "changed",
            "refactor": "changed",
            "perf": "changed",
            "test": "changed",
            "build": "changed",
            "ci": "changed",
            "chore": "changed",
            "revert": "fixed"
        }

        return {
            "type": type_mapping.get(commit_type, "changed"),
            "scope": scope,
            "description": description,
            "breaking": breaking,
            "raw_type": commit_type
        }

    # Fallback: try to infer type from keywords
    title_lower = pr_title.lower()
    if any(keyword in title_lower for keyword in ["add", "new", "implement", "create"]):
        inferred_type = "added"
    elif any(keyword in title_lower for keyword in ["fix", "bug", "error", "issue"]):
        inferred_type = "fixed"
    elif any(keyword in title_lower for keyword in ["update", "improve", "enhance", "refactor"]):
        inferred_type = "changed"
    elif any(keyword in title_lower for keyword in ["remove", "delete", "deprecate"]):
        inferred_type = "removed"
    else:
        inferred_type = "changed"

    return {
        "type": inferred_type,
        "scope": None,
        "description": pr_title,
        "breaking": "breaking" in title_lower or "deprecate" in title_lower,
        "raw_type": None
    }


def parse_pr_description(pr_description: str) -> Dict[str, Any]:
    """
    Parse PR description for additional changelog information.
    """
    if not pr_description:
        return {}

    info = {}

    # Look for breaking changes
    breaking_pattern = r'BREAKING CHANGE:\s*(.+?)(?=\n\n|\n[A-Z]|\Z)'
    breaking_match = re.search(breaking_pattern, pr_description, re.DOTALL | re.IGNORECASE)
    if breaking_match:
        info["breaking_description"] = breaking_match.group(1).strip()
        info["breaking"] = True

    # Look for issue references
    issue_pattern = r'(?:fixes|closes|resolves)\s+#(\d+)'
    issues = re.findall(issue_pattern, pr_description, re.IGNORECASE)
    if issues:
        info["issues"] = issues

    # Look for migration notes
    migration_keywords = ["migration", "upgrade", "backward compatible", "breaking"]
    if any(keyword in pr_description.lower() for keyword in migration_keywords):
        info["migration_required"] = True

    return info


def analyze_file_for_changelog(file: Any, title_info: Dict[str, Any], description_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Analyze individual file changes for changelog relevance.
    """
    changes = []
    filename = file.filename
    patch = file.patch
    status = file.status

    # Determine change category based on file type and content
    category = categorize_file_for_changelog(filename, patch, status)

    # Extract meaningful description
    description = extract_change_description(filename, patch, status, title_info)

    if description:
        changes.append({
            "type": category,
            "description": description,
            "scope": extract_scope_from_filename(filename),
            "breaking": title_info.get("breaking", False) or is_breaking_change(patch),
            "files": [filename],
            "major_change": is_major_change(patch, filename)
        })

    return changes


def categorize_file_for_changelog(filename: str, patch: str, status: str) -> str:
    """
    Categorize a file change for changelog purposes.
    """
    filename_lower = filename.lower()

    # Documentation
    if any(doc_ext in filename_lower for doc_ext in [".md", ".rst", ".txt", ".doc"]):
        return "changed"

    # Configuration files
    if any(config_ext in filename_lower for config_ext in [".json", ".yaml", ".yml", ".toml", ".ini", ".env"]):
        return "changed"

    # Test files
    if any(test_indicator in filename_lower for test_indicator in ["test", "spec", "mock"]):
        return "changed"

    # New files
    if status == "added":
        return "added"

    # Analyze patch content for type
    patch_lower = patch.lower()
    if any(keyword in patch_lower for keyword in ["fix", "bug", "error", "exception"]):
        return "fixed"
    elif any(keyword in patch_lower for keyword in ["deprecated", "removed", "delete"]):
        return "removed"
    elif any(keyword in patch_lower for keyword in ["security", "vulnerability", "auth"]):
        return "security"
    elif any(keyword in patch_lower for keyword in ["performance", "optimize", "speed", "memory"]):
        return "changed"
    else:
        return "changed"


def extract_scope_from_filename(filename: str) -> str:
    """
    Extract a meaningful scope from the filename.
    """
    # Remove common prefixes and extensions
    name = filename.lower()
    name = name.replace("src/", "").replace("lib/", "").replace("app/", "")
    name = name.replace(".py", "").replace(".js", "").replace(".ts", "")
    name = name.replace(".java", "").replace(".cpp", "").replace(".c", "")

    # Extract first meaningful directory or file name
    parts = name.split('/')
    if len(parts) > 1:
        return parts[0]
    elif parts:
        return parts[0]

    return "general"


def extract_change_description(filename: str, patch: str, status: str, title_info: Dict[str, Any]) -> Optional[str]:
    """
    Extract a meaningful description of the change.
    """
    # If we have good info from PR title, use that
    if title_info.get("description") and title_info.get("description") != filename:
        return title_info["description"]

    # Try to extract information from patch
    if patch:
        # Look for function/class additions
        symbols = extract_symbols_from_patch(patch)
        if symbols:
            if status == "added":
                return f"Added {', '.join(symbols[:3])} to {filename}"
            else:
                return f"Updated {', '.join(symbols[:3])} in {filename}"

        # Look for TODO/FIXME comments that might indicate purpose
        todo_pattern = r'(?:TODO|FIXME|NOTE):\s*(.+)'
        todo_matches = re.findall(todo_pattern, patch)
        if todo_matches:
            return f"Addressed: {todo_matches[0]}"

    # Fallback to filename-based description
    if status == "added":
        return f"Added {filename}"
    elif status == "removed":
        return f"Removed {filename}"
    else:
        return f"Updated {filename}"


def is_breaking_change(patch: str) -> bool:
    """
    Determine if a patch contains breaking changes.
    """
    breaking_indicators = [
        r'def\s+\w+\s*\([^)]*\)\s*->[^:]+:',  # Python function signature changes
        r'interface\s+\w+',  # TypeScript interface changes
        r'type\s+\w+',  # TypeScript type changes
        r'BREAKING CHANGE',
        r'@deprecated',
        r'remove|delete|deprecat',
    ]

    patch_lower = patch.lower()
    for pattern in breaking_indicators:
        if re.search(pattern, patch_lower):
            return True

    return False


def is_major_change(patch: str, filename: str) -> bool:
    """
    Determine if this is a major structural change.
    """
    # Large patches are often major changes
    if len(patch) > 1000:
        return True

    # Changes to core files are often major
    core_indicators = ["main", "index", "app", "core", "config", "init"]
    if any(indicator in filename.lower() for indicator in core_indicators):
        return True

    # Database schema changes
    if any(db_indicator in patch.lower() for db_indicator in ["schema", "migration", "table", "column"]):
        return True

    return False


def categorize_changelog_entries(changes: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Organize changes into standard changelog categories.
    """
    categories = {
        "added": [],
        "fixed": [],
        "changed": [],
        "deprecated": [],
        "removed": [],
        "security": []
    }

    breaking_changes = []

    for change in changes:
        category = change.get("type", "changed")
        if category not in categories:
            category = "changed"

        # Handle breaking changes separately
        if change.get("breaking", False):
            breaking_changes.append(change)

        categories[category].append(change)

    # Add breaking changes section if any exist
    if breaking_changes:
        categories["breaking"] = breaking_changes

    return categories


def create_changelog_summary(categorized: Dict[str, List[Dict[str, Any]]]) -> str:
    """
    Create a one-line summary of the changelog.
    """
    parts = []
    for category, changes in categorized.items():
        if category == "breaking":
            continue  # Handle separately
        if changes:
            parts.append(f"{len(changes)} {category}")

    summary = ", ".join(parts)
    if categorized.get("breaking"):
        summary += f" (including {len(categorized['breaking'])} breaking change{'s' if len(categorized['breaking']) != 1 else ''})"

    return summary if summary else "Various changes and improvements"


def format_changelog_markdown(categorized: Dict[str, List[Dict[str, Any]]]) -> str:
    """
    Format the categorized changes into markdown changelog format.
    """
    sections = []

    # Define category order and titles
    category_order = [
        ("breaking", "### 🚨 BREAKING CHANGES"),
        ("added", "### ✅ Added"),
        ("fixed", "### 🐛 Fixed"),
        ("security", "### 🔒 Security"),
        ("changed", "### 🔄 Changed"),
        ("deprecated", "### ⚠️ Deprecated"),
        ("removed", "### ❌ Removed")
    ]

    for category, title in category_order:
        changes = categorized.get(category, [])
        if changes:
            sections.append(title)
            for change in changes:
                # Format the change description
                desc = change["description"]
                if change.get("scope") and change["scope"] != "general":
                    desc = f"**{change['scope']}**: {desc}"
                if change.get("breaking", False) and category != "breaking":
                    desc += " ⚠️ **BREAKING**"

                sections.append(f"- {desc}")

            sections.append("")  # Add blank line after each section

    # If no changes found
    if not sections:
        sections.append("### 🔄 Changed")
        sections.append("- Various improvements and updates")

    return "\n".join(sections).strip()


def format_conventional_commit_message(changes: List[Dict[str, Any]], pr_title: str) -> str:
    """
    Format changes as a conventional commit message.
    """
    if not changes:
        return pr_title

    # Use the first change as the primary one
    primary_change = changes[0]
    commit_type = primary_change.get("type", "changed")
    scope = primary_change.get("scope", "")
    description = primary_change.get("description", pr_title)

    # Map changelog types back to conventional commit types
    type_mapping = {
        "added": "feat",
        "fixed": "fix",
        "security": "fix",
        "changed": "refactor",
        "deprecated": "refactor",
        "removed": "refactor"
    }

    conventional_type = type_mapping.get(commit_type, "refactor")
    scope_part = f"({scope})" if scope else ""
    breaking_part = "!" if primary_change.get("breaking", False) else ""

    return f"{conventional_type}{scope_part}{breaking_part}: {description}"


def extract_version_bump_suggestion(categorized: Dict[str, List[Dict[str, Any]]]) -> str:
    """
    Suggest a version bump based on the types of changes.
    """
    if categorized.get("breaking") or categorized.get("removed"):
        return "major (x.y.0)"
    elif categorized.get("added") or categorized.get("fixed"):
        return "minor (x.y.z)"
    elif any(categorized.get(cat) for cat in ["changed", "deprecated", "security"]):
        return "patch (x.y.z)"
    else:
        return "patch (x.y.z)"