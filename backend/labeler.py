# backend/labeler.py
"""
Automatic PR labeling system that applies relevant labels based on
file patterns, content analysis, and change characteristics.
"""

import re
from typing import List, Dict, Set, Any, Optional
from backend.utils import detect_language_from_filename, extract_symbols_from_patch
from backend.config import ENABLE_AUTO_LABELING


def generate_pr_labels(files: List[Any], pr_title: str, pr_description: str,
                      existing_labels: List[str] = None) -> Dict[str, Any]:
    """
    Generate appropriate labels for a pull request.

    Args:
        files: List of GitHub PR file objects
        pr_title: PR title
        pr_description: PR description
        existing_labels: Labels already present on the PR

    Returns:
        Dictionary with labeling information
    """
    if not ENABLE_AUTO_LABELING:
        return {"labels": [], "reasons": {}, "existing": existing_labels or []}

    existing_labels = set(existing_labels or [])
    all_labels = set()

    # Analyze different aspects for labeling
    file_labels, file_reasons = analyze_file_based_labels(files)
    content_labels, content_reasons = analyze_content_based_labels(files, pr_title, pr_description)
    size_labels, size_reasons = analyze_size_based_labels(files)
    type_labels, type_reasons = analyze_type_based_labels(pr_title, pr_description, files)

    # Combine all labels
    all_labels.update(file_labels)
    all_labels.update(content_labels)
    all_labels.update(size_labels)
    all_labels.update(type_labels)

    # Remove labels that already exist
    new_labels = all_labels - existing_labels

    # Ensure all labels are strings (not lists)
    clean_labels = set()
    for label in new_labels:
        if isinstance(label, list):
            # If it's a list, take the first item or join with underscores
            clean_labels.add(str(label[0]) if label else "")
        elif isinstance(label, str):
            clean_labels.add(label)
        else:
            # Convert to string for safety
            clean_labels.add(str(label))

    # Combine all reasons
    all_reasons = {**file_reasons, **content_reasons, **size_reasons, **type_reasons}

    return {
        "labels": sorted(list(clean_labels)),
        "reasons": {label: reason for label, reason in all_reasons.items() if label in clean_labels},
        "existing": sorted(list(existing_labels)),
        "all": sorted(list(all_labels))
    }


def analyze_file_based_labels(files: List[Any]) -> tuple[Set[str], Dict[str, str]]:
    """
    Generate labels based on file patterns and types.
    """
    labels = set()
    reasons = {}

    # File extension mappings
    extension_mapping = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "react",
        ".tsx": "react",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
        ".cs": "csharp",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".html": "html",
        ".css": "css",
        ".scss": "scss",
        ".sass": "sass",
        ".less": "less",
        ".sql": "sql",
        ".sh": "shell",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".xml": "xml",
        ".dockerfile": "docker",
        "Dockerfile": "docker",
        ".md": "documentation",
        ".rst": "documentation",
        ".txt": "documentation",
        ".pdf": "documentation",
        ".png": "assets",
        ".jpg": "assets",
        ".jpeg": "assets",
        ".gif": "assets",
        ".svg": "assets",
        ".ico": "assets"
    }

    # Directory pattern mappings
    directory_patterns = {
        "test": "testing",
        "tests": "testing",
        "spec": "testing",
        "__tests__": "testing",
        "docs": "documentation",
        "doc": "documentation",
        "config": "configuration",
        "configs": "configuration",
        "src": "source",
        "lib": "library",
        "scripts": "scripts",
        "tools": "tools",
        "build": "build",
        "dist": "build",
        "deployment": "deployment",
        "ci": "ci/cd",
        ".github": "github-actions",
        "assets": "assets",
        "static": "assets",
        "styles": "frontend",
        "components": "frontend",
        "pages": "frontend",
        "views": "frontend",
        "controllers": "backend",
        "models": "backend",
        "services": "backend",
        "api": "api",
        "routes": "api",
        "middleware": "backend",
        "database": "database",
        "migrations": "database",
        "seeds": "database"
    }

    # Special filename patterns
    filename_patterns = {
        "package.json": "dependencies",
        "package-lock.json": "dependencies",
        "yarn.lock": "dependencies",
        "requirements.txt": "dependencies",
        "pipfile": "dependencies",
        "poetry.lock": "dependencies",
        "gemfile": "dependencies",
        "cargo.toml": "dependencies",
        "pom.xml": "dependencies",
        "build.gradle": "dependencies",
        "docker-compose.yml": "docker",
        "docker-compose.yaml": "docker",
        ".gitignore": "configuration",
        ".env.example": "configuration",
        ".eslintrc": "configuration",
        ".prettierrc": "configuration",
        "tsconfig.json": "configuration",
        "webpack.config.js": "configuration",
        "babel.config.js": "configuration",
        "jest.config.js": "testing",
        "pytest.ini": "testing",
        "tox.ini": "testing"
    }

    # Analyze each file
    for file in files:
        if file.status == "removed":
            continue

        filename = file.filename.lower()

        # Check filename patterns first (most specific)
        for pattern, label in filename_patterns.items():
            if pattern in filename:
                labels.add(label)
                reasons[label] = f"Contains {pattern}"

        # Check directory patterns
        for pattern, label in directory_patterns.items():
            if f"/{pattern}/" in filename or filename.startswith(f"{pattern}/"):
                labels.add(label)
                reasons[label] = f"Modifies {pattern} directory"

        # Check file extensions
        for ext, label in extension_mapping.items():
            if filename.endswith(ext):
                labels.add(label)
                if ext in [".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs"]:
                    reasons[label] = f"Modifies {label} code"
                elif ext in [".md", ".rst"]:
                    reasons[label] = f"Updates {label}"
                else:
                    reasons[label] = f"Changes {label} file"

    return labels, reasons


def analyze_content_based_labels(files: List[Any], pr_title: str, pr_description: str) -> tuple[Set[str], Dict[str, str]]:
    """
    Generate labels based on content analysis of patches and text.
    """
    labels = set()
    reasons = {}

    # Combine all text for analysis
    all_text = f"{pr_title} {pr_description or ''}".lower()
    all_patches = " ".join([f.patch for f in files if f.patch]).lower()

    # Security-related patterns
    security_patterns = [
        r"secur", r"vulnerab", r"xss", r"sql injection", r"auth", r"authori",
        r"permission", r"encrypt", r"hash", r"token", r"jwt", r"oauth",
        r"password", r"secret", r"credential", r"csp", r"csrf"
    ]

    # Performance-related patterns
    performance_patterns = [
        r"perform", r"optimi", r"speed", r"fast", r"slow", r"memory",
        r"cache", r"lazy", r"async", r"parallel", r"batch", r"query",
        r"index", r"database", r"n\+1", r"loop"
    ]

    # Bug fix patterns
    bug_patterns = [
        r"fix", r"bug", r"error", r"issue", r"problem", r"crash",
        r"exception", r"fail", r"incorrect", r"wrong", r"regression"
    ]

    # Feature patterns
    feature_patterns = [
        r"add", r"new", r"implement", r"create", r"feature", r"enhance",
        r"improve", r"extend", r"support", r"enable"
    ]

    # Refactoring patterns
    refactor_patterns = [
        r"refactor", r"cleanup", r"reorgan", r"restructure", r"simplify",
        r"rework", r"rearrange", r"consolidat", r"modular"
    ]

    # Breaking change patterns
    breaking_patterns = [
        r"breaking", r"deprecat", r"remove", r"delete", r"replace",
        r"backward incompat", r"major change"
    ]

    # Testing patterns
    testing_patterns = [
        r"test", r"spec", r"mock", r"stub", r"assert", r"expect",
        r"coverage", r"pytest", r"jest", r"mocha", r"unit test"
    ]

    # Documentation patterns
    doc_patterns = [
        r"doc", r"readme", r"comment", r"document", r"guide", r"tutorial",
        r"example", r"usage", r"api"
    ]

    # Pattern definitions
    pattern_definitions = [
        (security_patterns, "security", "Contains security-related changes"),
        (performance_patterns, "performance", "Includes performance improvements"),
        (bug_patterns, "bugfix", "Fixes bugs or errors"),
        (feature_patterns, "enhancement", "Adds new functionality"),
        (refactor_patterns, "refactoring", "Code refactoring or cleanup"),
        (breaking_patterns, "breaking-change", "Contains breaking changes"),
        (testing_patterns, "testing", "Adds or modifies tests"),
        (doc_patterns, "documentation", "Updates documentation")
    ]

    # Check all patterns
    combined_text = f"{all_text} {all_patches}"
    for patterns, label, reason in pattern_definitions:
        if any(re.search(pattern, combined_text) for pattern in patterns):
            labels.add(label)
            reasons[label] = reason

    # Check for dependency changes
    if any(pattern in all_patches for pattern in [
        r"package\.json", r"requirements\.txt", r"pipfile", r"gemfile",
        r"cargo\.toml", r"pom\.xml", r"build\.gradle"
    ]):
        labels.add("dependencies")
        reasons["dependencies"] = "Updates project dependencies"

    # Check for database changes
    if any(pattern in all_patches for pattern in [
        r"migration", r"schema", r"table", r"column", r"sql", r"query"
    ]):
        labels.add("database")
        reasons["database"] = "Includes database changes"

    return labels, reasons


def analyze_size_based_labels(files: List[Any]) -> tuple[Set[str], Dict[str, str]]:
    """
    Generate labels based on the size and scope of changes.
    """
    labels = set()
    reasons = {}

    total_additions = sum(f.additions for f in files if hasattr(f, 'additions'))
    total_deletions = sum(f.deletions for f in files if hasattr(f, 'deletions'))
    total_changes = total_additions + total_deletions
    files_changed = len([f for f in files if f.status != "removed"])

    # Size-based labeling
    if total_changes == 0:
        size_label = "no-code-changes"
        size_reason = "No lines of code changed"
    elif total_changes <= 10:
        size_label = "trivial"
        size_reason = "Very small change (≤10 lines)"
    elif total_changes <= 50:
        size_label = "small"
        size_reason = "Small change (≤50 lines)"
    elif total_changes <= 200:
        size_label = "medium"
        size_reason = "Medium change (≤200 lines)"
    elif total_changes <= 500:
        size_label = "large"
        size_reason = "Large change (≤500 lines)"
    else:
        size_label = "xlarge"
        size_reason = "Very large change (>500 lines)"

    labels.add(size_label)
    reasons[size_label] = size_reason

    # File count labeling
    if files_changed == 1:
        labels.add("single-file")
        reasons["single-file"] = "Changes only one file"
    elif files_changed >= 10:
        labels.add("many-files")
        reasons["many-files"] = f"Changes {files_changed} files"

    # Deletion-heavy changes
    if total_deletions > total_additions * 2:
        labels.add("deletion-heavy")
        reasons["deletion-heavy"] = "Primarily removes code"

    # Addition-heavy changes
    if total_additions > total_deletions * 3:
        labels.add("addition-heavy")
        reasons["addition-heavy"] = "Primarily adds code"

    return labels, reasons


def analyze_type_based_labels(pr_title: str, pr_description: str, files: List[Any]) -> tuple[Set[str], Dict[str, str]]:
    """
    Generate labels based on the type of change (conventional commits).
    """
    labels = set()
    reasons = {}

    # Combine title and description for analysis
    text = f"{pr_title} {pr_description or ''}".lower()

    # Conventional commit type patterns
    type_patterns = [
        (r"^(feat|feature)\s*[:\(]", "feature", "Adds a new feature"),
        (r"^(fix|bugfix)\s*[:\(]", "bug", "Fixes a bug"),
        (r"^docs?\s*[:\(]", "documentation", "Documentation changes"),
        (r"^style\s*[:\(]", "style", "Code style changes (formatting, missing semicolons, etc.)"),
        (r"^(refactor|refact)\s*[:\(]", "refactoring", "Code refactoring without functional changes"),
        (r"^(perf|performance)\s*[:\(]", "performance", "Performance improvements"),
        (r"^test\s*[:\(]", "testing", "Adding or updating tests"),
        (r"^(build|ci)\s*[:\(]", "ci/cd", "Build system or CI/CD changes"),
        (r"^chore\s*[:\(]", "maintenance", "Maintenance tasks, dependency updates, etc."),
        (r"^revert\s*[:\(]", "revert", "Reverts previous changes")
    ]

    for pattern, label, reason in type_patterns:
        if re.search(pattern, text, re.MULTILINE):
            labels.add(label)
            reasons[label] = reason

    # Check for specific GitHub-related labels
    if "draft" in text:
        labels.add("draft")
        reasons["draft"] = "Marked as draft PR"

    if "wip" in text or "work in progress" in text:
        labels.add("work-in-progress")
        reasons["work-in-progress"] = "Work in progress"

    # Check for dependencies
    combined_patches = " ".join([f.patch for f in files if f.patch]).lower()
    if any(dep in combined_patches for dep in [
        "package.json", "requirements.txt", "yarn.lock", "package-lock.json",
        "pipfile", "gemfile", "cargo.toml", "pom.xml"
    ]):
        labels.add("dependencies")
        reasons["dependencies"] = "Updates project dependencies"

    return labels, reasons


def apply_labels_to_pr(pr, labels: List[str], dry_run: bool = False) -> bool:
    """
    Apply labels to a GitHub pull request.

    Args:
        pr: GitHub PullRequest object
        labels: List of labels to apply
        dry_run: If True, only returns what would be applied

    Returns:
        True if successful, False otherwise
    """
    if not labels:
        return True

    # Clean labels to ensure they're all strings
    clean_labels = []
    for label in labels:
        if isinstance(label, list):
            # If it's a list, take the first item or skip
            if label:
                clean_labels.append(str(label[0]))
        elif isinstance(label, str):
            clean_labels.append(label)
        else:
            # Convert to string if it's not a list or string
            clean_labels.append(str(label))

    if dry_run:
        try:
            print(f"Would apply labels: {', '.join(clean_labels)}")
        except Exception as e:
            print(f"Error in dry run label joining: {e}")
            print(f"Labels: {clean_labels}")
        return True

    try:
        # Get existing labels
        existing_labels = [label.name for label in pr.get_labels()]

        # Only add new labels
        new_labels = [label for label in clean_labels if label not in existing_labels]

        if new_labels:
            pr.add_to_labels(new_labels)
            try:
                print(f"✅ Applied labels: {', '.join(new_labels)}")
            except Exception as e:
                print(f"Error printing applied labels: {e}")
                print(f"Labels: {new_labels}")
        else:
            print("ℹ️ No new labels to apply")

        return True
    except Exception as e:
        print(f"❌ Error applying labels: {e}")
        return False


def create_missing_labels(repo, desired_labels: List[str]) -> Dict[str, bool]:
    """
    Create labels in the repository if they don't exist.

    Args:
        repo: GitHub Repository object
        desired_labels: List of desired label names

    Returns:
        Dictionary mapping label names to creation success status
    """
    # Define label colors and descriptions
    label_definitions = {
        "python": {"color": "3572A5", "description": "Python code changes"},
        "javascript": {"color": "F1E05A", "description": "JavaScript code changes"},
        "typescript": {"color": "2B7489", "description": "TypeScript code changes"},
        "react": {"color": "61DAFB", "description": "React component changes"},
        "java": {"color": "B07219", "description": "Java code changes"},
        "go": {"color": "00ADD8", "description": "Go code changes"},
        "rust": {"color": "DEA584", "description": "Rust code changes"},
        "documentation": {"color": "0075ca", "description": "Documentation changes"},
        "testing": {"color": "D4C5F9", "description": "Test related changes"},
        "security": {"color": "EE0701", "description": "Security related changes"},
        "performance": {"color": "E99695", "description": "Performance improvements"},
        "bugfix": {"color": "D73A49", "description": "Bug fixes"},
        "enhancement": {"color": "A2EEEF", "description": "New features or enhancements"},
        "refactoring": {"color": "FFEB8C", "description": "Code refactoring"},
        "breaking-change": {"color": "FBCA04", "description": "Breaking changes"},
        "dependencies": {"color": "0366D6", "description": "Dependency updates"},
        "database": {"color": "C2E0C6", "description": "Database related changes"},
        "api": {"color": "1D76DB", "description": "API changes"},
        "frontend": {"color": "FEF2C0", "description": "Frontend changes"},
        "backend": {"color": "BFDADC", "description": "Backend changes"},
        "docker": {"color": "0E8A16", "description": "Docker related changes"},
        "ci/cd": {"color": "0075CA", "description": "CI/CD pipeline changes"},
        "configuration": {"color": "FBCA04", "description": "Configuration changes"},
        "assets": {"color": "E4B2E6", "description": "Asset file changes"},
        "trivial": {"color": "E4B2E6", "description": "Trivial changes"},
        "small": {"color": "FAD8C7", "description": "Small changes"},
        "medium": {"color": "FAD8C7", "description": "Medium changes"},
        "large": {"color": "FAD8C7", "description": "Large changes"},
        "xlarge": {"color": "FAD8C7", "description": "Very large changes"},
        "draft": {"color": "D4C5F9", "description": "Draft pull request"},
        "work-in-progress": {"color": "FBCA04", "description": "Work in progress"},
        "maintenance": {"color": "FAD8C7", "description": "Maintenance tasks"}
    }

    results = {}
    existing_labels = {label.name for label in repo.get_labels()}

    for label in desired_labels:
        if label in existing_labels:
            results[label] = True  # Already exists
            continue

        label_info = label_definitions.get(label, {
            "color": "CCCCCC",  # Default gray color
            "description": f"Label for {label} changes"
        })

        try:
            repo.create_label(
                name=label,
                color=label_info["color"],
                description=label_info["description"]
            )
            results[label] = True
            print(f"✅ Created label: {label}")
        except Exception as e:
            results[label] = False
            print(f"❌ Failed to create label {label}: {e}")

    return results


def get_label_priority_order() -> List[str]:
    """
    Return a list of labels in priority order for display.
    Important labels should appear first.
    """
    return [
        "breaking-change",
        "security",
        "bugfix",
        "enhancement",
        "performance",
        "api",
        "database",
        "frontend",
        "backend",
        "documentation",
        "testing",
        "dependencies",
        "refactoring",
        "maintenance",
        "ci/cd",
        "docker",
        "configuration",
        "python",
        "javascript",
        "typescript",
        "react",
        "java",
        "go",
        "rust",
        "draft",
        "work-in-progress",
        "xlarge",
        "large",
        "medium",
        "small",
        "trivial"
    ]