# backend/review_lenses.py
"""
Multi-lens review system for specialized code analysis.
Provides different perspectives: Security, Performance, Best Practices, etc.
"""

import json
import google.generativeai as genai
from typing import List, Dict, Any, Optional
from backend.utils import detect_language_from_filename
from backend.config import GEMINI_API_KEY

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)


class ReviewLens:
    """Base class for specialized review lenses."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def analyze(self, patch: str, filename: str, context: List[Dict] = None) -> Dict[str, Any]:
        """
        Analyze code patch using this specific lens.

        Args:
            patch: The diff patch to analyze
            filename: The file being analyzed
            context: Additional context from semantic search

        Returns:
            Analysis results with comments and recommendations
        """
        raise NotImplementedError("Subclasses must implement analyze method")

    def create_prompt(self, patch: str, filename: str, context: List[Dict] = None) -> str:
        """Create the analysis prompt for this lens."""
        raise NotImplementedError("Subclasses must implement create_prompt method")


class SecurityLens(ReviewLens):
    """Security-focused review lens for vulnerability detection."""

    def __init__(self):
        super().__init__("security", "Security vulnerability detection and analysis")

    def analyze(self, patch: str, filename: str, context: List[Dict] = None) -> Dict[str, Any]:
        """Analyze code for security vulnerabilities."""
        prompt = self.create_prompt(patch, filename, context)

        model = genai.GenerativeModel("gemini-2.5-pro")

        try:
            response = model.generate_content(prompt)
            if response and response.text:
                return self.parse_security_response(response.text, filename)
        except Exception as e:
            print(f"Error in security analysis: {e}")

        return {
            "lens": self.name,
            "filename": filename,
            "issues": [],
            "summary": "Security analysis completed with no issues found."
        }

    def create_prompt(self, patch: str, filename: str, context: List[Dict] = None) -> str:
        """Create security-focused analysis prompt."""
        language = detect_language_from_filename(filename)

        context_str = ""
        if context:
            context_str = f"\n\nContext from related code:\n{json.dumps(context[:3], indent=2)}"

        return f"""
You are a cybersecurity expert reviewing code changes for security vulnerabilities.

File: {filename}
Language: {language}

Diff to review:
```
{patch}
```
{context_str}

Focus on these security issues:
1. **Injection Vulnerabilities**: SQL injection, command injection, XSS, etc.
2. **Authentication/Authorization**: Missing checks, weak auth, privilege escalation
3. **Data Validation**: Input sanitization, parameter validation, type safety
4. **Cryptographic Issues**: Weak encryption, hardcoded secrets, random number generation
5. **Information Disclosure**: Sensitive data exposure, error messages, debugging info
6. **Session Management**: Session fixation, CSRF, token handling
7. **File Operations**: Path traversal, file upload vulnerabilities
8. **Network Security**: HTTPS usage, certificate validation, API security

For each security issue found:
1. Specify the exact line number (using "Line X:" format)
2. Describe the vulnerability and its potential impact
3. Provide specific remediation steps
4. Rate severity: Critical/High/Medium/Low

Format your response as:
Line X: [SEVERITY] VULNERABILITY_TYPE - Brief description
Details: [Detailed explanation]
Impact: [Potential consequences]
Fix: [Specific remediation steps]

Only report actual security issues. If no issues are found, respond with "No security vulnerabilities detected."
"""

    def parse_security_response(self, response: str, filename: str) -> Dict[str, Any]:
        """Parse security analysis response."""
        lines = response.strip().split('\n')
        issues = []
        current_issue = {}

        for line in lines:
            line = line.strip()

            # Check for new issue (starts with "Line X:")
            if line.startswith("Line ") and ":" in line:
                if current_issue:
                    issues.append(current_issue)

                # Parse severity and vulnerability type
                severity = "Medium"  # default
                vuln_type = "Unknown"

                if "[" in line and "]" in line:
                    severity_part = line.split("[")[1].split("]")[0]
                    severity = severity_part

                # Extract vulnerability type
                if " - " in line:
                    vuln_type = line.split(" - ")[1].split(" - ")[0] if " - " in line.split(" - ")[1] else line.split(" - ")[1]

                current_issue = {
                    "line": line.split(":")[0].replace("Line ", ""),
                    "severity": severity,
                    "type": vuln_type,
                    "description": line.split(" - ", 1)[-1] if " - " in line else line,
                    "details": "",
                    "impact": "",
                    "fix": ""
                }

            elif line.startswith("Details:"):
                current_issue["details"] = line.replace("Details:", "").strip()
            elif line.startswith("Impact:"):
                current_issue["impact"] = line.replace("Impact:", "").strip()
            elif line.startswith("Fix:"):
                current_issue["fix"] = line.replace("Fix:", "").strip()
            elif current_issue and not any(line.startswith(prefix) for prefix in ["Line ", "Details:", "Impact:", "Fix:"]):
                # Continue previous field
                if current_issue["details"] and not current_issue["impact"]:
                    current_issue["details"] += " " + line
                elif current_issue["impact"] and not current_issue["fix"]:
                    current_issue["impact"] += " " + line
                elif current_issue["fix"]:
                    current_issue["fix"] += " " + line

        if current_issue:
            issues.append(current_issue)

        # Convert to GitHub comment format
        comments = []
        for issue in issues:
            comment_body = f"🔒 **Security Issue ({issue['severity']})**\n\n"
            comment_body += f"**Type:** {issue['type']}\n\n"
            comment_body += f"**Description:** {issue['description']}\n\n"
            if issue['details']:
                comment_body += f"**Details:** {issue['details']}\n\n"
            if issue['impact']:
                comment_body += f"**Impact:** {issue['impact']}\n\n"
            if issue['fix']:
                comment_body += f"**Fix:** {issue['fix']}"

            comments.append({
                "path": filename,
                "body": comment_body,
                "line": issue["line"] if issue["line"].isdigit() else None
            })

        return {
            "lens": self.name,
            "filename": filename,
            "issues": issues,
            "comments": comments,
            "summary": f"Found {len(issues)} security issue{'s' if len(issues) != 1 else ''} in {filename}"
        }


class PerformanceLens(ReviewLens):
    """Performance-focused review lens for optimization opportunities."""

    def __init__(self):
        super().__init__("performance", "Performance optimization and efficiency analysis")

    def analyze(self, patch: str, filename: str, context: List[Dict] = None) -> Dict[str, Any]:
        """Analyze code for performance issues."""
        prompt = self.create_prompt(patch, filename, context)

        model = genai.GenerativeModel("gemini-2.5-pro")

        try:
            response = model.generate_content(prompt)
            if response and response.text:
                return self.parse_performance_response(response.text, filename)
        except Exception as e:
            print(f"Error in performance analysis: {e}")

        return {
            "lens": self.name,
            "filename": filename,
            "issues": [],
            "summary": "Performance analysis completed with no issues found."
        }

    def create_prompt(self, patch: str, filename: str, context: List[Dict] = None) -> str:
        """Create performance-focused analysis prompt."""
        language = detect_language_from_filename(filename)

        context_str = ""
        if context:
            context_str = f"\n\nContext from related code:\n{json.dumps(context[:3], indent=2)}"

        return f"""
You are a performance optimization expert reviewing code changes for efficiency improvements.

File: {filename}
Language: {language}

Diff to review:
```
{patch}
```
{context_str}

Focus on these performance issues:
1. **Algorithmic Complexity**: O(n²) where O(n) would work, inefficient loops
2. **Database Queries**: N+1 query problems, missing indexes, inefficient joins
3. **Memory Usage**: Memory leaks, excessive allocations, large object creation
4. **I/O Operations**: File operations, network calls, blocking operations
5. **Caching Opportunities**: Missing cache, cache invalidation issues
6. **Concurrency**: Race conditions, lock contention, thread safety
7. **Resource Management**: Connection pooling, resource cleanup
8. **Data Structures**: Inappropriate data structure choices

For each performance issue found:
1. Specify the exact line number (using "Line X:" format)
2. Describe the performance problem
3. Explain the impact (time/space complexity)
4. Provide specific optimization recommendations
5. Rate impact: Critical/High/Medium/Low

Format your response as:
Line X: [IMPACT] ISSUE_TYPE - Brief description
Problem: [Detailed explanation of the performance issue]
Complexity: [Time/space complexity impact]
Optimization: [Specific performance improvement]

Only report actual performance issues. If no issues are found, respond with "No performance issues detected."
"""

    def parse_performance_response(self, response: str, filename: str) -> Dict[str, Any]:
        """Parse performance analysis response."""
        lines = response.strip().split('\n')
        issues = []
        current_issue = {}

        for line in lines:
            line = line.strip()

            if line.startswith("Line ") and ":" in line:
                if current_issue:
                    issues.append(current_issue)

                impact = "Medium"
                if "[" in line and "]" in line:
                    impact = line.split("[")[1].split("]")[0]

                issue_type = "Performance Issue"
                if " - " in line:
                    issue_type = line.split(" - ")[1]

                current_issue = {
                    "line": line.split(":")[0].replace("Line ", ""),
                    "impact": impact,
                    "type": issue_type,
                    "description": issue_type,
                    "problem": "",
                    "complexity": "",
                    "optimization": ""
                }

            elif line.startswith("Problem:"):
                current_issue["problem"] = line.replace("Problem:", "").strip()
            elif line.startswith("Complexity:"):
                current_issue["complexity"] = line.replace("Complexity:", "").strip()
            elif line.startswith("Optimization:"):
                current_issue["optimization"] = line.replace("Optimization:", "").strip()

        if current_issue:
            issues.append(current_issue)

        # Convert to GitHub comment format
        comments = []
        for issue in issues:
            comment_body = f"⚡ **Performance Issue ({issue['impact']})**\n\n"
            comment_body += f"**Type:** {issue['type']}\n\n"
            if issue['problem']:
                comment_body += f"**Problem:** {issue['problem']}\n\n"
            if issue['complexity']:
                comment_body += f"**Complexity Impact:** {issue['complexity']}\n\n"
            if issue['optimization']:
                comment_body += f"**Optimization:** {issue['optimization']}"

            comments.append({
                "path": filename,
                "body": comment_body,
                "line": issue["line"] if issue["line"].isdigit() else None
            })

        return {
            "lens": self.name,
            "filename": filename,
            "issues": issues,
            "comments": comments,
            "summary": f"Found {len(issues)} performance issue{'s' if len(issues) != 1 else ''} in {filename}"
        }


class BestPracticesLens(ReviewLens):
    """Best practices-focused review lens for code quality."""

    def __init__(self):
        super().__init__("best_practices", "Code quality and best practices analysis")

    def analyze(self, patch: str, filename: str, context: List[Dict] = None) -> Dict[str, Any]:
        """Analyze code for best practices violations."""
        prompt = self.create_prompt(patch, filename, context)

        model = genai.GenerativeModel("gemini-2.5-pro")

        try:
            response = model.generate_content(prompt)
            if response and response.text:
                return self.parse_best_practices_response(response.text, filename)
        except Exception as e:
            print(f"Error in best practices analysis: {e}")

        return {
            "lens": self.name,
            "filename": filename,
            "issues": [],
            "summary": "Best practices analysis completed with no issues found."
        }

    def create_prompt(self, patch: str, filename: str, context: List[Dict] = None) -> str:
        """Create best practices-focused analysis prompt."""
        language = detect_language_from_filename(filename)

        context_str = ""
        if context:
            context_str = f"\n\nContext from related code:\n{json.dumps(context[:3], indent=2)}"

        return f"""
You are a senior software engineer reviewing code for adherence to best practices and coding standards.

File: {filename}
Language: {language}

Diff to review:
```
{patch}
```
{context_str}

Focus on these best practices:
1. **SOLID Principles**: Single responsibility, Open/closed, Liskov substitution, Interface segregation, Dependency inversion
2. **Code Organization**: Proper structure, separation of concerns, modularity
3. **Naming Conventions**: Clear, descriptive variable/function/class names
4. **Error Handling**: Proper exception handling, error messages, graceful failures
5. **Code Duplication**: DRY principle violations, repeated code patterns
6. **Comments and Documentation**: Clear comments, documentation standards
7. **Testing**: Testability, test coverage, mocking
8. **Security**: Input validation, output encoding, secure defaults
9. **Maintainability**: Readability, complexity management, technical debt
10. **Language-specific idioms**: Proper use of language features and patterns

For each best practices issue found:
1. Specify the exact line number (using "Line X:" format)
2. Identify which principle or practice is violated
3. Explain why it's a problem
4. Suggest improvement
5. Rate importance: Critical/High/Medium/Low

Format your response as:
Line X: [IMPORTANCE] PRINCIPLE_VIOLATED - Brief description
Issue: [Detailed explanation of the best practices violation]
Why it matters: [Explanation of the impact]
Suggestion: [Specific improvement recommendation]

Only report actual best practices violations. If no issues are found, respond with "No best practices issues detected."
"""

    def parse_best_practices_response(self, response: str, filename: str) -> Dict[str, Any]:
        """Parse best practices analysis response."""
        lines = response.strip().split('\n')
        issues = []
        current_issue = {}

        for line in lines:
            line = line.strip()

            if line.startswith("Line ") and ":" in line:
                if current_issue:
                    issues.append(current_issue)

                importance = "Medium"
                if "[" in line and "]" in line:
                    importance = line.split("[")[1].split("]")[0]

                principle = "Best Practice"
                if " - " in line:
                    principle = line.split(" - ")[1]

                current_issue = {
                    "line": line.split(":")[0].replace("Line ", ""),
                    "importance": importance,
                    "principle": principle,
                    "description": principle,
                    "issue": "",
                    "why_matters": "",
                    "suggestion": ""
                }

            elif line.startswith("Issue:"):
                current_issue["issue"] = line.replace("Issue:", "").strip()
            elif line.startswith("Why it matters:"):
                current_issue["why_matters"] = line.replace("Why it matters:", "").strip()
            elif line.startswith("Suggestion:"):
                current_issue["suggestion"] = line.replace("Suggestion:", "").strip()

        if current_issue:
            issues.append(current_issue)

        # Convert to GitHub comment format
        comments = []
        for issue in issues:
            comment_body = f"✅ **Best Practices Issue ({issue['importance']})**\n\n"
            comment_body += f"**Principle:** {issue['principle']}\n\n"
            if issue['issue']:
                comment_body += f"**Issue:** {issue['issue']}\n\n"
            if issue['why_matters']:
                comment_body += f"**Why it matters:** {issue['why_matters']}\n\n"
            if issue['suggestion']:
                comment_body += f"**Suggestion:** {issue['suggestion']}"

            comments.append({
                "path": filename,
                "body": comment_body,
                "line": issue["line"] if issue["line"].isdigit() else None
            })

        return {
            "lens": self.name,
            "filename": filename,
            "issues": issues,
            "comments": comments,
            "summary": f"Found {len(issues)} best practices issue{'s' if len(issues) != 1 else ''} in {filename}"
        }


def multi_lens_review(patch: str, filename: str, lenses: List[str] = None,
                     context: List[Dict] = None) -> Dict[str, Any]:
    """
    Run multiple review lenses on a code patch.

    Args:
        patch: The diff patch to analyze
        filename: The file being analyzed
        lenses: List of lens names to run (default: all lenses)
        context: Additional context from semantic search

    Returns:
        Combined analysis results from all specified lenses
    """
    if lenses is None:
        lenses = ["security", "performance", "best_practices"]

    # Initialize lens instances
    lens_instances = {
        "security": SecurityLens(),
        "performance": PerformanceLens(),
        "best_practices": BestPracticesLens()
    }

    results = {
        "filename": filename,
        "lenses_run": lenses,
        "total_issues": 0,
        "all_comments": [],
        "lens_results": {}
    }

    for lens_name in lenses:
        if lens_name in lens_instances:
            try:
                lens_result = lens_instances[lens_name].analyze(patch, filename, context)
                results["lens_results"][lens_name] = lens_result
                results["total_issues"] += len(lens_result.get("issues", []))
                results["all_comments"].extend(lens_result.get("comments", []))
            except Exception as e:
                print(f"Error running {lens_name} lens: {e}")
                results["lens_results"][lens_name] = {
                    "error": str(e),
                    "summary": f"Error in {lens_name} analysis"
                }

    # Generate combined summary
    lens_summaries = []
    for lens_name, result in results["lens_results"].items():
        if "summary" in result:
            lens_summaries.append(result["summary"])

    results["summary"] = f"Multi-lens analysis complete. {results['total_issues']} total issues found across {len(lenses)} lenses.\n" + "\n".join(lens_summaries)

    return results


def get_available_lenses() -> Dict[str, str]:
    """Get information about all available review lenses."""
    return {
        "security": "Security vulnerability detection and analysis",
        "performance": "Performance optimization and efficiency analysis",
        "best_practices": "Code quality and best practices analysis"
    }