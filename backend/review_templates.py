# backend/review_templates.py
"""
Review templates and custom checklists system.
Provides structured review frameworks for different types of changes and team-specific policies.
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import os
import re

from backend.config import DATA_DIR


@dataclass
class ReviewTemplate:
    """Represents a review template with criteria and checklists."""
    id: str
    name: str
    description: str
    language: str  # 'all', 'python', 'javascript', etc.
    file_patterns: List[str]  # File patterns this template applies to
    priority: int  # 1=high, 2=medium, 3=low
    is_enabled: bool
    criteria: List[Dict[str, Any]]  # Review criteria
    checklist: List[Dict[str, Any]]  # Checklist items
    auto_apply: bool  # Whether to automatically apply this template


@dataclass
class ChecklistItem:
    """Represents a single checklist item."""
    id: str
    title: str
    description: str
    category: str  # 'security', 'performance', 'code_quality', etc.
    severity: str  # 'critical', 'high', 'medium', 'low'
    check_type: str  # 'manual', 'automated', 'ai_assisted'
    validation_pattern: Optional[str]  # Regex pattern for automated checking
    suggestion: Optional[str]  # Suggested action if item fails


class ReviewTemplateManager:
    """Manages review templates and custom checklists."""

    def __init__(self):
        self.templates_file = DATA_DIR / "templates" / "review_templates.json"
        self.templates_file.parent.mkdir(parents=True, exist_ok=True)
        self.templates = {}
        self.load_templates()

    def load_templates(self):
        """Load templates from file."""
        try:
            if self.templates_file.exists():
                with open(self.templates_file, 'r') as f:
                    data = json.load(f)

                self.templates = {
                    template_id: ReviewTemplate(**template_data)
                    for template_id, template_data in data.get('templates', {}).items()
                }
        except Exception as e:
            print(f"Error loading templates: {e}")
            self.create_default_templates()

    def save_templates(self):
        """Save templates to file."""
        try:
            data = {
                'templates': {
                    template_id: asdict(template)
                    for template_id, template in self.templates.items()
                },
                'last_updated': json.dumps({"timestamp": "now"}, default=str)
            }

            with open(self.templates_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving templates: {e}")

    def create_default_templates(self):
        """Create default review templates."""
        default_templates = {
            "security_critical": ReviewTemplate(
                id="security_critical",
                name="Security Critical Review",
                description="Comprehensive security review for critical security changes",
                language="all",
                file_patterns=["**/*auth*", "**/*security*", "**/*password*", "**/*token*"],
                priority=1,
                is_enabled=True,
                criteria=[
                    {
                        "title": "Input Validation",
                        "description": "All user inputs must be properly validated and sanitized",
                        "check_pattern": r"(validate|sanitize|clean|escape)"
                    },
                    {
                        "title": "Authentication/Authorization",
                        "description": "Proper authentication and authorization mechanisms must be in place",
                        "check_pattern": r"(auth|authorize|permission|login)"
                    },
                    {
                        "title": "Sensitive Data Handling",
                        "description": "Sensitive data must be properly protected and not logged",
                        "check_pattern": r"(password|secret|key|token|credential)"
                    },
                    {
                        "title": "SQL Injection Prevention",
                        "description": "Must use parameterized queries or ORM to prevent SQL injection",
                        "check_pattern": r"(query|sql)\s*(execute|query|run)"
                    }
                ],
                checklist=[
                    ChecklistItem(
                        id="sec_001",
                        title="Input validation implemented",
                        description="Are all user inputs validated before processing?",
                        category="security",
                        severity="critical",
                        check_type="manual",
                        validation_pattern=r"(validate|sanitize|clean|escape)",
                        suggestion="Add input validation functions for all user inputs"
                    ),
                    ChecklistItem(
                        id="sec_002",
                        title="SQL injection prevention",
                        description="Are database queries protected against SQL injection?",
                        category="security",
                        severity="critical",
                        check_type="automated",
                        validation_pattern=r"(prepare|execute|cursor)",
                        suggestion="Use parameterized queries or prepared statements"
                    ),
                    ChecklistItem(
                        id="sec_003",
                        title="HTTPS usage",
                        description="Are all external API calls using HTTPS?",
                        category="security",
                        severity="high",
                        check_type="manual",
                        validation_pattern=r"https://",
                        suggestion="Ensure all external API calls use HTTPS protocol"
                    )
                ],
                auto_apply=True
            ),
            "performance_optimization": ReviewTemplate(
                id="performance_optimization",
                name="Performance Optimization Review",
                description="Review focused on performance improvements and optimizations",
                language="all",
                file_patterns=["**/*cache*", "**/*optimize*", "**/*performance*", "**/*database*"],
                priority=2,
                is_enabled=True,
                criteria=[
                    {
                        "title": "Database Query Optimization",
                        "description": "Database queries should be optimized to prevent N+1 problems",
                        "check_pattern": r"(query|select|fetch|find)"
                    },
                    {
                        "title": "Caching Implementation",
                        "description": "Appropriate caching mechanisms should be implemented",
                        "check_pattern": r"(cache|redis|memcached|store)"
                    },
                    {
                        "title": "Memory Management",
                        "description": "Memory usage should be efficient and leaks should be prevented",
                        "check_pattern": r"(memory|malloc|free|gc|collect)"
                    }
                ],
                checklist=[
                    ChecklistItem(
                        id="perf_001",
                        title="Database query optimization",
                        description="Are database queries optimized to avoid N+1 problems?",
                        category="performance",
                        severity="high",
                        check_type="manual",
                        validation_pattern=r"JOIN|WHERE|INDEX",
                        suggestion="Use JOIN optimization, proper WHERE clauses, and database indexing"
                    ),
                    ChecklistItem(
                        id="perf_002",
                        title="Loop optimization",
                        description="Are loops optimized for performance?",
                        category="performance",
                        severity="medium",
                        check_type="automated",
                        validation_pattern=r"for\s+.*in\s+.*:",
                        suggestion="Optimize loop conditions and consider using list comprehensions"
                    ),
                    ChecklistItem(
                        id="perf_003",
                        title="Resource cleanup",
                        description="Are resources properly cleaned up after use?",
                        category="performance",
                        severity="medium",
                        check_type="manual",
                        validation_pattern=r"(close|dispose|cleanup|finally)",
                        suggestion="Ensure proper resource cleanup in finally blocks or context managers"
                    )
                ],
                auto_apply=True
            ),
            "code_quality_python": ReviewTemplate(
                id="code_quality_python",
                name="Python Code Quality Review",
                description="Python-specific code quality and best practices review",
                language="python",
                file_patterns=["**/*.py"],
                priority=2,
                is_enabled=True,
                criteria=[
                    {
                        "title": "PEP 8 Compliance",
                        "description": "Code should follow PEP 8 style guidelines",
                        "check_pattern": r"(class|def|import|from)"
                    },
                    {
                        "title": "Type Hints",
                        "description": "Functions should have proper type hints",
                        "check_pattern": r"def\s+\w+.*:"
                    },
                    {
                        "title": "Error Handling",
                        "description": "Proper error handling should be implemented",
                        "check_pattern": r"(try:|except|raise|finally)"
                    }
                ],
                checklist=[
                    ChecklistItem(
                        id="py_001",
                        title="PEP 8 compliance",
                        description="Does the code follow PEP 8 style guidelines?",
                        category="code_quality",
                        severity="medium",
                        check_type="automated",
                        validation_pattern=r"[A-Z_][a-z0-9_]*",
                        suggestion="Use snake_case for variable and function names"
                    ),
                    ChecklistItem(
                        id="py_002",
                        title="Type hints",
                        description="Are functions and methods properly type hinted?",
                        category="code_quality",
                        severity="low",
                        check_type="manual",
                        validation_pattern=r"def\s+\w+.*:",
                        suggestion="Add type hints to function signatures"
                    ),
                    ChecklistItem(
                        id="py_003",
                        title="Exception handling",
                        description="Is appropriate exception handling implemented?",
                        category="code_quality",
                        severity="high",
                        check_type="manual",
                        validation_pattern=r"(try:|except|finally)",
                        suggestion="Add proper try-except blocks for error handling"
                    )
                ],
                auto_apply=True
            ),
            "code_quality_javascript": ReviewTemplate(
                id="code_quality_javascript",
                name="JavaScript/TypeScript Code Quality",
                description="JavaScript and TypeScript code quality and best practices review",
                language="javascript",
                file_patterns=["**/*.js", "**/*.ts", "**/*.jsx", "**/*.tsx"],
                priority=2,
                is_enabled=True,
                criteria=[
                    {
                        "title": "ESLint Compliance",
                        "description": "Code should pass ESLint rules",
                        "check_pattern": r"(let|const|var|function|class)"
                    },
                    {
                        "title": "Async/Await Usage",
                        "description": "Async operations should use async/await pattern",
                        "check_pattern": r"(async|await|Promise|then)"
                    },
                    {
                        "title": "Error Handling",
                        "description": "Promises should have proper error handling",
                        "check_pattern": r"\.catch\(|\.then\(.*\)|try\s*{"
                    }
                ],
                checklist=[
                    ChecklistItem(
                        id="js_001",
                        title="ESLint compliance",
                        description="Does the code pass ESLint rules?",
                        category="code_quality",
                        severity="medium",
                        check_type="automated",
                        validation_pattern=r"(let|const)\s+[a-zA-Z]",
                        suggestion="Use 'let' or 'const' instead of 'var'"
                    ),
                    ChecklistItem(
                        id="js_002",
                        title="Promise handling",
                        description="Are Promises properly handled with catch blocks?",
                        category="code_quality",
                        severity="high",
                        check_type="manual",
                        validation_pattern=r"\.catch\(",
                        suggestion="Add proper catch blocks for Promise error handling"
                    ),
                    ChecklistItem(
                        id="js_003",
                        title="Async/Await usage",
                        description="Are async operations using async/await?",
                        category="code_quality",
                        severity="low",
                        check_type="manual",
                        validation_pattern=r"await\s+[a-zA-Z]",
                        suggestion="Use async/await for cleaner async code"
                    )
                ],
                auto_apply=True
            ),
            "database_schema": ReviewTemplate(
                id="database_schema",
                name="Database Schema Review",
                description="Review for database schema changes and migrations",
                language="all",
                file_patterns=["**/*.sql", "**/*migration*", "**/*schema*"],
                priority=1,
                is_enabled=True,
                criteria=[
                    {
                        "title": "Indexing Strategy",
                        "description": "Appropriate indexes should be created for query performance",
                        "check_pattern": r"(CREATE\s+INDEX|INDEX\s+\w+)"
                    },
                    {
                        "title": "Foreign Key Constraints",
                        "description": "Foreign key relationships should be properly defined",
                        "check_pattern": r"(FOREIGN\s+KEY|REFERENCES)"
                    },
                    {
                        "title": "Data Validation Rules",
                        "description": "Database constraints should enforce data integrity",
                        "check_pattern": r"(CONSTRAINT|CHECK|UNIQUE|NOT\s+NULL)"
                    }
                ],
                checklist=[
                    ChecklistItem(
                        id="db_001",
                        title="Index optimization",
                        description="Are indexes created for frequently queried columns?",
                        category="performance",
                        severity="high",
                        check_type="manual",
                        validation_pattern=r"CREATE\s+INDEX",
                        suggestion="Add indexes for columns used in WHERE clauses and JOINs"
                    ),
                    ChecklistItem(
                        id="db_002",
                        title="Foreign key constraints",
                        description="Are foreign key relationships properly defined?",
                        category="data_integrity",
                        severity="high",
                        check_type="manual",
                        validation_pattern=r"FOREIGN\s+KEY",
                        suggestion="Add foreign key constraints to maintain referential integrity"
                    ),
                    ChecklistItem(
                        id="db_003",
                        title="Migration rollback",
                        description="Are rollback procedures available for schema changes?",
                        category="safety",
                        severity="critical",
                        check_type="manual",
                        validation_pattern=r"(ROLLBACK|DOWN)",
                        suggestion="Ensure rollback scripts are available for database migrations"
                    )
                ],
                auto_apply=True
            )
        }

        self.templates = default_templates
        self.save_templates()

    def get_template_for_file(self, filename: str, language: str = None) -> List[ReviewTemplate]:
        """Get applicable templates for a given file."""
        applicable_templates = []

        for template in self.templates.values():
            if not template.is_enabled:
                continue

            # Check language filter
            if template.language != 'all' and language:
                if template.language.lower() != language.lower():
                    continue

            # Check file patterns
            for pattern in template.file_patterns:
                if self.matches_pattern(filename, pattern):
                    applicable_templates.append(template)
                    break

        # Sort by priority
        applicable_templates.sort(key=lambda t: t.priority)
        return applicable_templates

    def matches_pattern(self, filename: str, pattern: str) -> bool:
        """Check if filename matches a pattern."""
        # Simple glob pattern matching
        pattern = pattern.lower()
        filename = filename.lower()

        # Handle ** wildcard
        if '**' in pattern:
            parts = pattern.split('**')
            if len(parts) == 2:
                prefix = parts[0]
                suffix = parts[1]
                return filename.startswith(prefix) and filename.endswith(suffix)

        # Handle * wildcard
        if '*' in pattern:
            pattern = pattern.replace('*', '.*')
            return re.match(pattern, filename) is not None

        # Exact match
        return filename == pattern

    def apply_template_to_review(self, template: ReviewTemplate, patch: str, filename: str) -> Dict[str, Any]:
        """Apply a template to generate review results."""

        results = {
            "template_id": template.id,
            "template_name": template.name,
            "applied_criteria": [],
            "checklist_results": [],
            "overall_score": 0
        }

        # Check criteria
        for criterion in template.criteria:
            criterion_result = self.check_criterion(criterion, patch, filename)
            results["applied_criteria"].append(criterion_result)

        # Check checklist items
        for checklist_item in template.checklist:
            checklist_result = self.check_checklist_item(checklist_item, patch, filename)
            results["checklist_results"].append(checklist_result)

        # Calculate overall score
        total_items = len(results["checklist_results"])
        passed_items = sum(1 for item in results["checklist_results"] if item["passed"])

        if total_items > 0:
            results["overall_score"] = (passed_items / total_items) * 100

        return results

    def check_criterion(self, criterion: Dict[str, Any], patch: str, filename: str) -> Dict[str, Any]:
        """Check if a criterion is met."""
        pattern = criterion.get("check_pattern", "")
        if not pattern:
            return {"title": criterion["title"], "met": True, "details": "No validation pattern provided"}

        pattern_found = re.search(pattern, patch, re.IGNORECASE)
        return {
            "title": criterion["title"],
            "met": bool(pattern_found),
            "details": f"Pattern '{pattern}' {'found' if pattern_found else 'not found'} in the code"
        }

    def check_checklist_item(self, item: ChecklistItem, patch: str, filename: str) -> Dict[str, Any]:
        """Check a checklist item."""
        result = {
            "id": item.id,
            "title": item.title,
            "category": item.category,
            "severity": item.severity,
            "passed": True,
            "details": "Check passed",
            "suggestion": item.suggestion
        }

        # Automated checking if pattern is provided
        if item.validation_pattern and item.check_type == "automated":
            pattern_found = re.search(item.validation_pattern, patch, re.IGNORECASE)
            result["passed"] = bool(pattern_found)
            result["details"] = f"Validation pattern '{item.validation_pattern}' {'found' if pattern_found else 'not found'}"

        elif item.check_type == "manual":
            # Manual items default to not passed (require human verification)
            result["passed"] = False
            result["details"] = "Manual verification required"

        return result

    def create_custom_template(self, template_data: Dict[str, Any]) -> str:
        """Create a custom review template."""
        try:
            # Validate required fields
            required_fields = ['id', 'name', 'description', 'language', 'file_patterns']
            for field in required_fields:
                if field not in template_data:
                    raise ValueError(f"Missing required field: {field}")

            # Create template
            template = ReviewTemplate(
                id=template_data['id'],
                name=template_data['name'],
                description=template_data['description'],
                language=template_data['language'],
                file_patterns=template_data['file_patterns'],
                priority=template_data.get('priority', 2),
                is_enabled=template_data.get('is_enabled', True),
                criteria=template_data.get('criteria', []),
                checklist=template_data.get('checklist', []),
                auto_apply=template_data.get('auto_apply', False)
            )

            # Check for duplicate ID
            if template.id in self.templates:
                raise ValueError(f"Template with ID '{template.id}' already exists")

            self.templates[template.id] = template
            self.save_templates()

            return f"Created template '{template.name}' with ID '{template.id}'"

        except Exception as e:
            return f"Error creating template: {e}"

    def update_template(self, template_id: str, updates: Dict[str, Any]) -> str:
        """Update an existing template."""
        try:
            if template_id not in self.templates:
                return f"Template with ID '{template_id}' not found"

            template = self.templates[template_id]

            # Update allowed fields
            updatable_fields = ['name', 'description', 'priority', 'is_enabled', 'auto_apply']
            for field, value in updates.items():
                if field in updatable_fields:
                    setattr(template, field, value)

            self.save_templates()
            return f"Updated template '{template.name}'"

        except Exception as e:
            return f"Error updating template: {e}"

    def delete_template(self, template_id: str) -> str:
        """Delete a review template."""
        try:
            if template_id not in self.templates:
                return f"Template with ID '{template_id}' not found"

            template_name = self.templates[template_id].name
            del self.templates[template_id]
            self.save_templates()

            return f"Deleted template '{template_name}'"

        except Exception as e:
            return f"Error deleting template: {e}"

    def list_templates(self, language: str = None, enabled_only: bool = False) -> List[Dict[str, Any]]:
        """List available templates."""
        templates = []

        for template in self.templates.values():
            # Apply filters
            if language and template.language != 'all' and template.language.lower() != language.lower():
                continue
            if enabled_only and not template.is_enabled:
                continue

            templates.append({
                'id': template.id,
                'name': template.name,
                'description': template.description,
                'language': template.language,
                'file_patterns': template.file_patterns,
                'priority': template.priority,
                'is_enabled': template.is_enabled,
                'criteria_count': len(template.criteria),
                'checklist_count': len(template.checklist)
            })

        return templates

    def get_template_checklist(self, template_id: str) -> List[Dict[str, Any]]:
        """Get checklist for a specific template."""
        if template_id not in self.templates:
            return []

        template = self.templates[template_id]
        return [
            {
                'id': item.id,
                'title': item.title,
                'description': item.description,
                'category': item.category,
                'severity': item.severity,
                'check_type': item.check_type,
                'validation_pattern': item.validation_pattern,
                'suggestion': item.suggestion
            }
            for item in template.checklist
        ]

    def generate_template_report(self, template_id: str, review_results: Dict[str, Any]) -> str:
        """Generate a detailed report for template application results."""
        if template_id not in self.templates:
            return "Template not found"

        template = self.templates[template_id]
        results = review_results

        report = f"# {template.name} Review Report\n\n"
        report += f"**Description:** {template.description}\n"
        report += f"**Language:** {template.language}\n"
        report += f"**File Patterns:** {', '.join(template.file_patterns)}\n\n"

        # Criteria results
        if results.get("applied_criteria"):
            report += "## Criteria Check\n\n"
            for criterion_result in results["applied_criteria"]:
                status = "✅" if criterion_result["met"] else "❌"
                report += f"{status} **{criterion_result['title']}**\n"
                report += f"   {criterion_result['details']}\n\n"

        # Checklist results
        if results.get("checklist_results"):
            report += "## Checklist Results\n\n"
            report += f"**Overall Score:** {results['overall_score']:.1f}%\n\n"

            # Group by category
            categories = {}
            for item in results["checklist_results"]:
                if item["category"] not in categories:
                    categories[item["category"]] = []
                categories[item["category"]].append(item)

            for category, items in categories.items():
                report += f"### {category.title()}\n\n"
                for item in items:
                    status = "✅" if item["passed"] else "❌"
                    report += f"{status} **{item['title']}** ({item['severity']})\n"
                    if not item["passed"] and item["suggestion"]:
                        report += f"   💡 *Suggestion:* {item['suggestion']}\n"
                    report += "\n"

        return report


# Global template manager instance
template_manager = ReviewTemplateManager()