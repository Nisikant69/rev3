# backend/rule_engine.py
"""
Custom rule engine for team-specific review policies.
Allows teams to define custom rules and validation logic for their code review process.
"""

import json
import re
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import os
from datetime import datetime

from backend.config import DATA_DIR


@dataclass
class Rule:
    """Represents a custom review rule."""
    id: str
    name: str
    description: str
    category: str  # 'security', 'performance', 'style', 'architecture', 'business'
    severity: str  # 'critical', 'high', 'medium', 'low', 'info'
    enabled: bool
    language: str  # 'all', 'python', 'javascript', etc.
    file_patterns: List[str]
    tags: List[str]
    condition: str  # JavaScript expression for rule evaluation
    action: str  # 'warn', 'error', 'suggestion', 'block'
    message: str  # User-facing message
    auto_fix: Optional[str]  # Auto-fix suggestion if applicable
    test_code: Optional[str]  # Code to test the rule against

    # Metadata
    created_at: datetime
    created_by: str
    usage_count: int = 0
    last_used: Optional[datetime] = None


@dataclass
class RuleExecutionResult:
    """Result of applying a rule to code."""
    rule_id: str
    matched: bool
        passed: bool
        severity: str
        message: str
        line_number: Optional[int] = None
        suggestion: Optional[str] = None
        auto_fix_available: bool = False
        context: Dict[str, Any] = None


@dataclass
class RuleExecutionSummary:
    """Summary of rule execution across a review."""
    total_rules_checked: int
    rules_passed: int
    rules_failed: List[RuleExecutionResult]
    rules_warned: List[RuleExecutionResult]
    rules_blocked: List[RuleExecutionResult]
    overall_score: float


class RuleEngine:
    """Advanced rule engine for custom team policies."""

    def __init__(self):
        self.rules_file = DATA_DIR / "rules" / "custom_rules.json"
        self.rules_file.parent.mkdir(parents=True, exist_ok=True)
        self.rules = {}
        self.rule_usage_stats = {}
        self.load_rules()

    def load_rules(self):
        """Load custom rules from file."""
        try:
            if self.rules_file.exists():
                with open(self.rules_file, 'r') as f:
                    data = json.load(f)

                for rule_data in data.get('rules', []):
                    rule = Rule(
                        id=rule_data['id'],
                        name=rule_data['name'],
                        description=rule_data['description'],
                        category=rule_data['category'],
                        severity=rule_data['severity'],
                        enabled=rule_data.get('enabled', True),
                        language=rule_data.get('language', 'all'),
                        file_patterns=rule_data.get('file_patterns', []),
                        tags=rule_data.get('tags', []),
                        condition=rule_data['condition'],
                        action=rule_data.get('action', 'warn'),
                        message=rule_data.get('message', ''),
                        auto_fix=rule_data.get('auto_fix'),
                        test_code=rule_data.get('test_code'),
                        created_at=datetime.fromisoformat(rule_data.get('created_at', datetime.now().isoformat())),
                        created_by=rule_data.get('created_by', 'system'),
                        usage_count=rule_data.get('usage_count', 0),
                        last_used=datetime.fromisoformat(rule_data.get('last_used', "1970-01-01T00:00:00") if rule_data.get('last_used') else None)
                    )
                    self.rules[rule.id] = rule

                # Load usage stats
                usage_stats = data.get('usage_stats', {})
                self.rule_usage_stats = usage_stats

        except Exception as e:
            print(f"Error loading rules: {e}")
            self.create_default_rules()

    def save_rules(self):
        """Save rules to file."""
        try:
            data = {
                'rules': {rule_id: asdict(rule) for rule_id, rule in self.rules.items()},
                'usage_stats': self.rule_usage_stats,
                'last_updated': datetime.now().isoformat()
            }

            with open(self.rules_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving rules: {e}")

    def save_rule_usage_stats(self):
        """Save rule usage statistics."""
        try:
            data = {
                'usage_stats': self.rule_usage_stats,
                'last_updated': datetime.now().isoformat()
            }

            with open(self.rules_file, 'w') as f:
                # Update the last_updated field in the main rules file
                temp_data = {
                    'rules': {rule_id: asdict(rule) for rule_id, rule in self.rules.items()},
                    'usage_stats': self.rule_usage_stats,
                    'last_updated': datetime.now().isoformat()
                }
                with open(self.rules_file, 'w') as f:
                    json.dump(temp_data, f, indent=2)

        except Exception as e:
            print(f"Error saving usage stats: {e}")

    def create_default_rules(self):
        """Create default custom rules."""
        default_rules = {
            "python_naming_convention": Rule(
                id="python_naming_convention",
                name="Python Naming Convention",
                description="Enforces PEP 8 naming conventions for variables, functions, and classes",
                category="style",
                severity="medium",
                enabled=True,
                language="python",
                file_patterns=["**/*.py"],
                tags=["pep8", "naming", "style"],
                condition="item.name.includes('_') && item.name != item.name.lower()",
                action="warn",
                message="Variable/function names should use snake_case per PEP 8 guidelines",
                auto_fix="item.name = item.name.lower()",
                test_code="def test_naming_convention():\n    good_name = 'test_variable'\n    bad_name = 'TestVariable'\n    return good_name == bad_name.lower()",
                created_at=datetime.now().isoformat(),
                created_by="system",
                usage_count=0
            ),
            "javascript_eslint_compliance": Rule(
                id="javascript_eslint_compliance",
                name="ESLint Compliance",
                description="Ensures code follows ESLint rules and best practices",
                category="style",
                severity="medium",
                enabled=True,
                language="javascript",
                file_patterns=["**/*.js", "**/*.ts", "**/*.jsx", "**/*.tsx"],
                tags=["eslint", "style", "code_quality"],
                condition="line.includes('var ') && !line.includes('const ')",
                action="suggestion",
                message="Use 'const' or 'let' instead of 'var'",
                auto_fix="line.replace('var ', 'const ')",
                test_code="const test_eslint():\n    // test with 'const'\n    good_code = 'let x = 1'\n    bad_code = 'var x = 1'\n    return good_code == bad_code.replace('var ', 'const ')",
                created_at=datetime.now().isoformat(),
                created_by="system",
                usage_count=0
            ),
            "security_hardcoded_secrets": Rule(
                id="security_hardcoded_secrets",
                name="Hardcoded Secrets Detection",
                description="Detects hardcoded passwords, API keys, or secrets in code",
                category="security",
                severity="critical",
                enabled=True,
                language="all",
                file_patterns=["**/*.py", "**/*.js", "**/*.ts", "**/*.java", "**/*.php"],
                tags=["security", "secrets", "authentication", "credentials"],
                condition="re.search(r'(password|secret|key|token|credential)\s*[\'\"=]\s*[\'\"]', line)",
                action="error",
                message="Hardcoded credentials detected. Remove hardcoded secrets from code.",
                auto_fix="# REMOVE: Replace with environment variables",
                test_code="# Test hardcoded secrets detection\n    good_code = 'API_KEY = os.getenv(\"API_KEY\")'\n    bad_code = 'API_KEY = \"abc123\"'\n    return good_code == bad_code.replace('\"', '\"')",
                created_at=datetime.now().isoformat(),
                created_by="system",
                usage_count=0
            ),
            "performance_n_plus_one_query": Rule(
                id="performance_n_plus_one_query",
                name="N+1 Query Prevention",
                description="Detects potential N+1 query problems in database access patterns",
                category="performance",
                severity="high",
                enabled=True,
                language="all",
                file_patterns=["**/*.py", "**/*.js", "**/*.ts", "**/*.java", "**/*.php"],
                tags=["performance", "database", "optimization"],
                condition="line.includes('.all(') and 'for' in line and 'select' in line)",
                action="warn",
                message="Potential N+1 query detected. Consider using JOIN optimization or batch operations.",
                auto_fix="# SUGGESTION: Use JOIN or batch operations",
                test_code="# Test N+1 query detection\n    # This is a simplified example\n    good_code = list(Model.objects.filter(field__in=some_list))\n    bad_code = [obj for obj in some_list]\n    return len(bad_code) > 0",
                created_at=datetime.now().isoformat(),
                created_by="system",
                usage_count=0
            ),
            "code_duplication_dry": Rule(
                id="code_duplication_dry",
                name="Code Duplication (DRY)",
                description="Detects repeated code patterns that violate DRY principle",
                category="code_quality",
                severity="medium",
                enabled=True,
                language="all",
                file_patterns=["**/*.py", "**/*.js", "**/*.ts", "**/*.java"],
                tags=["refactor", "dry", "duplication"],
                condition="len(set(line.strip()) < 3 and line.strip() in previous_lines[:10])",
                action="suggestion",
                message="Code duplication detected. Consider creating a function for repeated code.",
                auto_fix="# SUGGESTION: Extract to a reusable function",
                created_at=datetime.now().isoformat(),
                created_by="system",
                usage_count=0
            ),
            "async_await_error_handling": Rule(
                id="async_await_error_handling",
                name="Async/Await Error Handling",
                description="Ensures async functions have proper error handling",
                category="error_handling",
                severity="high",
                enabled=True,
                language="javascript",
                file_patterns=["**/*.js", "**/*.ts", "**/*.jsx", "**/*.tsx"],
                tags=["async", "error", "javascript"],
                condition="line.includes('await') and 'catch' not in line and ('try:' in line or 'finally:' in line)",
                action="error",
                message="Async function without proper error handling detected.",
                auto_fix="# SUGGESTION: Add try-catch block or use .catch()",
                test_code="# Test async error handling\n    try:\n        await async_function()\n        return True\n    except Exception as e:\n        return False",
                created_at=datetime.now().isoformat(),
                created_by="system",
                usage_count=0
            ),
            "database_connection_leak": Rule(
                id="database_connection_leak",
                name="Database Connection Leak",
                description="Detects database connections that aren't properly closed",
                category="performance",
                "severity="high",
                enabled=True,
                language="all",
                file_patterns=["**/*.py", "**/*.js", "**/*.ts", "**/*.java", "**/*.php"],
                tags=["database", "connection", "leak", "resource"],
                condition="re.search(r'(connect|open)\s*=\s*\)", line) and 'close' not in line and 'finally:' not in line",
                action="error",
                message="Database connection not properly closed. Use try-finally or context managers.",
                auto_fix="# SUGGESTION: Use context managers (with statements) or ensure proper connection cleanup",
                test_code="# Test database connection handling\n    try:\n        conn = db.connect()\n        return conn\n    finally:\n        conn.close()\n    return True",
                created_at=datetime.now().isoformat(),
                created_by="system",
                usage_count=0
            ),
            "no_unit_tests": Rule(
                id="no_unit_tests",
                name="Missing Unit Tests",
                description="Identifies files that lack unit tests",
                category="testing",
                severity="medium",
                enabled=True,
                language="all",
                file_patterns=["**/*.py", "**/*.js", "**/*.ts", "**/*.java", "**/*.php"],
                tags=["testing", "test", "coverage", "unit"],
                condition="not re.search(r'test\\s+test', line.lower())",
                action="suggestion",
                message="Consider adding unit tests to improve code coverage and reliability.",
                auto_fix="# SUGGESTION: Add comprehensive unit tests",
                test_code="# Test for unit tests\n    def test_function():\n        assert True\n    return True",
                created_at=datetime.now().isoformat(),
                created_by="system",
                usage_count=0
            )
        }

        self.rules = default_rules
        self.save_rules()

    def add_rule(self, rule_data: Dict[str, Any], rule_id: str = None) -> str:
        """Add a custom rule to the rule engine."""
        try:
            if rule_id is None:
                rule_id = f"rule_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(rule_data['name'])}"

            # Validate required fields
            required_fields = ['id', 'name', 'description', 'category', 'severity']
            for field in required_fields:
                if field not in rule_data:
                    raise ValueError(f"Missing required field: {field}")

            # Check for duplicate ID
            if rule_id in self.rules:
                raise ValueError(f"Rule with ID '{rule_id}' already exists")

            # Create rule
            rule = Rule(
                id=rule_id,
                name=rule_data['name'],
                description=rule_data['description'],
                category=rule_data['category'],
                severity=rule_data['severity'],
                enabled=rule_data.get('enabled', True),
                language=rule_data.get('language', 'all'),
                file_patterns=rule_data.get('file_patterns', []),
                tags=rule_data.get('tags', []),
                condition=rule_data.get('condition', ''),
                action=rule_data.get('action', 'warn'),
                message=rule_data.get('message', ''),
                auto_fix=rule_data.get('auto_fix'),
                test_code=rule_data.get('test_code'),
                created_at=datetime.now().isoformat(),
                created_by=rule_data.get('created_by', 'user'),
                usage_count=0
            )

            self.rules[rule_id] = rule
            self.save_rules()

            return f"Created rule '{rule.name}' with ID '{rule_id}'"

        except Exception as e:
            return f"Error adding rule: {e}"

    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> str:
        """Update an existing rule."""
        try:
            if rule_id not in self.rules:
                return f"Rule with ID '{rule_id}' not found"

            rule = self.rules[rule_id]

            # Update allowed fields
            updatable_fields = ['name', 'description', 'category', 'severity', 'enabled', 'file_patterns', 'tags', 'condition', 'action', 'message', 'auto_fix']
            for field, value in updates.items():
                if field in updatable_fields:
                    setattr(rule, field, value)

            rule.last_used = datetime.now().isoformat()
            self.save_rules()

            return f"Updated rule '{rule.name}'"

        except Exception as e:
            return f"Error updating rule: {e}"

    def delete_rule(self, rule_id: str) -> str:
        """Delete a custom rule."""
        try:
            if rule_id not in self.rules:
                return f"Rule with ID '{rule_id}' not found"

            rule_name = self.rules[rule_id].name
            del self.rules[rule_id]
            self.save_rules()

            return f"Deleted rule '{rule_name}'"

        except Exception as e:
            return f"Error deleting rule: {e}"

    def enable_rule(self, rule_id: str) -> str:
        """Enable a rule."""
        return self.update_rule(rule_id, {"enabled": True})

    def disable_rule(self, rule_id: str) -> str:
        """Disable a rule."""
        return self.update_rule(rule_id, {"enabled": False})

    def get_rules_for_file(self, filename: str, language: str = None,
                         categories: List[str] = None,
                         tags: List[str] = None) -> List[Rule]:
        """Get applicable rules for a given file."""
        applicable_rules = []

        for rule in self.rules.values():
            if not rule.enabled:
                    continue

            # Check language filter
            if language and rule.language != 'all' and rule.language.lower() != language.lower():
                continue

            # Check category filter
            if categories and rule.category not in categories:
                continue

            # Check tag filter
            if tags and not any(tag in rule.tags for tag in tags):
                continue

            # Check file patterns
            for pattern in rule.file_patterns:
                if self.matches_pattern(filename, pattern):
                    applicable_rules.append(rule)
                    break

        return applicable_rules

    def apply_rules_to_file(self, filename: str, language: str = None,
                           content: str, categories: List[str] = None,
                           tags: List[str] = None) -> RuleExecutionSummary:
        """Apply applicable rules to a file and return execution summary."""
        applicable_rules = self.get_rules_for_file(filename, language, categories, tags)

        if not applicable_rules:
            return RuleExecutionSummary(
                total_rules_checked=0,
                rules_passed=0,
                rules_failed=[],
                rules_warned=[],
                rules_blocked=[],
                overall_score=0.0
            )

        results = []

        lines = content.split('\n')
        total_rules_checked = len(applicable_rules)
        rules_passed = 0
        rules_failed = []
        rules_warned = []
        rules_blocked = []

        for rule in applicable_rules:
            rule_result = self.apply_rule_to_content(rule, content, filename)
            results.append(rule_result)

            if rule_result.passed:
                rules_passed += 1
            elif rule_result.severity == 'error':
                rules_failed.append(rule_result)
            elif rule_result.severity == 'critical':
                rules_blocked.append(rule_result)
            else:
                rules_warned.append(rule_result)

        # Calculate overall score
        total_severity = sum(
            1 for r in results if r.severity in ['critical', 'high', 'medium', 'low', 'info']
        )
        total_score = max(0, 100 - (total_severity * 10))

        return RuleExecutionSummary(
            total_rules_checked=total_rules_checked,
            rules_passed=rules_passed,
            rules_failed=rules_failed,
            rules_warned=rules_warned,
            rules_blocked=rules_blocked,
            overall_score=total_score
        )

    def apply_rule_to_content(self, rule: Rule, content: str, filename: str) -> RuleExecutionResult:
        """Apply a single rule to content and return result."""
        try:
            # Prepare context for rule evaluation
            context = {
                "filename": filename,
                "language": rule.language,
                "content_lines": content.split('\n'),
                "rule": rule
            }

            # Execute the rule's condition
            if rule.condition:
                try:
                    # Safe evaluation with limited built-in functions
                    if any(pattern in rule.condition for pattern in ['item.', 'line.', 'content.', 'message.', 'error.']):
                        # Check for item, line, content, message patterns
                        if 'item.' in rule.condition:
                            # Check for item access (item.property)
                            if '.' in rule.condition:
                                prop = rule.condition.split('.')[1]
                                if prop in ['name', 'title', 'id', 'url', 'path']):
                                    # Property access check
                                    condition_met = any(line.split('.')[-1] == prop for line in content.split('\n'))
                                    break
                    else:
                        # General string search
                        pattern = rule.condition
                        pattern_found = re.search(pattern, content, re.IGNORECASE)
                        condition_met = bool(pattern_found)

                    if not condition_met:
                        return RuleExecutionResult(
                            rule_id=rule.id,
                            matched=False,
                            passed=False,
                            severity=rule.severity,
                            message=rule.message,
                            line_number=None,
                            suggestion=rule.auto_fix,
                            auto_fix_available=bool(rule.auto_fix),
                            context=context
                        )

                    # Check for line-specific patterns
                    if rule.validation_pattern:
                        pattern_found = re.search(rule.validation_pattern, content, re.IGNORECASE)
                        if not pattern_found:
                            return RuleExecutionResult(
                                rule_id=rule.id,
                                matched=False,
                                passed=True,
                                severity=rule.severity,
                                message=rule.message,
                                line_number=None,
                                suggestion=rule.suggestion,
                                auto_fix_available=bool(rule.auto_fix),
                                context=context
                            )

                    # Default to passed if no condition
                    return RuleExecutionResult(
                        rule_id=rule.id,
                        matched=True,
                        passed=True,
                        severity=rule.severity,
                        message=rule.message,
                        line_number=None,
                        suggestion=rule.suggestion,
                        auto_fix_available=bool(rule.auto_fix),
                        context=context
                    )

        except Exception as e:
            return RuleExecutionResult(
                rule_id=rule.id,
                matched=False,
                passed=False,
                severity="error",
                message=f"Error applying rule: {str(e)}",
                line_number=None,
                suggestion=None,
                auto_fix_available=False,
                context=context
            )

    def get_rule_by_id(self, rule_id: str) -> Optional[Rule]:
        """Get a rule by its ID."""
        return self.rules.get(rule_id)

    def list_rules(self, language: str = None, category: str = None,
                   severity: str = None, tags: List[str] = None) -> List[Dict[str, Any]]:
        """List rules with optional filtering."""
        rules = []

        for rule in self.rules.values():
            # Apply filters
            if language and rule.language != 'all' and rule.language.lower() != language.lower():
                continue
            if category and rule.category != category and rule.category != category.lower():
                continue
            if severity and rule.severity != severity and rule.severity != severity.lower():
                continue
            if tags and not any(tag in rule.tags for tag in tags):
                continue

            rules.append({
                'id': rule.id,
                'name': rule.name,
                'description': rule.description,
                'category': rule.category,
                'severity': rule.severity,
                'enabled': rule.enabled,
                'language': rule.language,
                'file_patterns': rule.file_patterns,
                'tags': rule.tags,
                'priority': rule.priority,
                'usage_count': rule.usage_count
            })

        return rules

    def export_rules(self, language: str = None, category: str = None,
                    format: str = "json") -> str:
        """Export rules in specified format."""
        try:
            rules_to_export = []
            for rule in self.rules.values():
                # Apply filters
                if language and rule.language != 'all' and rule.language.lower() != language.lower():
                    continue
                if category and rule.category != category and rule.category != category.lower():
                    continue

                rule_dict = asdict(rule)
                if format == "csv":
                    # For CSV format, flatten nested structures
                    flattened = {}
                    for key, value in rule_dict.items():
                        if isinstance(value, list):
                            flattened[f"{key}_{i}"] = v for i, v in enumerate(value)
                        else:
                            flattened[key] = value
                    rules_to_export.append(flattened)
                else:
                    rules_to_export.append(rule_dict)

            # Create filename based on format
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if format == "json":
                filename = f"rules_{timestamp}.json"
            elif format == "csv":
                filename = f"rules_{timestamp}.csv"
            else:
                filename = f"rules_{timestamp}.txt"

            file_path = DATA_DIR / "rules" / filename

            with open(file_path, 'w') as f:
                if format == "json":
                    json.dump({'rules': rules_to_export}, f, indent=2)
                elif format == "csv":
                    import csv
                    writer = csv.writer(f)
                    if rules_to_export:
                        writer.writerow(['id', 'name', 'description', 'category', 'severity',
                                   'enabled', 'language', 'file_patterns',
                                   'tags', 'priority', 'usage_count',
                                   'created_at', 'created_by'])
                        for rule_dict in rules_to_export:
                            writer.writerow([
                                rule_dict.get('id'),
                                rule_dict.get('name'),
                                rule_dict.get('description'),
                                rule_dict.get('category'),
                                rule_dict.get('severity'),
                                rule_dict.get('enabled'),
                                rule_dict.get('language'),
                                ', '.join(rule_dict.get('file_patterns', [])),
                                ', '.join(rule_dict.get('tags', [])),
                                rule_dict.get('priority'),
                                rule_dict.get('usage_count'),
                                rule_dict.get('created_at'),
                                rule_dict.get('created_by')
                            ])
                else:
                    f.write(str(rule) for rule in rules_to_export)

            return f"Exported {len(rules_to_export)} rules to {filename}"

        except Exception as e:
            return f"Error exporting rules: {e}"

    def validate_rule(self, rule_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a rule definition."""
        validation_errors = []

        # Check required fields
        required_fields = ['id', 'name', 'description', 'category', 'severity']
        for field in required_fields:
            if field not in rule_data:
                validation_errors.append(f"Missing required field: {field}")

        # Validate field values
        if rule_data.get('id', '').strip() == "":
            validation_errors.append("Rule ID cannot be empty")

        if not rule_data.get('name', '').strip():
            validation_errors.append("Rule name cannot be empty")

        if rule_data.get('severity') not in ['critical', 'high', 'medium', 'low', 'info']:
            validation_errors.append("Severity must be: critical, high, medium, low, or info")

        if rule_data.get('language') not in ['all', 'python', 'javascript', 'typescript', 'java',
                                 'cpp', 'c', 'cs', 'go', 'ruby', 'php', 'rust',
                                 'swift', 'kotlin', 'scala']:
            validation_errors.append(f"Unsupported language: {rule_data.get('language')}")

        if not rule_data.get('file_patterns'):
            validation_errors.append("File patterns cannot be empty")

        # Validate file patterns
        for pattern in rule_data.get('file_patterns', []):
            if not isinstance(pattern, str):
                validation_errors.append("File patterns must be strings")
            elif not self.is_valid_glob_pattern(pattern):
                validation_errors.append(f"Invalid glob pattern: {pattern}")

        # Validate severity levels
        if rule_data.get('condition') and not self.is_valid_js_expression(rule_data['condition']):
            validation_errors.append("Invalid JavaScript condition expression")

        return {
            "valid": len(validation_errors) == 0,
            "errors": validation_errors
        }

    def is_valid_glob_pattern(self, pattern: str) -> bool:
        """Check if a pattern is a valid glob pattern."""
        try:
            # Simple glob pattern validation
            if any(char in pattern for char in ['*', '?', '[', ']', '{', '}'] and
               not pattern.count('{') == pattern.count('}')):
                return False

            # Check for balanced brackets and special characters
            if pattern.count('{') != pattern.count('}') or pattern.count('[')']) != pattern.count('(') or pattern.count(']') != pattern.count(']')):
                return False

            return True

        except Exception as e:
            return False

    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get statistics about rule usage and performance."""
        stats = {
            "total_rules": len(self.rules),
            "enabled_rules": len([r for r in self.rules.values() if r.enabled]),
            "rules_by_category": {},
            "rules_by_language": {},
            "rules_by_severity": {},
            "top_used_rules": sorted(
                [(rule.id, rule.usage_count) for rule in self.rules.values() if rule.usage_count > 0],
                key=lambda x: x[1],
                reverse=True
            ),
            "recently_used_rules": [
                rule.id for rule in self.rules.values()
                if rule.last_used and
                (datetime.now() - rule.last_used).days <= 7
            ],
            "rules_with_auto_fix": len([r for r in self.rules.values() if r.auto_fix])
        }

        # Count by category
        for rule in self.rules.values():
            category = rule.category
            stats["rules_by_category"][category] = stats["rules_by_category"].get(category, 0) + 1

        # Count by language
        for rule in self.rules.values():
            language = rule.language
            stats["rules_by_language"][language] = stats["rules_by_language"].get(language, 0) + 1

        # Count by severity
        for rule in self.rules.values():
            severity = rule.severity
            stats["rules_by_severity"][severity] = stats["rules_by_severity"].get(severity, 0) + 1

        return stats

    def import_rules(self, rules: List[str]) -> str:
        """Import rules from various sources."""
        imported_count = 0

        for rule_id in rules:
            if rule_id in self.rules:
                imported_count += 1
            else:
                    # Try to create rule from ID or data
                    # For now, we'll just track it
                    print(f"Rule '{rule_id}' not found in existing rules")

        return f"Imported {imported_count} rules"

    def get_rule_by_id_or_name(self, identifier: str) -> Optional[Rule]:
        """Get rule by ID or name."""
        # Try by ID first
        rule = self.get_rule_by_id(identifier)
        if rule:
            return rule

        # Then try by name
        for rule in self.rules.values():
            if rule.name.lower() == identifier.lower():
                return rule

        return None


# Global rule engine instance
rule_engine = RuleEngine()