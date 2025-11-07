# backend/advanced_security.py
"""
Advanced Security Features System
Provides comprehensive security analysis including vulnerability database integration,
secret detection, license compliance, and supply chain analysis.
"""

import re
import json
import asyncio
import hashlib
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import subprocess
import xml.etree.ElementTree as ET

from config import DATA_DIR
from utils import detect_language_from_filename


class SeverityLevel(Enum):
    """Security vulnerability severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class VulnerabilityType(Enum):
    """Types of security vulnerabilities."""
    CWE = "cwe"  # Common Weakness Enumeration
    CVE = "cve"  # Common Vulnerabilities and Exposures
    OWASP = "owasp"  # OWASP Top 10
    SECRET = "secret"  # Secret/key exposure
    DEPENDENCY = "dependency"  # Third-party dependency issues
    LICENSE = "license"  # License compliance issues


@dataclass
class SecurityVulnerability:
    """Security vulnerability information."""
    id: str
    title: str
    description: str
    severity: SeverityLevel
    vulnerability_type: VulnerabilityType
    cwe_id: Optional[str] = None
    cve_id: Optional[str] = None
    owasp_category: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    remediation: Optional[str] = None
    references: List[str] = None
    discovered_at: datetime = None
    cvss_score: Optional[float] = None


@dataclass
class SecretDetection:
    """Detected secret information."""
    secret_type: str
    value: str  # Hashed or masked value
    file_path: str
    line_number: int
    context: str
    severity: SeverityLevel
    detected_at: datetime


@dataclass
class LicenseInfo:
    """License information for dependencies."""
    license_name: str
    license_type: str  # permissive, copyleft, proprietary, problematic
    risk_level: SeverityLevel
    allowed: bool
    dependencies: List[str] = None


class AdvancedSecurityAnalyzer:
    """Advanced security analysis system."""

    def __init__(self):
        self.data_dir = DATA_DIR / "security"
        self.data_dir.mkdir(exist_ok=True)

        # Load security databases
        self.cwe_database = self._load_cwe_database()
        self.owasp_database = self._load_owasp_database()
        self.secret_patterns = self._load_secret_patterns()
        self.license_database = self._load_license_database()

        # Cache for vulnerability lookups
        self.vulnerability_cache = {}

    def _load_cwe_database(self) -> Dict[str, Dict]:
        """Load CWE vulnerability database."""
        cwe_file = self.data_dir / "cwe_database.json"
        if cwe_file.exists():
            with open(cwe_file, 'r') as f:
                return json.load(f)

        # Sample CWE data (in practice, load from official CWE database)
        return {
            "CWE-79": {
                "name": "Cross-site Scripting (XSS)",
                "description": "Improper neutralization of input during web page generation",
                "severity": SeverityLevel.HIGH.value,
                "owasp_category": "A03:2021 – Injection"
            },
            "CWE-89": {
                "name": "SQL Injection",
                "description": "Improper neutralization of special elements used in SQL commands",
                "severity": SeverityLevel.CRITICAL.value,
                "owasp_category": "A03:2021 – Injection"
            },
            "CWE-20": {
                "name": "Improper Input Validation",
                "description": "Failure to validate input can lead to various vulnerabilities",
                "severity": SeverityLevel.MEDIUM.value,
                "owasp_category": "A01:2021 – Broken Access Control"
            },
            "CWE-22": {
                "name": "Path Traversal",
                "description": "Improper limitation of a pathname to a restricted directory",
                "severity": SeverityLevel.HIGH.value,
                "owasp_category": "A01:2021 – Broken Access Control"
            },
            "CWE-78": {
                "name": "OS Command Injection",
                "description": "Improper neutralization of OS commands",
                "severity": SeverityLevel.CRITICAL.value,
                "owasp_category": "A03:2021 – Injection"
            },
            "CWE-200": {
                "name": "Exposure of Sensitive Information",
                "description": "Exposure of sensitive information to an unauthorized actor",
                "severity": SeverityLevel.HIGH.value,
                "owasp_category": "A02:2021 – Cryptographic Failures"
            },
            "CWE-287": {
                "name": "Improper Authentication",
                "description": "Improper authentication mechanisms",
                "severity": SeverityLevel.HIGH.value,
                "owasp_category": "A07:2021 – Identification and Authentication Failures"
            },
            "CWE-352": {
                "name": "Cross-Site Request Forgery (CSRF)",
                "description": "Improper neutralization of request elements",
                "severity": SeverityLevel.MEDIUM.value,
                "owasp_category": "A01:2021 – Broken Access Control"
            },
            "CWE-400": {
                "name": "Uncontrolled Resource Consumption",
                "description": "Resource exhaustion or DoS vulnerabilities",
                "severity": SeverityLevel.MEDIUM.value,
                "owasp_category": "A04:2021 – Insecure Design"
            },
            "CWE-502": {
                "name": "Deserialization of Untrusted Data",
                "description": "Unsafe deserialization can lead to remote code execution",
                "severity": SeverityLevel.CRITICAL.value,
                "owasp_category": "A08:2021 – Software and Data Integrity Failures"
            }
        }

    def _load_owasp_database(self) -> Dict[str, Dict]:
        """Load OWASP Top 10 database."""
        owasp_file = self.data_dir / "owasp_database.json"
        if owasp_file.exists():
            with open(owasp_file, 'r') as f:
                return json.load(f)

        # OWASP Top 10 2021
        return {
            "A01:2021": {
                "name": "Broken Access Control",
                "description": "Restrictions on what authenticated users are allowed to do are not properly enforced",
                "cwe_ids": ["CWE-22", "CWE-287", "CWE-862", "CWE-863"],
                "examples": ["Vertical privilege escalation", "Horizontal privilege escalation", "Parameter tampering"]
            },
            "A02:2021": {
                "name": "Cryptographic Failures",
                "description": "Failures related to cryptography (or lack thereof)",
                "cwe_ids": ["CWE-200", "CWE-259", "CWE-327", "CWE-328"],
                "examples": ["Sensitive data exposure", "Weak encryption", "Missing encryption"]
            },
            "A03:2021": {
                "name": "Injection",
                "description": "User-supplied data is not validated, filtered, or sanitized by the application",
                "cwe_ids": ["CWE-78", "CWE-79", "CWE-89", "CWE-94"],
                "examples": ["SQL injection", "Cross-site scripting", "OS command injection"]
            },
            "A04:2021": {
                "name": "Insecure Design",
                "description": "Flaws in design and architecture that lead to security vulnerabilities",
                "cwe_ids": ["CWE-400", "CWE-732", "CWE-829"],
                "examples": ["Business logic flaws", "Insecure workflow", "Trust boundaries"]
            },
            "A05:2021": {
                "name": "Security Misconfiguration",
                "description": "Failure to implement security controls or misconfiguring security features",
                "cwe_ids": ["CWE-16", "CWE-611", "CWE-942"],
                "examples": ["Default credentials", "Verbose error messages", "Unnecessary services"]
            },
            "A06:2021": {
                "name": "Vulnerable and Outdated Components",
                "description": "Using components with known vulnerabilities",
                "cwe_ids": ["CWE-1104", "CWE-937", "CWE-1035"],
                "examples": ["Outdated libraries", "Vulnerable dependencies", "Missing patches"]
            },
            "A07:2021": {
                "name": "Identification and Authentication Failures",
                "description": "User identity authentication and session management is not properly implemented",
                "cwe_ids": ["CWE-255", "CWE-287", "CWE-307", "CWE-384"],
                "examples": ["Weak passwords", "Session fixation", "Missing authentication"]
            },
            "A08:2021": {
                "name": "Software and Data Integrity Failures",
                "description": "Code and data integrity verification is not properly implemented",
                "cwe_ids": ["CWE-345", "CWE-352", "CWE-502"],
                "examples": ["Insecure deserialization", "Missing integrity checks", "Code injection"]
            },
            "A09:2021": {
                "name": "Security Logging and Monitoring Failures",
                "description": "Insufficient logging and monitoring of security events",
                "cwe_ids": ["CWE-117", "CWE-223", "CWE-532"],
                "examples": ["Missing audit logs", "Insufficient monitoring", "Log injection"]
            },
            "A10:2021": {
                "name": "Server-Side Request Forgery (SSRF)",
                "description": "Server fetches a remote resource without validating the user-supplied URL",
                "cwe_ids": ["CWE-918", "CWE-942"],
                "examples": ["URL validation bypass", "Internal network access", "Cloud metadata access"]
            }
        }

    def _load_secret_patterns(self) -> Dict[str, Dict]:
        """Load secret detection patterns."""
        patterns_file = self.data_dir / "secret_patterns.json"
        if patterns_file.exists():
            with open(patterns_file, 'r') as f:
                return json.load(f)

        # Common secret patterns
        return {
            "aws_access_key": {
                "pattern": r'AKIA[0-9A-Z]{16}',
                "severity": SeverityLevel.CRITICAL.value,
                "description": "AWS Access Key ID",
                "example": "AKIAIOSFODNN7EXAMPLE"
            },
            "aws_secret_key": {
                "pattern": r'[A-Za-z0-9/+=]{40}',
                "severity": SeverityLevel.CRITICAL.value,
                "description": "AWS Secret Access Key",
                "context": r'(aws|secret|access).*key'
            },
            "github_token": {
                "pattern": r'ghp_[a-zA-Z0-9]{36}',
                "severity": SeverityLevel.CRITICAL.value,
                "description": "GitHub Personal Access Token"
            },
            "slack_token": {
                "pattern": r'xox[baprs]-[a-zA-Z0-9-]+',
                "severity": SeverityLevel.HIGH.value,
                "description": "Slack Bot/User Token"
            },
            "api_key": {
                "pattern": r'(?i)(api[_-]?key[_-]?)\s*[:=]\s*["\']?[a-zA-Z0-9_-]{16,}["\']?',
                "severity": SeverityLevel.HIGH.value,
                "description": "Generic API Key"
            },
            "private_key": {
                "pattern": r'-----BEGIN (RSA |OPENSSH |DSA |EC |PGP )?PRIVATE KEY-----',
                "severity": SeverityLevel.CRITICAL.value,
                "description": "Private Key"
            },
            "password": {
                "pattern": r'(?i)(password|passwd|pwd)\s*[:=]\s*["\']?[^\s"\']{4,}["\']?',
                "severity": SeverityLevel.MEDIUM.value,
                "description": "Password in code"
            },
            "database_url": {
                "pattern": r'(?:postgres|mysql|mongodb)://[^\s]+:[^\s]+@[^\s]+',
                "severity": SeverityLevel.HIGH.value,
                "description": "Database Connection String with credentials"
            },
            "jwt_secret": {
                "pattern": r'(?i)(jwt[_-]?secret|secret[_-]?key)\s*[:=]\s*["\']?[a-zA-Z0-9_-]{20,}["\']?',
                "severity": SeverityLevel.MEDIUM.value,
                "description": "JWT Secret Key"
            },
            "encryption_key": {
                "pattern": r'(?i)(encrypt|decrypt|cipher)[_-]?key\s*[:=]\s*["\']?[a-zA-Z0-9/+=]{16,}["\']?',
                "severity": SeverityLevel.HIGH.value,
                "description": "Encryption Key"
            }
        }

    def _load_license_database(self) -> Dict[str, Dict]:
        """Load license compliance database."""
        license_file = self.data_dir / "license_database.json"
        if license_file.exists():
            with open(license_file, 'r') as f:
                return json.load(f)

        return {
            "MIT": {
                "type": "permissive",
                "risk_level": SeverityLevel.LOW.value,
                "allowed": True,
                "description": "Permissive license with minimal restrictions"
            },
            "Apache-2.0": {
                "type": "permissive",
                "risk_level": SeverityLevel.LOW.value,
                "allowed": True,
                "description": "Permissive license with patent grant"
            },
            "BSD-3-Clause": {
                "type": "permissive",
                "risk_level": SeverityLevel.LOW.value,
                "allowed": True,
                "description": "Permissive license with 3-clause BSD"
            },
            "GPL-3.0": {
                "type": "copyleft",
                "risk_level": SeverityLevel.MEDIUM.value,
                "allowed": True,
                "description": "Strong copyleft license"
            },
            "AGPL-3.0": {
                "type": "copyleft",
                "risk_level": SeverityLevel.HIGH.value,
                "allowed": False,
                "description": "Strong copyleft with network provision"
            },
            "LGPL-2.1": {
                "type": "copyleft",
                "risk_level": SeverityLevel.MEDIUM.value,
                "allowed": True,
                "description": "Weak copyleft license"
            },
            "Proprietary": {
                "type": "proprietary",
                "risk_level": SeverityLevel.HIGH.value,
                "allowed": False,
                "description": "Commercial proprietary license"
            }
        }

    async def analyze_code_security(self, file_path: str, content: str) -> List[SecurityVulnerability]:
        """Analyze code for security vulnerabilities."""
        vulnerabilities = []
        language = detect_language_from_filename(file_path)

        # Analyze for common vulnerability patterns
        vulnerabilities.extend(self._check_injection_vulnerabilities(file_path, content, language))
        vulnerabilities.extend(self._check_authentication_issues(file_path, content, language))
        vulnerabilities.extend(self._check_authorization_issues(file_path, content, language))
        vulnerabilities.extend(self._check_crypto_issues(file_path, content, language))
        vulnerabilities.extend(self._check_input_validation(file_path, content, language))
        vulnerabilities.extend(self._check_xss_vulnerabilities(file_path, content, language))
        vulnerabilities.extend(self._check_csrf_protection(file_path, content, language))
        vulnerabilities.extend(self._check_path_traversal(file_path, content, language))
        vulnerabilities.extend(self._check_ssrf_vulnerabilities(file_path, content, language))
        vulnerabilities.extend(self._check_deserialization(file_path, content, language))

        return vulnerabilities

    def _check_injection_vulnerabilities(self, file_path: str, content: str, language: str) -> List[SecurityVulnerability]:
        """Check for injection vulnerabilities."""
        vulnerabilities = []
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            line_lower = line.lower()

            # SQL Injection patterns
            if language in ["Python", "JavaScript", "Java", "PHP", "Ruby", "Go"]:
                sql_patterns = [
                    r'execute\s*\(\s*["\'].*?\+.*?["\']',  # String concatenation in SQL
                    r'query\s*\(\s*["\'].*?\+.*?["\']',    # Similar pattern
                    r'format\s*\(\s*["\'].*?\{.*?\}.*?["\'].*?%.*?\)',  # String formatting with user input
                    r'\.query\s*\([^)]*\+',  # Node.js MySQL
                    r'prepare\s*\(\s*["\'].*?\+.*?["\']'  # Unsafe prepared statements
                ]

                for pattern in sql_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        vulnerabilities.append(SecurityVulnerability(
                            id="CWE-89",
                            title="Potential SQL Injection",
                            description="User input appears to be concatenated directly into SQL queries without proper parameterization",
                            severity=SeverityLevel.HIGH,
                            vulnerability_type=VulnerabilityType.CWE,
                            cwe_id="CWE-89",
                            owasp_category="A03:2021 – Injection",
                            file_path=file_path,
                            line_number=i,
                            code_snippet=line.strip(),
                            remediation="Use parameterized queries or prepared statements instead of string concatenation",
                            references=["https://owasp.org/www-community/attacks/SQL_Injection"],
                            discovered_at=datetime.utcnow()
                        ))

            # Command Injection patterns
            command_patterns = [
                r'system\s*\(\s*.*?\+.*?\)',  # Python system()
                r'exec\s*\(',  # Python exec()
                r'eval\s*\(',  # Python eval()
                r'subprocess\..*?\+\s*.*?shell\s*=\s*True',  # Python subprocess with shell
                r'child_process\..*?\+\s*.*?\)',  # Node.js child_process
                r'Runtime\.getRuntime\(\)\.exec\s*\(',  # Java Runtime exec
                r'exec\s*\(',  # PHP exec()
                r'shell_exec\s*\(',  # PHP shell_exec()
            ]

            for pattern in command_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    vulnerabilities.append(SecurityVulnerability(
                        id="CWE-78",
                        title="Potential Command Injection",
                        description="User input appears to be passed to system commands without proper sanitization",
                        severity=SeverityLevel.CRITICAL,
                        vulnerability_type=VulnerabilityType.CWE,
                        cwe_id="CWE-78",
                        owasp_category="A03:2021 – Injection",
                        file_path=file_path,
                        line_number=i,
                        code_snippet=line.strip(),
                        remediation="Use safe APIs with parameterized commands or validate/sanitize all user input",
                        references=["https://owasp.org/www-project-top-ten/2021/A03_2021-Injection"],
                        discovered_at=datetime.utcnow()
                    ))

        return vulnerabilities

    def _check_xss_vulnerabilities(self, file_path: str, content: str, language: str) -> List[SecurityVulnerability]:
        """Check for XSS vulnerabilities."""
        vulnerabilities = []
        lines = content.split('\n')

        if language in ["JavaScript", "TypeScript", "HTML", "Python", "PHP", "Ruby"]:
            for i, line in enumerate(lines, 1):
                line_lower = line.lower()

                # XSS patterns
                xss_patterns = [
                    r'innerHTML\s*=\s*.*?\+',  # Direct assignment with concatenation
                    r'outerHTML\s*=\s*.*?\+',  # Similar for outerHTML
                    r'document\.write\s*\(\s*.*?\+',  # document.write with user input
                    r'\.html\s*\(\s*.*?\+',  # jQuery .html() with user input
                    r'eval\s*\(\s*.*?\+',  # eval with user input
                    r'setTimeout\s*\(\s*["\'].*?\+.*?["\']',  # setTimeout with string concatenation
                    r'setInterval\s*\(\s*["\'].*?\+.*?["\']',  # setInterval with string concatenation
                ]

                for pattern in xss_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        vulnerabilities.append(SecurityVulnerability(
                            id="CWE-79",
                            title="Potential Cross-Site Scripting (XSS)",
                            description="User input appears to be directly inserted into HTML without proper escaping",
                            severity=SeverityLevel.HIGH,
                            vulnerability_type=VulnerabilityType.CWE,
                            cwe_id="CWE-79",
                            owasp_category="A03:2021 – Injection",
                            file_path=file_path,
                            line_number=i,
                            code_snippet=line.strip(),
                            remediation="Use proper HTML escaping, DOM manipulation, or template engines with auto-escaping",
                            references=["https://owasp.org/www-project-top-ten/2021/A03_2021-Injection"],
                            discovered_at=datetime.utcnow()
                        ))

        return vulnerabilities

    def _check_authentication_issues(self, file_path: str, content: str, language: str) -> List[SecurityVulnerability]:
        """Check for authentication vulnerabilities."""
        vulnerabilities = []
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            line_lower = line.lower()

            # Hardcoded credentials
            credential_patterns = [
                r'(password|passwd|pwd)\s*[:=]\s*["\'][^"\']{4,}["\']',
                r'(secret|key|token)\s*[:=]\s*["\'][^"\']{8,}["\']',
                r'(username|user)\s*[:=]\s*["\'][^"\']+["\']',
            ]

            for pattern in credential_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    vulnerabilities.append(SecurityVulnerability(
                        id="CWE-798",
                        title="Hardcoded Credentials",
                        description="Credentials are hardcoded in source code",
                        severity=SeverityLevel.HIGH,
                        vulnerability_type=VulnerabilityType.SECRET,
                        cwe_id="CWE-798",
                        owasp_category="A02:2021 – Cryptographic Failures",
                        file_path=file_path,
                        line_number=i,
                        code_snippet=line.strip(),
                        remediation="Remove hardcoded credentials and use secure configuration management",
                        references=["https://cwe.mitre.org/data/definitions/798.html"],
                        discovered_at=datetime.utcnow()
                    ))

            # Weak authentication patterns
            weak_auth_patterns = [
                r'if\s+.*password\s*==\s*["\'][^"\']+["\']',  # Plain text password comparison
                r'md5\s*\(',  # Weak hashing
                r'sha1\s*\(',  # Weak hashing
                r'hash\s*\(\s*["\'][^"\']+["\']\s*\)',  # Hashing without salt
            ]

            for pattern in weak_auth_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    vulnerabilities.append(SecurityVulnerability(
                        id="CWE-523",
                        title="Weak Authentication",
                        description="Weak authentication mechanism detected",
                        severity=SeverityLevel.MEDIUM,
                        vulnerability_type=VulnerabilityType.CWE,
                        cwe_id="CWE-523",
                        owasp_category="A07:2021 – Identification and Authentication Failures",
                        file_path=file_path,
                        line_number=i,
                        code_snippet=line.strip(),
                        remediation="Use strong hashing algorithms with salt (bcrypt, Argon2, scrypt)",
                        references=["https://owasp.org/www-project-top-ten/2021/A07_2021-Identification_and_Authentication_Failures"],
                        discovered_at=datetime.utcnow()
                    ))

        return vulnerabilities

    def _check_crypto_issues(self, file_path: str, content: str, language: str) -> List[SecurityVulnerability]:
        """Check for cryptographic vulnerabilities."""
        vulnerabilities = []
        lines = content.split('\n')

        weak_crypto_patterns = [
            r'des\s*\(',  # DES encryption
            r'rc4\s*\(',  # RC4 encryption
            r'md5\s*\(',  # MD5 hash
            r'sha1\s*\(',  # SHA-1 hash
            r'random\s*\(',  # Weak random number generation
            r'math\.random\s*\(',  # Weak random in Python
            r'crypto\.createCipher\s*\(\s*["\']des',  # Node.js DES
            r'crypto\.createCipheriv\s*\(\s*["\']des',  # Node.js DES
            r'openssl_encrypt\s*\([^)]*des[^)]*\)',  # PHP DES
        ]

        for i, line in enumerate(lines, 1):
            for pattern in weak_crypto_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    vulnerabilities.append(SecurityVulnerability(
                        id="CWE-327",
                        title="Use of Broken or Risky Cryptographic Algorithm",
                        description="Weak cryptographic algorithm detected that should not be used",
                        severity=SeverityLevel.HIGH,
                        vulnerability_type=VulnerabilityType.CWE,
                        cwe_id="CWE-327",
                        owasp_category="A02:2021 – Cryptographic Failures",
                        file_path=file_path,
                        line_number=i,
                        code_snippet=line.strip(),
                        remediation="Replace with strong cryptographic algorithms (AES-256, SHA-256+, bcrypt)",
                        references=["https://owasp.org/www-project-top-ten/2021/A02_2021-Cryptographic_Failures"],
                        discovered_at=datetime.utcnow()
                    ))

        return vulnerabilities

    def _check_input_validation(self, file_path: str, content: str, language: str) -> List[SecurityVulnerability]:
        """Check for input validation issues."""
        vulnerabilities = []
        lines = content.split('\n')

        # Look for patterns that might indicate missing input validation
        validation_patterns = [
            r'request\.(get|post|params|query)\[[^\]]+\]',  # Direct request parameter access
            r'\$_GET\[',  # PHP direct GET access
            r'\$_POST\[',  # PHP direct POST access
            r'request\.form\[',  # Python Flask direct form access
        ]

        for i, line in enumerate(lines, 1):
            for pattern in validation_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Check if there's validation in the next few lines
                    has_validation = False
                    for j in range(max(0, i-5), min(len(lines), i+5)):
                        if any(validation_keyword in lines[j].lower()
                              for validation_keyword in ['validate', 'sanitize', 'filter', 'escape', 'clean']):
                            has_validation = True
                            break

                    if not has_validation:
                        vulnerabilities.append(SecurityVulnerability(
                            id="CWE-20",
                            title="Potential Input Validation Issue",
                            description="Direct use of user input without apparent validation",
                            severity=SeverityLevel.MEDIUM,
                            vulnerability_type=VulnerabilityType.CWE,
                            cwe_id="CWE-20",
                            owasp_category="A03:2021 – Injection",
                            file_path=file_path,
                            line_number=i,
                            code_snippet=line.strip(),
                            remediation="Validate and sanitize all user input before processing",
                            references=["https://owasp.org/www-project-top-ten/2021/A03_2021-Injection"],
                            discovered_at=datetime.utcnow()
                        ))

        return vulnerabilities

    def _check_authorization_issues(self, file_path: str, content: str, language: str) -> List[SecurityVulnerability]:
        """Check for authorization vulnerabilities."""
        vulnerabilities = []
        lines = content.split('\n')

        # Look for admin/privileged operations without authorization checks
        admin_patterns = [
            r'admin\s*=\s*True',
            r'is_admin\s*=\s*True',
            r'role\s*=\s*["\']?admin["\']?',
            r'delete.*where.*id\s*=\s*request\.',  # Delete based on request ID
            r'update.*set.*where.*id\s*=\s*request\.',  # Update based on request ID
        ]

        for i, line in enumerate(lines, 1):
            for pattern in admin_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Check if there's authorization check in previous lines
                    has_auth_check = False
                    for j in range(max(0, i-10), i):
                        if any(auth_keyword in lines[j].lower()
                              for auth_keyword in ['if', 'check', 'verify', 'authorize', 'permission']):
                            has_auth_check = True
                            break

                    if not has_auth_check:
                        vulnerabilities.append(SecurityVulnerability(
                            id="CWE-285",
                            title="Potential Authorization Bypass",
                            description="Privileged operation without apparent authorization check",
                            severity=SeverityLevel.HIGH,
                            vulnerability_type=VulnerabilityType.CWE,
                            cwe_id="CWE-285",
                            owasp_category="A01:2021 – Broken Access Control",
                            file_path=file_path,
                            line_number=i,
                            code_snippet=line.strip(),
                            remediation="Implement proper authorization checks before privileged operations",
                            references=["https://owasp.org/www-project-top-ten/2021/A01_2021-Broken_Access_Control"],
                            discovered_at=datetime.utcnow()
                        ))

        return vulnerabilities

    def _check_csrf_protection(self, file_path: str, content: str, language: str) -> List[SecurityVulnerability]:
        """Check for CSRF protection."""
        vulnerabilities = []
        lines = content.split('\n')

        if language in ["Python", "JavaScript", "PHP", "Ruby"]:
            # Look for state-changing operations without CSRF protection
            state_change_patterns = [
                r'POST\s*/',
                r'PUT\s*/',
                r'DELETE\s*/',
                r'\.post\s*\(',
                r'\.put\s*\(',
                r'\.delete\s*\(',
                r'request\.method\s*==\s*["\']?POST["\']?',
            ]

            for i, line in enumerate(lines, 1):
                for pattern in state_change_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        # Check if CSRF token is being validated
                        has_csrf_check = False
                        for j in range(max(0, i-15), min(len(lines), i+5)):
                            csrf_keywords = ['csrf', 'token', 'xsrf', 'protect', 'validate']
                            if any(keyword in lines[j].lower() for keyword in csrf_keywords):
                                has_csrf_check = True
                                break

                        if not has_csrf_check:
                            vulnerabilities.append(SecurityVulnerability(
                                id="CWE-352",
                                title="Potential Cross-Site Request Forgery (CSRF)",
                                description="State-changing operation without apparent CSRF protection",
                                severity=SeverityLevel.MEDIUM,
                                vulnerability_type=VulnerabilityType.CWE,
                                cwe_id="CWE-352",
                                owasp_category="A01:2021 – Broken Access Control",
                                file_path=file_path,
                                line_number=i,
                                code_snippet=line.strip(),
                                remediation="Implement CSRF tokens for all state-changing operations",
                                references=["https://owasp.org/www-project-top-ten/2021/A01_2021-Broken_Access_Control"],
                                discovered_at=datetime.utcnow()
                            ))

        return vulnerabilities

    def _check_path_traversal(self, file_path: str, content: str, language: str) -> List[SecurityVulnerability]:
        """Check for path traversal vulnerabilities."""
        vulnerabilities = []
        lines = content.split('\n')

        path_traversal_patterns = [
            r'request\.(get|post|params)\[[^\]]+\].*open\s*\(',
            r'request\.(get|post|params)\[[^\]]+\].*read\s*\(',
            r'request\.(get|post|params)\[[^\]]+\].*file\s*\(',
            r'\$_GET\[.*\].*file\s*\(',
            r'\$_POST\[.*\].*file\s*\(',
            r'open\s*\([^)]*\+.*request\.',
            r'open\s*\([^)]*\+\s*["\'][^"\']*["\']\s*\+.*request\.',
        ]

        for i, line in enumerate(lines, 1):
            for pattern in path_traversal_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Check if path sanitization is performed
                    has_sanitization = False
                    sanitization_patterns = [
                        r'replace\(["\'][\./]["\']',
                        r'replace\(["\']\.\.["\']',
                        r'basename\s*\(',
                        r'realpath\s*\(',
                        r'path\.normalize\s*\(',
                        r'join\s*\([^)]*\+\s*["\'][^"\']*["\']',
                    ]

                    for j in range(max(0, i-5), min(len(lines), i+5)):
                        if any(re.search(pattern, lines[j], re.IGNORECASE) for pattern in sanitization_patterns):
                            has_sanitization = True
                            break

                    if not has_sanitization:
                        vulnerabilities.append(SecurityVulnerability(
                            id="CWE-22",
                            title="Potential Path Traversal",
                            description="User input used in file operations without proper path sanitization",
                            severity=SeverityLevel.HIGH,
                            vulnerability_type=VulnerabilityType.CWE,
                            cwe_id="CWE-22",
                            owasp_category="A01:2021 – Broken Access Control",
                            file_path=file_path,
                            line_number=i,
                            code_snippet=line.strip(),
                            remediation="Validate and sanitize file paths, use basename() or realpath()",
                            references=["https://owasp.org/www-project-top-ten/2021/A01_2021-Broken_Access_Control"],
                            discovered_at=datetime.utcnow()
                        ))

        return vulnerabilities

    def _check_ssrf_vulnerabilities(self, file_path: str, content: str, language: str) -> List[SecurityVulnerability]:
        """Check for SSRF vulnerabilities."""
        vulnerabilities = []
        lines = content.split('\n')

        ssrf_patterns = [
            r'urllib\.request\.urlopen\s*\(\s*request\.',
            r'requests\.(get|post|put|delete)\s*\(\s*request\.',
            r'http\.get\s*\(\s*request\.',
            r'fetch\s*\(\s*request\.',
            r'curl_exec\s*\(\s*.*request\.',
            r'file_get_contents\s*\(\s*.*request\.',
            r'socket_create\s*\([^)]*\).*request\.',
        ]

        for i, line in enumerate(lines, 1):
            for pattern in ssrf_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Check if URL validation is performed
                    has_validation = False
                    validation_patterns = [
                        r'whitelist|allowlist',
                        r'validate.*url',
                        r'filter.*url',
                        r'parse.*url',
                        r'regex.*url',
                    ]

                    for j in range(max(0, i-10), min(len(lines), i+5)):
                        if any(pattern in lines[j].lower() for pattern in validation_patterns):
                            has_validation = True
                            break

                    if not has_validation:
                        vulnerabilities.append(SecurityVulnerability(
                            id="CWE-918",
                            title="Potential Server-Side Request Forgery (SSRF)",
                            description="User-controlled URL used in server request without proper validation",
                            severity=SeverityLevel.HIGH,
                            vulnerability_type=VulnerabilityType.CWE,
                            cwe_id="CWE-918",
                            owasp_category="A10:2021 – Server-Side Request Forgery",
                            file_path=file_path,
                            line_number=i,
                            code_snippet=line.strip(),
                            remediation="Validate URLs against allowlist, avoid user-controlled URLs",
                            references=["https://owasp.org/www-project-top-ten/2021/A10_2021-Server-Side_Request_Forgery"],
                            discovered_at=datetime.utcnow()
                        ))

        return vulnerabilities

    def _check_deserialization(self, file_path: str, content: str, language: str) -> List[SecurityVulnerability]:
        """Check for unsafe deserialization."""
        vulnerabilities = []
        lines = content.split('\n')

        unsafe_deserial_patterns = [
            r'pickle\.loads?\s*\(\s*request\.',
            r'json\.loads?\s*\(\s*request\.',
            r'yaml\.loads?\s*\(\s*request\.',
            r'marshal\.loads?\s*\(\s*request\.',
            r'eval\s*\(\s*request\.',
            r'exec\s*\(\s*request\.',
        ]

        for i, line in enumerate(lines, 1):
            for pattern in unsafe_deserial_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    vulnerabilities.append(SecurityVulnerability(
                        id="CWE-502",
                        title="Potential Unsafe Deserialization",
                        description="Deserializing user-controlled data without proper validation",
                        severity=SeverityLevel.CRITICAL,
                        vulnerability_type=VulnerabilityType.CWE,
                        cwe_id="CWE-502",
                        owasp_category="A08:2021 – Software and Data Integrity Failures",
                        file_path=file_path,
                        line_number=i,
                        code_snippet=line.strip(),
                        remediation="Avoid deserializing untrusted data, use safe serialization formats",
                        references=["https://owasp.org/www-project-top-ten/2021/A08_2021-Software_and_Data_Integrity_Failures"],
                        discovered_at=datetime.utcnow()
                    ))

        return vulnerabilities

    async def detect_secrets(self, file_path: str, content: str) -> List[SecretDetection]:
        """Detect secrets and sensitive information in code."""
        secrets = []
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            for secret_type, pattern_info in self.secret_patterns.items():
                pattern = pattern_info['pattern']
                matches = re.finditer(pattern, line, re.IGNORECASE)

                for match in matches:
                    # Mask the secret value
                    secret_value = match.group()
                    masked_value = self._mask_secret(secret_value)

                    secrets.append(SecretDetection(
                        secret_type=secret_type,
                        value=masked_value,
                        file_path=file_path,
                        line_number=i,
                        context=line.strip(),
                        severity=SeverityLevel(pattern_info['severity']),
                        detected_at=datetime.utcnow()
                    ))

        return secrets

    def _mask_secret(self, secret: str) -> str:
        """Mask a secret value for logging."""
        if len(secret) <= 8:
            return "*" * len(secret)
        return secret[:4] + "*" * (len(secret) - 8) + secret[-4:]

    async def check_dependency_vulnerabilities(self, dependency_file: str) -> List[Dict[str, Any]]:
        """Check for vulnerabilities in dependencies."""
        vulnerabilities = []

        try:
            if dependency_file.endswith('package.json'):
                vulnerabilities = await self._check_npm_vulnerabilities(dependency_file)
            elif dependency_file.endswith('requirements.txt') or dependency_file.endswith('Pipfile'):
                vulnerabilities = await self._check_python_vulnerabilities(dependency_file)
            elif dependency_file.endswith('pom.xml'):
                vulnerabilities = await self._check_maven_vulnerabilities(dependency_file)
            elif dependency_file.endswith('Gemfile'):
                vulnerabilities = await self._check_ruby_vulnerabilities(dependency_file)

        except Exception as e:
            print(f"Error checking {dependency_file}: {e}")

        return vulnerabilities

    async def _check_npm_vulnerabilities(self, package_file: str) -> List[Dict[str, Any]]:
        """Check npm package.json for vulnerabilities."""
        vulnerabilities = []

        try:
            with open(package_file, 'r') as f:
                package_data = json.load(f)

            dependencies = {}
            if 'dependencies' in package_data:
                dependencies.update(package_data['dependencies'])
            if 'devDependencies' in package_data:
                dependencies.update(package_data['devDependencies'])

            # In a real implementation, you would query npm audit API or OSV
            # For demo, return sample vulnerabilities
            for dep_name, version in dependencies.items():
                if 'lodash' in dep_name.lower() and '4.' in version:
                    vulnerabilities.append({
                        'dependency': dep_name,
                        'version': version,
                        'severity': 'high',
                        'cve': 'CVE-2021-23337',
                        'title': 'Prototype Pollution in lodash',
                        'description': 'Prototype pollution vulnerability allows modification of Object.prototype'
                    })

        except Exception as e:
            print(f"Error reading package.json: {e}")

        return vulnerabilities

    async def _check_python_vulnerabilities(self, requirements_file: str) -> List[Dict[str, Any]]:
        """Check Python requirements for vulnerabilities."""
        vulnerabilities = []

        try:
            with open(requirements_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split('==')
                    if len(parts) == 2:
                        dep_name, version = parts[0].strip(), parts[1].strip()

                        # Sample vulnerability checks
                        if 'requests' in dep_name.lower() and version.startswith('2.'):
                            vulnerabilities.append({
                                'dependency': dep_name,
                                'version': version,
                                'severity': 'medium',
                                'title': 'Potential security issue in requests',
                                'description': 'Older versions may have security vulnerabilities'
                            })

        except Exception as e:
            print(f"Error reading requirements file: {e}")

        return vulnerabilities

    async def _check_maven_vulnerabilities(self, pom_file: str) -> List[Dict[str, Any]]:
        """Check Maven pom.xml for vulnerabilities."""
        vulnerabilities = []

        try:
            tree = ET.parse(pom_file)
            root = tree.getroot()

            # Handle namespace
            namespace = {'maven': 'http://maven.apache.org/POM/4.0.0'}
            if root.tag.startswith('{'):
                namespace['maven'] = root.tag.split('}')[0][1:]

            dependencies = root.find('.//maven:dependencies', namespace)
            if dependencies is not None:
                for dep in dependencies.findall('maven:dependency', namespace):
                    group_id = dep.find('maven:groupId', namespace)
                    artifact_id = dep.find('maven:artifactId', namespace)
                    version = dep.find('maven:version', namespace)

                    if group_id is not None and artifact_id is not None:
                        dep_name = f"{group_id.text}:{artifact_id.text}"
                        dep_version = version.text if version is not None else "unknown"

                        # Sample vulnerability check
                        if 'log4j' in dep_name.lower() and dep_version.startswith('2.'):
                            vulnerabilities.append({
                                'dependency': dep_name,
                                'version': dep_version,
                                'severity': 'critical',
                                'cve': 'CVE-2021-44228',
                                'title': 'Log4Shell Remote Code Execution',
                                'description': 'Critical RCE vulnerability in Log4j'
                            })

        except Exception as e:
            print(f"Error parsing pom.xml: {e}")

        return vulnerabilities

    async def _check_ruby_vulnerabilities(self, gemfile: str) -> List[Dict[str, Any]]:
        """Check Ruby Gemfile for vulnerabilities."""
        vulnerabilities = []

        try:
            with open(gemfile, 'r') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if line.startswith('gem '):
                    # Extract gem name and version
                    parts = line.split(' ', 2)
                    if len(parts) >= 2:
                        gem_name = parts[1].strip('"\'')
                        version = parts[2].strip('"\'') if len(parts) > 2 else "unknown"

                        # Sample vulnerability check
                        if 'rails' in gem_name.lower() and version.startswith('5.'):
                            vulnerabilities.append({
                                'dependency': gem_name,
                                'version': version,
                                'severity': 'medium',
                                'title': 'Potential security issues in Rails 5.x',
                                'description': 'Older Rails versions may have security vulnerabilities'
                            })

        except Exception as e:
            print(f"Error reading Gemfile: {e}")

        return vulnerabilities

    async def check_license_compliance(self, dependency_file: str) -> List[LicenseInfo]:
        """Check license compliance for dependencies."""
        license_issues = []

        try:
            if dependency_file.endswith('package.json'):
                license_issues = await self._check_npm_licenses(dependency_file)
            elif dependency_file.endswith('requirements.txt'):
                license_issues = await self._check_python_licenses(dependency_file)
            elif dependency_file.endswith('pom.xml'):
                license_issues = await self._check_maven_licenses(dependency_file)

        except Exception as e:
            print(f"Error checking licenses in {dependency_file}: {e}")

        return license_issues

    async def _check_npm_licenses(self, package_file: str) -> List[LicenseInfo]:
        """Check npm package licenses."""
        license_issues = []

        try:
            with open(package_file, 'r') as f:
                package_data = json.load(f)

            dependencies = {}
            if 'dependencies' in package_data:
                dependencies.update(package_data['dependencies'])
            if 'devDependencies' in package_data:
                dependencies.update(package_data['devDependencies'])

            # In a real implementation, you would query npm registry or use tools like license-checker
            # For demo, return sample license issues
            for dep_name in dependencies.keys():
                if 'react' in dep_name.lower():
                    license_info = LicenseInfo(
                        license_name="MIT",
                        license_type="permissive",
                        risk_level=SeverityLevel.LOW,
                        allowed=True,
                        dependencies=[dep_name]
                    )
                elif 'agpl' in dep_name.lower():
                    license_info = LicenseInfo(
                        license_name="AGPL-3.0",
                        license_type="copyleft",
                        risk_level=SeverityLevel.HIGH,
                        allowed=False,
                        dependencies=[dep_name]
                    )
                    license_issues.append(license_info)

        except Exception as e:
            print(f"Error checking npm licenses: {e}")

        return license_issues

    async def _check_python_licenses(self, requirements_file: str) -> List[LicenseInfo]:
        """Check Python package licenses."""
        license_issues = []

        try:
            with open(requirements_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    dep_name = line.split('==')[0].strip()

                    # Sample license mapping (in practice, use tools like pip-licenses)
                    if 'django' in dep_name.lower():
                        license_info = LicenseInfo(
                            license_name="BSD-3-Clause",
                            license_type="permissive",
                            risk_level=SeverityLevel.LOW,
                            allowed=True,
                            dependencies=[dep_name]
                        )
                    elif 'mysql' in dep_name.lower():
                        license_info = LicenseInfo(
                            license_name="GPL-2.0",
                            license_type="copyleft",
                            risk_level=SeverityLevel.MEDIUM,
                            allowed=True,
                            dependencies=[dep_name]
                        )
                        license_issues.append(license_info)

        except Exception as e:
            print(f"Error checking Python licenses: {e}")

        return license_issues

    async def _check_maven_licenses(self, pom_file: str) -> List[LicenseInfo]:
        """Check Maven dependency licenses."""
        license_issues = []

        try:
            tree = ET.parse(pom_file)
            root = tree.getroot()

            namespace = {'maven': 'http://maven.apache.org/POM/4.0.0'}
            if root.tag.startswith('{'):
                namespace['maven'] = root.tag.split('}')[0][1:]

            dependencies = root.find('.//maven:dependencies', namespace)
            if dependencies is not None:
                for dep in dependencies.findall('maven:dependency', namespace):
                    group_id = dep.find('maven:groupId', namespace)
                    artifact_id = dep.find('maven:artifactId', namespace)

                    if group_id is not None and artifact_id is not None:
                        dep_name = f"{group_id.text}:{artifact_id.text}"

                        # Sample license mapping
                        if 'spring' in dep_name.lower():
                            license_info = LicenseInfo(
                                license_name="Apache-2.0",
                                license_type="permissive",
                                risk_level=SeverityLevel.LOW,
                                allowed=True,
                                dependencies=[dep_name]
                            )

        except Exception as e:
            print(f"Error checking Maven licenses: {e}")

        return license_issues

    def generate_security_report(self, vulnerabilities: List[SecurityVulnerability],
                                secrets: List[SecretDetection],
                                dependency_vulns: List[Dict[str, Any]],
                                license_issues: List[LicenseInfo]) -> Dict[str, Any]:
        """Generate comprehensive security report."""

        # Count by severity
        severity_counts = {
            SeverityLevel.CRITICAL.value: 0,
            SeverityLevel.HIGH.value: 0,
            SeverityLevel.MEDIUM.value: 0,
            SeverityLevel.LOW.value: 0,
            SeverityLevel.INFO.value: 0
        }

        for vuln in vulnerabilities:
            severity_counts[vuln.severity.value] += 1

        for secret in secrets:
            severity_counts[secret.severity.value] += 1

        for dep_vuln in dependency_vulns:
            severity_counts[dep_vuln['severity']] += 1

        for license_issue in license_issues:
            severity_counts[license_issue.risk_level.value] += 1

        # Calculate risk score
        risk_score = (
            severity_counts[SeverityLevel.CRITICAL.value] * 10 +
            severity_counts[SeverityLevel.HIGH.value] * 5 +
            severity_counts[SeverityLevel.MEDIUM.value] * 2 +
            severity_counts[SeverityLevel.LOW.value] * 1
        )

        return {
            "summary": {
                "total_vulnerabilities": len(vulnerabilities) + len(secrets) + len(dependency_vulns) + len(license_issues),
                "severity_breakdown": severity_counts,
                "risk_score": risk_score,
                "risk_level": self._calculate_risk_level(risk_score)
            },
            "vulnerabilities": [asdict(v) for v in vulnerabilities],
            "secrets": [asdict(s) for s in secrets],
            "dependency_vulnerabilities": dependency_vulns,
            "license_issues": [asdict(l) for l in license_issues],
            "recommendations": self._generate_recommendations(vulnerabilities, secrets, dependency_vulns, license_issues),
            "generated_at": datetime.utcnow().isoformat()
        }

    def _calculate_risk_level(self, score: int) -> str:
        """Calculate overall risk level."""
        if score >= 50:
            return "Critical"
        elif score >= 25:
            return "High"
        elif score >= 10:
            return "Medium"
        elif score >= 5:
            return "Low"
        else:
            return "Info"

    def _generate_recommendations(self, vulnerabilities: List[SecurityVulnerability],
                                   secrets: List[SecretDetection],
                                   dependency_vulns: List[Dict[str, Any]],
                                   license_issues: List[LicenseInfo]) -> List[str]:
        """Generate security recommendations."""
        recommendations = []

        if vulnerabilities:
            recommendations.append("🔐 Fix code-level security vulnerabilities immediately")
            recommendations.append("🛡️ Implement security testing in CI/CD pipeline")

        if secrets:
            recommendations.append("🔑 Remove all hardcoded secrets and use secure configuration management")
            recommendations.append("🔄 Rotate any exposed credentials immediately")

        if dependency_vulns:
            recommendations.append("📦 Update vulnerable dependencies to latest secure versions")
            recommendations.append("🔍 Implement dependency vulnerability scanning")

        if license_issues:
            recommendations.append("⚖️ Review license compliance with your legal team")
            recommendations.append("📋 Consider alternative libraries for problematic licenses")

        return recommendations


# Global security analyzer instance
_security_analyzer = None

def get_security_analyzer() -> AdvancedSecurityAnalyzer:
    """Get or create security analyzer instance."""
    global _security_analyzer
    if _security_analyzer is None:
        _security_analyzer = AdvancedSecurityAnalyzer()
    return _security_analyzer