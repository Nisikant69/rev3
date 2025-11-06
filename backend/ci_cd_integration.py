# backend/ci_cd_integration.py
"""
CI/CD Pipeline Integration System
Provides comprehensive integration with GitHub Actions, status checks, and build workflows.
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import re
import yaml

from config import DATA_DIR
from auth import get_installation_token
from github import Github, GithubException


class CheckStatus(Enum):
    """Status check states."""
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class CheckConclusion(Enum):
    """Check conclusion states."""
    SUCCESS = "success"
    FAILURE = "failure"
    NEUTRAL = "neutral"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    ACTION_REQUIRED = "action_required"


@dataclass
class CICDConfig:
    """CI/CD configuration settings."""
    enable_status_checks: bool = True
    enable_merge_blocking: bool = True
    critical_issue_threshold: int = 1
    warning_issue_threshold: int = 3
    auto_approve_safe_changes: bool = False
    require_human_review: bool = True
    max_review_age_hours: int = 24
    check_timeout_minutes: int = 30
    enable_build_integration: bool = True


@dataclass
class StatusCheckResult:
    """Result of a status check."""
    check_name: str
    status: CheckStatus
    conclusion: Optional[CheckConclusion]
    details_url: Optional[str]
    external_id: Optional[str]
    started_at: datetime
    completed_at: Optional[datetime]
    output: Optional[Dict[str, Any]]


@dataclass
class ReviewCriteria:
    """Review criteria for status checks."""
    max_security_issues: int = 0
    max_performance_issues: int = 2
    max_best_practice_issues: int = 5
    require_test_coverage: bool = True
    min_test_coverage_percent: int = 80
    require_documentation: bool = True
    forbid_secrets: bool = True
    require_license_check: bool = True


class CICDIntegration:
    """Main CI/CD integration manager."""

    def __init__(self, github_token: str, config: Optional[CICDConfig] = None):
        self.github = Github(github_token)
        self.config = config or CICDConfig()
        self.data_dir = DATA_DIR / "cicd"
        self.data_dir.mkdir(exist_ok=True)

        # Load configuration
        self.review_criteria = self._load_review_criteria()

    def _load_review_criteria(self) -> ReviewCriteria:
        """Load review criteria from configuration."""
        criteria_file = self.data_dir / "review_criteria.json"
        if criteria_file.exists():
            with open(criteria_file, 'r') as f:
                data = json.load(f)
            return ReviewCriteria(**data)
        return ReviewCriteria()

    def _save_review_criteria(self, criteria: ReviewCriteria):
        """Save review criteria to configuration."""
        criteria_file = self.data_dir / "review_criteria.json"
        with open(criteria_file, 'w') as f:
            json.dump(asdict(criteria), f, indent=2, default=str)

    async def create_status_check(self, repo_name: str, sha: str, check_name: str,
                                 title: str, summary: str, status: CheckStatus = CheckStatus.QUEUED,
                                 conclusion: Optional[CheckConclusion] = None,
                                 details_url: Optional[str] = None) -> Dict[str, Any]:
        """Create a GitHub status check."""
        try:
            repo = self.github.get_repo(repo_name)

            check_data = {
                "name": check_name,
                "head_sha": sha,
                "status": status.value,
                "started_at": datetime.utcnow().isoformat(),
                "output": {
                    "title": title,
                    "summary": summary
                }
            }

            if conclusion:
                check_data["conclusion"] = conclusion.value
                check_data["completed_at"] = datetime.utcnow().isoformat()

            if details_url:
                check_data["details_url"] = details_url

            # Create the check run
            check_run = repo.create_check_run(**check_data)

            return {
                "check_id": check_run.id,
                "status": check_run.status,
                "conclusion": check_run.conclusion,
                "html_url": check_run.html_url
            }

        except Exception as e:
            print(f"Error creating status check: {e}")
            return {}

    async def update_status_check(self, repo_name: str, check_id: int,
                                 status: CheckStatus, conclusion: Optional[CheckConclusion] = None,
                                 output: Optional[Dict[str, Any]] = None) -> bool:
        """Update an existing status check."""
        try:
            repo = self.github.get_repo(repo_name)
            check_run = repo.get_check_run(check_id)

            update_data = {"status": status.value}

            if conclusion:
                update_data["conclusion"] = conclusion.value
                update_data["completed_at"] = datetime.utcnow().isoformat()

            if output:
                update_data["output"] = output

            check_run.edit(**update_data)
            return True

        except Exception as e:
            print(f"Error updating status check: {e}")
            return False

    async def run_review_check(self, repo_name: str, pr_number: int,
                              review_data: Dict[str, Any]) -> StatusCheckResult:
        """Run comprehensive review check for a PR."""
        pr = self.github.get_repo(repo_name).get_pull(pr_number)
        sha = pr.head.sha

        # Create initial status check
        check_result = await self.create_status_check(
            repo_name, sha, "ai-code-review",
            "AI Code Review Analysis",
            "Analyzing pull request with AI review system..."
        )

        if not check_result:
            return StatusCheckResult(
                check_name="ai-code-review",
                status=CheckStatus.COMPLETED,
                conclusion=CheckConclusion.FAILURE,
                details_url=None,
                external_id=None,
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                output={"title": "Review Failed", "summary": "Unable to start review process"}
            )

        # Analyze review results
        issues = review_data.get("issues", [])
        security_issues = [i for i in issues if i.get("severity") == "critical"]
        performance_issues = [i for i in issues if i.get("category") == "performance"]
        best_practice_issues = [i for i in issues if i.get("category") == "best_practices"]

        # Check against criteria
        criteria_met = True
        failures = []

        if len(security_issues) > self.review_criteria.max_security_issues:
            criteria_met = False
            failures.append(f"Too many security issues: {len(security_issues)} (max: {self.review_criteria.max_security_issues})")

        if len(performance_issues) > self.review_criteria.max_performance_issues:
            criteria_met = False
            failures.append(f"Too many performance issues: {len(performance_issues)} (max: {self.review_criteria.max_performance_issues})")

        if len(best_practice_issues) > self.review_criteria.max_best_practice_issues:
            criteria_met = False
            failures.append(f"Too many best practice issues: {len(best_practice_issues)} (max: {self.review_criteria.max_best_practice_issues})")

        # Prepare output
        conclusion = CheckConclusion.SUCCESS if criteria_met else CheckConclusion.FAILURE

        output = {
            "title": "AI Code Review Complete" if criteria_met else "AI Code Review Issues Found",
            "summary": self._generate_review_summary(issues, criteria_met, failures),
            "text": self._generate_detailed_report(issues)
        }

        # Update status check
        await self.update_status_check(
            repo_name, check_result["check_id"],
            CheckStatus.COMPLETED, conclusion, output
        )

        return StatusCheckResult(
            check_name="ai-code-review",
            status=CheckStatus.COMPLETED,
            conclusion=conclusion,
            details_url=check_result.get("html_url"),
            external_id=str(check_result["check_id"]),
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            output=output
        )

    def _generate_review_summary(self, issues: List[Dict], criteria_met: bool, failures: List[str]) -> str:
        """Generate a summary of the review results."""
        if criteria_met:
            return f"✅ Review passed! Found {len(issues)} non-critical issues that can be addressed."

        summary = f"❌ Review failed with {len(issues)} issues found.\n\n"
        if failures:
            summary += "Critical failures:\n"
            for failure in failures:
                summary += f"- {failure}\n"

        return summary

    def _generate_detailed_report(self, issues: List[Dict]) -> str:
        """Generate detailed report of all issues."""
        if not issues:
            return "No issues found. Great job!"

        report = "## Detailed Issues Report\n\n"

        # Group by severity
        critical_issues = [i for i in issues if i.get("severity") == "critical"]
        warning_issues = [i for i in issues if i.get("severity") == "warning"]
        info_issues = [i for i in issues if i.get("severity") == "info"]

        if critical_issues:
            report += "### 🚨 Critical Issues\n\n"
            for issue in critical_issues:
                report += f"**{issue.get('title', 'Unknown Issue')}**\n"
                report += f"- File: `{issue.get('file', 'Unknown')}`\n"
                report += f"- Line: {issue.get('line', 'Unknown')}\n"
                report += f"- Description: {issue.get('description', 'No description')}\n\n"

        if warning_issues:
            report += "### ⚠️ Warning Issues\n\n"
            for issue in warning_issues:
                report += f"**{issue.get('title', 'Unknown Issue')}**\n"
                report += f"- File: `{issue.get('file', 'Unknown')}`\n"
                report += f"- Line: {issue.get('line', 'Unknown')}\n"
                report += f"- Description: {issue.get('description', 'No description')}\n\n"

        if info_issues:
            report += "### ℹ️ Suggestions\n\n"
            for issue in info_issues:
                report += f"**{issue.get('title', 'Unknown Issue')}**\n"
                report += f"- File: `{issue.get('file', 'Unknown')}`\n"
                report += f"- Line: {issue.get('line', 'Unknown')}\n"
                report += f"- Description: {issue.get('description', 'No description')}\n\n"

        return report

    async def check_merge_blocking(self, repo_name: str, pr_number: int,
                                  review_results: StatusCheckResult) -> Dict[str, Any]:
        """Check if PR should be blocked from merging."""
        if not self.config.enable_merge_blocking:
            return {"blocked": False, "reason": "Merge blocking is disabled"}

        pr = self.github.get_repo(repo_name).get_pull(pr_number)

        # Check if there are critical issues
        if review_results.conclusion == CheckConclusion.FAILURE:
            return {
                "blocked": True,
                "reason": "Critical issues found in review",
                "issues_count": self._count_critical_issues(review_results.output),
                "check_url": review_results.details_url
            }

        # Check if human review is required and pending
        if self.config.require_human_review:
            if not self._has_human_approval(pr):
                return {
                    "blocked": True,
                    "reason": "Human review required but not yet completed",
                    "required_approvers": 1,
                    "current_approvers": len(self._get_human_approvers(pr))
                }

        # Check if review is too old
        if self.config.max_review_age_hours > 0:
            review_age = self._get_review_age(pr)
            if review_age > timedelta(hours=self.config.max_review_age_hours):
                return {
                    "blocked": True,
                    "reason": f"Review is too old ({review_age.total_seconds() / 3600:.1f} hours)",
                    "max_age_hours": self.config.max_review_age_hours
                }

        return {"blocked": False, "reason": "All checks passed"}

    def _count_critical_issues(self, output: Optional[Dict[str, Any]]) -> int:
        """Count critical issues from review output."""
        if not output or not output.get("text"):
            return 0

        text = output["text"]
        critical_section = re.search(r"### 🚨 Critical Issues\n\n(.*?)(?=\n###|\n\n|$)", text, re.DOTALL)

        if not critical_section:
            return 0

        # Count issue entries (each starts with **)
        return len(re.findall(r"\*\*.*?\*\*", critical_section.group(1)))

    def _has_human_approval(self, pr) -> bool:
        """Check if PR has human approval."""
        reviews = pr.get_reviews()
        for review in reviews:
            if review.state == "APPROVED" and review.user.type != "Bot":
                return True
        return False

    def _get_human_approvers(self, pr) -> List[str]:
        """Get list of human approvers."""
        approvers = []
        reviews = pr.get_reviews()
        for review in reviews:
            if review.state == "APPROVED" and review.user.type != "Bot":
                approvers.append(review.user.login)
        return approvers

    def _get_review_age(self, pr) -> timedelta:
        """Get the age of the latest review."""
        reviews = pr.get_reviews()
        latest_review = None
        latest_time = None

        for review in reviews:
            if review.submitted_at and (latest_time is None or review.submitted_at > latest_time):
                latest_time = review.submitted_at
                latest_review = review

        if latest_time:
            return datetime.utcnow() - latest_time.replace(tzinfo=None)

        return datetime.utcnow() - pr.created_at.replace(tzinfo=None)

    async def run_build_integration(self, repo_name: str, branch_name: str,
                                   commit_sha: str) -> Dict[str, Any]:
        """Run review as part of build process."""
        if not self.config.enable_build_integration:
            return {"success": False, "reason": "Build integration is disabled"}

        try:
            repo = self.github.get_repo(repo_name)

            # Check if there are any commits to review
            branch = repo.get_branch(branch_name)
            if branch.commit.sha == commit_sha:
                # Create build status check
                await self.create_status_check(
                    repo_name, commit_sha, "build-review",
                    "Build Integration Review",
                    f"Running review for branch {branch_name}"
                )

                return {
                    "success": True,
                    "message": f"Build review started for {branch_name}",
                    "branch": branch_name,
                    "commit": commit_sha
                }

            return {"success": False, "reason": "Commit SHA mismatch"}

        except Exception as e:
            return {"success": False, "reason": str(e)}

    def generate_github_actions_workflow(self, repo_name: str) -> str:
        """Generate GitHub Actions workflow file for CI/CD integration."""
        workflow = {
            "name": "AI Code Review",
            "on": {
                "pull_request": {
                    "types": ["opened", "synchronize", "reopened"]
                },
                "push": {
                    "branches": ["main", "develop", "master"]
                }
            },
            "jobs": {
                "ai-review": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {
                            "name": "Checkout code",
                            "uses": "actions/checkout@v3"
                        },
                        {
                            "name": "Setup Python",
                            "uses": "actions/setup-python@v4",
                            "with": {
                                "python-version": "3.9"
                            }
                        },
                        {
                            "name": "Install dependencies",
                            "run": "pip install -r requirements.txt"
                        },
                        {
                            "name": "Run AI Review",
                            "env": {
                                "GITHUB_TOKEN": "${{ secrets.GITHUB_TOKEN }}",
                                "PR_NUMBER": "${{ github.event.number }}",
                                "REPO_NAME": "${{ github.repository }}"
                            },
                            "run": |
                                python -c "
import asyncio
import os
from ci_cd_integration import CICDIntegration

async def run_review():
    token = os.environ['GITHUB_TOKEN']
    pr_number = int(os.environ['PR_NUMBER'])
    repo_name = os.environ['REPO_NAME']

    cicd = CICDIntegration(token)

    # This would integrate with your existing review system
    # For now, we'll create a placeholder check
    await cicd.create_status_check(
        repo_name, os.environ['GITHUB_SHA'], 'ai-code-review',
        'AI Code Review', 'Review process initiated'
    )

asyncio.run(run_review())
"
                        },
                        {
                            "name": "Status Check",
                            "run": "echo 'Review completed - check GitHub status checks'"
                        }
                    ]
                }
            }
        }

        return yaml.dump(workflow, default_flow_style=False, sort_keys=False)

    def save_workflow_to_file(self, repo_name: str, workflow_content: str) -> str:
        """Save workflow content to a file."""
        workflow_dir = self.data_dir / "workflows"
        workflow_dir.mkdir(exist_ok=True)

        filename = f"{repo_name.replace('/', '_')}_ai_review.yml"
        workflow_file = workflow_dir / filename

        with open(workflow_file, 'w') as f:
            f.write(workflow_content)

        return str(workflow_file)


# Global CI/CD integration instance
_cicd_integration = None

def get_cicd_integration(github_token: str) -> CICDIntegration:
    """Get or create CI/CD integration instance."""
    global _cicd_integration
    if _cicd_integration is None:
        _cicd_integration = CICDIntegration(github_token)
    return _cicd_integration


# Utility functions for GitHub Actions integration
def create_workflow_template() -> str:
    """Create a basic GitHub Actions workflow template."""
    template = {
        "name": "AI Code Review",
        "on": {
            "pull_request": {
                "branches": ["main", "develop"]
            }
        },
        "jobs": {
            "ai-review": {
                "runs-on": "ubuntu-latest",
                "steps": [
                    {
                        "name": "AI Code Review",
                        "uses": "your-org/ai-review-action@v1",
                        "env": {
                            "GITHUB_TOKEN": "${{ secrets.GITHUB_TOKEN }}"
                        }
                    }
                ]
            }
        }
    }

    return yaml.dump(template, default_flow_style=False)