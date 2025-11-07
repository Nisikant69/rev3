# backend/team_collaboration.py
"""
Team Collaboration System
Provides comprehensive team collaboration features including review assignment,
comments, approvals, and escalation workflows.
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import random

from config import DATA_DIR
from auth import get_installation_token
from github import Github, GithubException


class AssignmentStrategy(Enum):
    """Review assignment strategies."""
    ROUND_ROBIN = "round_robin"
    EXPERTISE_BASED = "expertise_based"
    WORKLOAD_BALANCED = "workload_balanced"
    RANDOM = "random"
    MANUAL = "manual"


class ReviewStatus(Enum):
    """Review status states."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"
    COMMENTED = "commented"
    ESCALATED = "escalated"


class Priority(Enum):
    """Issue priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TeamMember:
    """Team member information."""
    username: str
    email: str
    expertise_areas: List[str]
    max_reviews_per_day: int = 5
    current_review_count: int = 0
    timezone: str = "UTC"
    is_available: bool = True
    seniority_level: int = 1  # 1=Junior, 2=Mid, 3=Senior, 4=Lead
    github_user_id: Optional[int] = None


@dataclass
class ReviewAssignment:
    """Review assignment details."""
    pr_number: int
    repo_name: str
    assigned_to: str
    assigned_by: str
    assigned_at: datetime
    status: ReviewStatus
    priority: Priority
    deadline: Optional[datetime] = None
    notes: Optional[str] = None
    files_to_review: List[str] = None


@dataclass
class ReviewComment:
    """Review comment from team member."""
    pr_number: int
    repo_name: str
    author: str
    comment: str
    timestamp: datetime
    line_number: Optional[int] = None
    file_path: Optional[str] = None
    is_approval: bool = False
    is_request_changes: bool = False
    parent_comment_id: Optional[str] = None
    reactions: Dict[str, List[str]] = None


@dataclass
class EscalationRule:
    """Escalation rule configuration."""
    name: str
    condition: str  # JavaScript expression
    trigger_after_hours: int
    escalate_to: List[str]
    priority_threshold: Priority
    auto_escalate: bool = True


class TeamCollaboration:
    """Main team collaboration manager."""

    def __init__(self, github_token: str, repo_name: str):
        self.github = Github(github_token)
        self.repo_name = repo_name
        self.repo = self.github.get_repo(repo_name)
        self.data_dir = DATA_DIR / "team_collaboration" / repo_name.replace("/", "_")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Load team configuration
        self.team_members = self._load_team_members()
        self.assignments = self._load_assignments()
        self.comments = self._load_comments()
        self.escalation_rules = self._load_escalation_rules()

        # Assignment tracking
        self.assignment_counters = {}
        self._initialize_assignment_counters()

    def _initialize_assignment_counters(self):
        """Initialize assignment counters for round-robin."""
        for member in self.team_members:
            self.assignment_counters[member.username] = 0

    def _load_team_members(self) -> List[TeamMember]:
        """Load team members from configuration."""
        members_file = self.data_dir / "team_members.json"
        if members_file.exists():
            with open(members_file, 'r') as f:
                data = json.load(f)
            return [TeamMember(**member) for member in data]

        # Default team members
        return [
            TeamMember(
                username="senior-developer",
                email="senior@company.com",
                expertise_areas=["security", "architecture", "performance"],
                seniority_level=4,
                max_reviews_per_day=3
            ),
            TeamMember(
                username="mid-developer-1",
                email="mid1@company.com",
                expertise_areas=["frontend", "testing"],
                seniority_level=2,
                max_reviews_per_day=5
            ),
            TeamMember(
                username="mid-developer-2",
                email="mid2@company.com",
                expertise_areas=["backend", "database"],
                seniority_level=2,
                max_reviews_per_day=5
            ),
            TeamMember(
                username="junior-developer",
                email="junior@company.com",
                expertise_areas=["documentation", "minor-features"],
                seniority_level=1,
                max_reviews_per_day=8
            )
        ]

    def _load_assignments(self) -> List[ReviewAssignment]:
        """Load review assignments from storage."""
        assignments_file = self.data_dir / "assignments.json"
        if assignments_file.exists():
            with open(assignments_file, 'r') as f:
                data = json.load(f)
            return [ReviewAssignment(**assignment) for assignment in data]
        return []

    def _load_comments(self) -> List[ReviewComment]:
        """Load review comments from storage."""
        comments_file = self.data_dir / "comments.json"
        if comments_file.exists():
            with open(comments_file, 'r') as f:
                data = json.load(f)
            return [ReviewComment(**comment) for comment in data]
        return []

    def _load_escalation_rules(self) -> List[EscalationRule]:
        """Load escalation rules from configuration."""
        rules_file = self.data_dir / "escalation_rules.json"
        if rules_file.exists():
            with open(rules_file, 'r') as f:
                data = json.load(f)
            return [EscalationRule(**rule) for rule in data]

        # Default escalation rules
        return [
            EscalationRule(
                name="Critical Security Issues",
                condition="issue.severity === 'critical' && issue.category === 'security'",
                trigger_after_hours=2,
                escalate_to=["senior-developer", "security-lead"],
                priority_threshold=Priority.CRITICAL
            ),
            EscalationRule(
                name="High Priority Stale Reviews",
                condition="review.age_hours > 24 && review.status === 'pending'",
                trigger_after_hours=24,
                escalate_to=["team-lead"],
                priority_threshold=Priority.HIGH
            ),
            EscalationRule(
                name="Performance Regression",
                condition="issue.category === 'performance' && issue.impact === 'high'",
                trigger_after_hours=4,
                escalate_to=["performance-lead", "senior-developer"],
                priority_threshold=Priority.HIGH
            )
        ]

    def _save_team_members(self):
        """Save team members to configuration."""
        members_file = self.data_dir / "team_members.json"
        with open(members_file, 'w') as f:
            json.dump([asdict(member) for member in self.team_members], f, indent=2, default=str)

    def _save_assignments(self):
        """Save assignments to storage."""
        assignments_file = self.data_dir / "assignments.json"
        with open(assignments_file, 'w') as f:
            json.dump([asdict(assignment) for assignment in self.assignments], f, indent=2, default=str)

    def _save_comments(self):
        """Save comments to storage."""
        comments_file = self.data_dir / "comments.json"
        with open(comments_file, 'w') as f:
            json.dump([asdict(comment) for comment in self.comments], f, indent=2, default=str)

    async def assign_reviewer(self, pr_number: int, strategy: AssignmentStrategy = AssignmentStrategy.EXPERTISE_BASED,
                             files: List[str] = None, priority: Priority = Priority.MEDIUM,
                             requested_reviewer: Optional[str] = None, deadline_hours: int = 24) -> Dict[str, Any]:
        """Assign a reviewer to a pull request."""
        try:
            pr = self.repo.get_pull(pr_number)

            # Check if already assigned
            existing_assignment = self._get_assignment_for_pr(pr_number)
            if existing_assignment:
                return {
                    "success": False,
                    "message": f"PR #{pr_number} already assigned to {existing_assignment.assigned_to}",
                    "assignment": asdict(existing_assignment)
                }

            # Get available reviewers
            available_reviewers = self._get_available_reviewers()
            if not available_reviewers:
                return {
                    "success": False,
                    "message": "No available reviewers at the moment"
                }

            # Select reviewer based on strategy
            selected_reviewer = await self._select_reviewer(
                available_reviewers, pr, strategy, files, requested_reviewer
            )

            if not selected_reviewer:
                return {
                    "success": False,
                    "message": "Could not find suitable reviewer"
                }

            # Create assignment
            deadline = datetime.utcnow() + timedelta(hours=deadline_hours)
            assignment = ReviewAssignment(
                pr_number=pr_number,
                repo_name=self.repo_name,
                assigned_to=selected_reviewer.username,
                assigned_by="ai-review-bot",
                assigned_at=datetime.utcnow(),
                status=ReviewStatus.PENDING,
                priority=priority,
                deadline=deadline,
                files_to_review=files or [],
                notes=f"Assigned using {strategy.value} strategy"
            )

            self.assignments.append(assignment)
            self._save_assignments()

            # Update reviewer's current count
            selected_reviewer.current_review_count += 1

            # Post assignment comment on PR
            assignment_comment = f"""
🤖 **Review Assignment**

**Assigned to:** @{selected_reviewer.username}
**Strategy:** {strategy.value}
**Deadline:** {deadline.strftime('%Y-%m-%d %H:%M UTC')}
**Priority:** {priority.value}

**Files to review:** {', '.join(files or pr.get_files())[:5]}{'...' if len(pr.get_files()) > 5 else ''}

Please review this pull request and provide your feedback by the deadline.
"""

            pr.create_issue_comment(assignment_comment)

            # Request reviewer on GitHub
            try:
                pr.create_review_request(reviewers=[selected_reviewer.username])
            except GithubException as e:
                print(f"Could not create GitHub review request: {e}")

            return {
                "success": True,
                "message": f"Successfully assigned PR #{pr_number} to {selected_reviewer.username}",
                "assignment": asdict(assignment),
                "reviewer": asdict(selected_reviewer)
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Error assigning reviewer: {str(e)}"
            }

    async def _select_reviewer(self, available_reviewers: List[TeamMember], pr,
                               strategy: AssignmentStrategy, files: List[str] = None,
                               requested_reviewer: Optional[str] = None) -> Optional[TeamMember]:
        """Select a reviewer based on the specified strategy."""
        if requested_reviewer:
            # Manual assignment
            for reviewer in available_reviewers:
                if reviewer.username == requested_reviewer:
                    return reviewer
            return None

        if strategy == AssignmentStrategy.RANDOM:
            return random.choice(available_reviewers)

        elif strategy == AssignmentStrategy.ROUND_ROBIN:
            # Find reviewer with minimum assignments
            min_assignments = min(self.assignment_counts[reviewer.username] for reviewer in available_reviewers)
            candidates = [r for r in available_reviewers if self.assignment_counts[r.username] == min_assignments]
            selected = random.choice(candidates)
            self.assignment_counts[selected.username] += 1
            return selected

        elif strategy == AssignmentStrategy.WORKLOAD_BALANCED:
            # Select reviewer with least current reviews
            return min(available_reviewers, key=lambda r: r.current_review_count)

        elif strategy == AssignmentStrategy.EXPERTISE_BASED:
            # Analyze PR content and match with expertise
            if not files:
                files = [f.filename for f in pr.get_files()]

            required_expertise = self._analyze_required_expertise(files, pr.title, pr.body)

            # Score reviewers based on expertise match
            scored_reviewers = []
            for reviewer in available_reviewers:
                score = self._calculate_expertise_score(reviewer, required_expertise)
                scored_reviewers.append((reviewer, score))

            # Sort by score and select top scorer
            scored_reviewers.sort(key=lambda x: x[1], reverse=True)
            return scored_reviewers[0][0] if scored_reviewers else None

        return random.choice(available_reviewers)

    def _analyze_required_expertise(self, files: List[str], pr_title: str, pr_body: str) -> List[str]:
        """Analyze PR content to determine required expertise."""
        expertise_needed = set()
        content = (pr_title + " " + (pr_body or "")).lower()
        file_content = " ".join(files).lower()

        # Security indicators
        if any(keyword in content + file_content for keyword in ["security", "auth", "vulnerability", "encryption", "token"]):
            expertise_needed.add("security")

        # Performance indicators
        if any(keyword in content + file_content for keyword in ["performance", "optimize", "speed", "memory", "cache"]):
            expertise_needed.add("performance")

        # Frontend indicators
        if any(keyword in file_content for keyword in [".js", ".jsx", ".ts", ".tsx", ".css", ".html", "react", "vue", "angular"]):
            expertise_needed.add("frontend")

        # Backend indicators
        if any(keyword in file_content for keyword in [".py", ".java", ".go", ".rs", ".cpp", ".sql", "api"]):
            expertise_needed.add("backend")

        # Database indicators
        if any(keyword in content + file_content for keyword in ["database", "schema", "migration", "query", "sql"]):
            expertise_needed.add("database")

        # Testing indicators
        if any(keyword in file_content for keyword in ["test", "spec", "mock", "jest", "pytest"]):
            expertise_needed.add("testing")

        # DevOps/CI indicators
        if any(keyword in file_content for keyword in ["docker", "kubernetes", "ci", "cd", "deploy", "pipeline"]):
            expertise_needed.add("devops")

        return list(expertise_needed)

    def _calculate_expertise_score(self, reviewer: TeamMember, required_expertise: List[str]) -> float:
        """Calculate expertise match score for a reviewer."""
        if not required_expertise:
            return 0.5  # Neutral score

        matching_expertise = set(reviewer.expertise_areas) & set(required_expertise)
        score = len(matching_expertise) / len(required_expertise)

        # Boost score for senior developers
        score += (reviewer.seniority_level - 1) * 0.1

        # Reduce score if reviewer is busy
        if reviewer.current_review_count >= reviewer.max_reviews_per_day:
            score *= 0.5

        return min(score, 1.0)

    def _get_available_reviewers(self) -> List[TeamMember]:
        """Get list of available reviewers."""
        available = []
        now = datetime.utcnow()

        for member in self.team_members:
            if not member.is_available:
                continue

            # Check if reviewer is at capacity
            if member.current_review_count >= member.max_reviews_per_day:
                continue

            # Check if reviewer has overdue assignments
            overdue_count = sum(1 for assignment in self.assignments
                              if assignment.assigned_to == member.username
                              and assignment.status == ReviewStatus.PENDING
                              and assignment.deadline < now)

            if overdue_count > 0:
                continue

            available.append(member)

        return available

    def _get_assignment_for_pr(self, pr_number: int) -> Optional[ReviewAssignment]:
        """Get assignment for a specific PR."""
        for assignment in self.assignments:
            if assignment.pr_number == pr_number and assignment.status in [ReviewStatus.PENDING, ReviewStatus.IN_PROGRESS]:
                return assignment
        return None

    async def add_review_comment(self, pr_number: int, author: str, comment: str,
                                 line_number: Optional[int] = None, file_path: Optional[str] = None,
                                 is_approval: bool = False, is_request_changes: bool = False,
                                 parent_comment_id: Optional[str] = None) -> Dict[str, Any]:
        """Add a review comment to a PR."""
        try:
            pr = self.repo.get_pull(pr_number)

            # Create comment object
            review_comment = ReviewComment(
                pr_number=pr_number,
                repo_name=self.repo_name,
                author=author,
                comment=comment,
                timestamp=datetime.utcnow(),
                line_number=line_number,
                file_path=file_path,
                is_approval=is_approval,
                is_request_changes=is_request_changes,
                parent_comment_id=parent_comment_id,
                reactions={}
            )

            # Post comment to GitHub
            if line_number and file_path:
                # Line-specific comment
                commit = pr.head.sha
                pr.create_review_comment(
                    body=comment,
                    commit=commit,
                    path=file_path,
                    position=line_number
                )
            else:
                # General PR comment
                pr.create_issue_comment(comment)

            # Save comment
            self.comments.append(review_comment)
            self._save_comments()

            # Update assignment status if applicable
            assignment = self._get_assignment_for_pr(pr_number)
            if assignment and assignment.assigned_to == author:
                if is_approval:
                    assignment.status = ReviewStatus.APPROVED
                elif is_request_changes:
                    assignment.status = ReviewStatus.CHANGES_REQUESTED
                else:
                    assignment.status = ReviewStatus.COMMENTED

                self._save_assignments()

            return {
                "success": True,
                "message": "Comment added successfully",
                "comment": asdict(review_comment)
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Error adding comment: {str(e)}"
            }

    async def approve_suggestion(self, pr_number: int, reviewer: str, suggestion_id: str,
                                 comment: Optional[str] = None) -> Dict[str, Any]:
        """Approve an AI suggestion."""
        approval_text = f"""
✅ **Suggestion Approved**

**Reviewer:** @{reviewer}
**Suggestion ID:** {suggestion_id}
**Approved at:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

{comment or "This suggestion has been reviewed and approved."}
"""

        return await self.add_review_comment(pr_number, reviewer, approval_text)

    async def reject_suggestion(self, pr_number: int, reviewer: str, suggestion_id: str,
                                reason: str) -> Dict[str, Any]:
        """Reject an AI suggestion with reason."""
        rejection_text = f"""
❌ **Suggestion Rejected**

**Reviewer:** @{reviewer}
**Suggestion ID:** {suggestion_id}
**Rejected at:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

**Reason:** {reason}
"""

        return await self.add_review_comment(pr_number, reviewer, rejection_text)

    async def check_escalations(self) -> List[Dict[str, Any]]:
        """Check for reviews that need escalation."""
        escalations = []
        now = datetime.utcnow()

        for assignment in self.assignments:
            if assignment.status != ReviewStatus.PENDING:
                continue

            # Check deadline
            if assignment.deadline and now > assignment.deadline:
                escalation = {
                    "type": "deadline_exceeded",
                    "assignment": asdict(assignment),
                    "overdue_hours": (now - assignment.deadline).total_seconds() / 3600,
                    "suggested_escalation": ["team-lead", "senior-developer"]
                }
                escalations.append(escalation)
                continue

            # Check escalation rules
            time_until_deadline = (assignment.deadline - now) if assignment.deadline else timedelta(hours=24)

            for rule in self.escalation_rules:
                if time_until_deadline.total_seconds() / 3600 <= rule.trigger_after_hours:
                    if self._evaluate_escalation_rule(rule, assignment):
                        escalation = {
                            "type": "rule_triggered",
                            "rule": asdict(rule),
                            "assignment": asdict(assignment),
                            "escalate_to": rule.escalate_to
                        }
                        escalations.append(escalation)

        return escalations

    def _evaluate_escalation_rule(self, rule: EscalationRule, assignment: ReviewAssignment) -> bool:
        """Evaluate if an escalation rule should be triggered."""
        try:
            # This is a simplified evaluation - in practice, you'd want to use a proper JS engine
            # For now, we'll do basic string matching
            condition_lower = rule.condition.lower()

            # Check priority
            if rule.priority_threshold.value and assignment.priority.value:
                priority_levels = {"low": 1, "medium": 2, "high": 3, "critical": 4}
                if priority_levels[assignment.priority.value] < priority_levels[rule.priority_threshold.value]:
                    return False

            # Simple keyword matching for demo
            if "critical" in condition_lower and assignment.priority == Priority.CRITICAL:
                return True
            if "security" in condition_lower and "security" in str(assignment.files_to_review).lower():
                return True
            if "performance" in condition_lower and "performance" in str(assignment.files_to_review).lower():
                return True

            return False

        except Exception as e:
            print(f"Error evaluating escalation rule: {e}")
            return False

    async def execute_escalation(self, escalation: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an escalation."""
        try:
            assignment_data = escalation["assignment"]
            escalate_to = escalation["escalate_to"]

            pr = self.repo.get_pull(assignment_data["pr_number"])

            escalation_comment = f"""
🚨 **Review Escalation**

**Original Reviewer:** @{assignment_data['assigned_to']}
**Escalated to:** {', '.join(f'@{user}' for user in escalate_to)}
**Reason:** {escalation.get('type', 'Review requires attention')}
**Escalated at:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

**Original Assignment Details:**
- **PR:** #{assignment_data['pr_number']}
- **Assigned:** {assignment_data['assigned_at']}
- **Deadline:** {assignment_data['deadline']}
- **Priority:** {assignment_data['priority']}

Please review this pull request as soon as possible.
"""

            # Post escalation comment
            pr.create_issue_comment(escalation_comment)

            # Request new reviewers
            try:
                pr.create_review_request(reviewers=escalate_to)
            except GithubException as e:
                print(f"Could not create escalation review request: {e}")

            # Update assignment status
            for assignment in self.assignments:
                if assignment.pr_number == assignment_data["pr_number"] and assignment.assigned_to == assignment_data["assigned_to"]:
                    assignment.status = ReviewStatus.ESCALATED
                    break

            self._save_assignments()

            return {
                "success": True,
                "message": f"Escalated PR #{assignment_data['pr_number']} to {', '.join(escalate_to)}",
                "escalation": escalation
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Error executing escalation: {str(e)}"
            }

    async def get_team_workload(self) -> Dict[str, Any]:
        """Get current team workload statistics."""
        workload = {}

        for member in self.team_members:
            member_assignments = [a for a in self.assignments if a.assigned_to == member.username]

            pending_count = len([a for a in member_assignments if a.status == ReviewStatus.PENDING])
            in_progress_count = len([a for a in member_assignments if a.status == ReviewStatus.IN_PROGRESS])
            overdue_count = len([a for a in member_assignments
                               if a.status == ReviewStatus.PENDING and a.deadline and a.deadline < datetime.utcnow()])

            workload[member.username] = {
                "total_assignments": len(member_assignments),
                "pending": pending_count,
                "in_progress": in_progress_count,
                "overdue": overdue_count,
                "capacity": member.max_reviews_per_day,
                "utilization": len(member_assignments) / member.max_reviews_per_day,
                "expertise_areas": member.expertise_areas,
                "seniority_level": member.seniority_level
            }

        return {
            "team_workload": workload,
            "total_pending": sum(w["pending"] for w in workload.values()),
            "total_overdue": sum(w["overdue"] for w in workload.values()),
            "team_utilization": sum(w["utilization"] for w in workload.values()) / len(workload) if workload else 0
        }

    def get_review_history(self, reviewer: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
        """Get review history for the team or specific reviewer."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        relevant_assignments = []
        for assignment in self.assignments:
            if assignment.assigned_at >= cutoff_date:
                if reviewer is None or assignment.assigned_to == reviewer:
                    relevant_assignments.append(assignment)

        # Calculate statistics
        completed_reviews = [a for a in relevant_assignments if a.status in [ReviewStatus.APPROVED, ReviewStatus.CHANGES_REQUESTED, ReviewStatus.COMMENTED]]
        pending_reviews = [a for a in relevant_assignments if a.status == ReviewStatus.PENDING]

        avg_review_time = 0
        if completed_reviews:
            review_times = []
            for review in completed_reviews:
                # This is simplified - in practice, you'd track actual completion time
                review_times.append(24)  # Assume 24 hours for demo
            avg_review_time = sum(review_times) / len(review_times)

        return {
            "total_reviews": len(relevant_assignments),
            "completed_reviews": len(completed_reviews),
            "pending_reviews": len(pending_reviews),
            "average_review_time_hours": avg_review_time,
            "completion_rate": len(completed_reviews) / len(relevant_assignments) if relevant_assignments else 0,
            "assignments": [asdict(a) for a in relevant_assignments[:10]]  # Return last 10
        }

    def add_team_member(self, member: TeamMember) -> Dict[str, Any]:
        """Add a new team member."""
        self.team_members.append(member)
        self.assignment_counters[member.username] = 0
        self._save_team_members()

        return {
            "success": True,
            "message": f"Added {member.username} to the team",
            "member": asdict(member)
        }

    def update_team_member(self, username: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update team member information."""
        for member in self.team_members:
            if member.username == username:
                for key, value in updates.items():
                    if hasattr(member, key):
                        setattr(member, key, value)
                self._save_team_members()

                return {
                    "success": True,
                    "message": f"Updated {username}'s information",
                    "member": asdict(member)
                }

        return {
            "success": False,
            "message": f"Team member {username} not found"
        }


# Global team collaboration instance
_team_collaboration_instances = {}

def get_team_collaboration(github_token: str, repo_name: str) -> TeamCollaboration:
    """Get or create team collaboration instance for a repository."""
    key = f"{repo_name}"
    if key not in _team_collaboration_instances:
        _team_collaboration_instances[key] = TeamCollaboration(github_token, repo_name)
    return _team_collaboration_instances[key]