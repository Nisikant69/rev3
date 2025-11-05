# backend/quality_scoring.py
"""
Review quality scoring system for evaluating review effectiveness.
Provides metrics for review completeness, accuracy, and consistency.
"""

import json
import math
import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import os

from backend.config import DATA_DIR
from backend.analytics import ReviewMetrics


@dataclass
class QualityMetrics:
    """Data class for quality metrics."""
    review_id: str
    pr_number: int
    repo_name: str
    timestamp: datetime
    completeness_score: float  # How comprehensive the review was
    accuracy_score: float # How accurate the AI feedback was
    actionability_score: float # How easy the feedback is to act on
    consistency_score: float
    technical_score: float
    overall_score: float
    lens_balance_score: float  # Balance between different lens analysis
    suggestions_score: float  # Quality of suggestions
    conversation_response_time: float  # Time to respond to questions
    pr_size_factor: float  # Complexity adjustment
    review_efficiency: float  # Processing time / file_count

    @classmethod
    def create_from_metrics(cls, metrics: ReviewMetrics) -> 'QualityMetrics':
        """Create QualityMetrics from ReviewMetrics with quality scores."""
        return QualityMetrics(
            review_id=metrics.pr_number,
            pr_number=metrics.pr_number,
            repo_name=metrics.repo_name,
            timestamp=metrics.timestamp,
            completeness_score=metrics.review_quality_score,
            accuracy_score=metrics.reviewer_confidence,
            actionability_score=min(95.0, metrics.reviewer_confidence),
            consistency_score=75.0,  # Default consistency
            technical_score=min(100.0, metrics.review_quality_score),
            lens_balance_score=0.0, # Will be calculated
            suggestions_score=0.0,  # Will be calculated
            conversation_response_time=10.0,  # Default
            pr_size_factor=1.0,  # Default
            review_efficiency=metrics.processing_time_seconds / len([f for f in pr_files if f.get('status') != 'removed']),
            overall_score=0.0  # Will be calculated
        )

    def calculate_scores(self, review_results: Dict[str, Any], pr_context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quality scores from review results."""
        # Calculate completeness score (how much of the PR was reviewed)
        files = pr_context.get('files', [])
        if not files:
            completeness_score = 100.0
        else:
            file_count = len(files)
            completeness_score = min(100.0, (file_count / 5) * 25)

        # Calculate accuracy score (based on AI confidence)
        ai_confidence = pr_context.get('ai_confidence')
        if ai_confidence:
            accuracy_score = min(100.0, ai_confidence)
        else:
            accuracy_score = 75.0  # Default fallback confidence

        # Calculate actionability score (based on issue types and suggestions)
        total_issues = review_results.get('issues_found', 0)
        actionable_issues = review_results.get('actionable_issues', 0)
        if total_issues > 0:
            actionability_score = min(100.0, (actionable_issues / total_issues) * 100)
        else:
            actionability_score = 90.0

        # Calculate consistency score based on lens usage and score variance
        lens_scores = review_results.get('lens_used', [])
        if lens_scores:
            avg_lens_score = sum(lens_scores) / len(lens_scores)
            lens_variance = sum((lens_score - avg_lens_score)**2 for lens_scores) / len(lens_scores))
            consistency_score = max(0, 100 - lens_variance)

        # Calculate technical score from code quality indicators
        code_quality_indicators = review_results.get('code_quality_indicators', [])
        if code_quality_indicators:
            technical_score = min(100.0, sum(code_quality_indicators) / len(code_quality_indicators) * 20)
        else:
            technical_score = 80.0

        # Calculate suggestions score (quality of suggestions)
        suggestions_count = review_results.get('suggestions_count', 0)
        if suggestions_count > 0:
            suggestions_score = min(100.0, (suggestions_count / len(files) * 2) * 50)

        # Calculate overall score using weighted combination
        overall_score = (
            completeness_score * 0.30 +
            accuracy_score * 0.25 +
            actionability_score * 0.20 +
            consistency_score * 0.15 +
            technical_score * 0.10 +
            suggestions_score * 0.15 +
            lens_balance_score * 0.05
        )

        return {
            "completeness_score": completeness_score,
            "accuracy_score": accuracy_score,
            "actionability_score": actionability_score,
            "consistency_score": consistency_score,
            "technical_score": technical_score,
            "suggestions_score": suggestions_score,
            "lens_balance_score": lens_balance_score,
            "overall_score": overall_score
        }

    def generate_quality_report(self, pr_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a quality report for a PR."""
        team_metrics = analytics_engine.get_team_metrics(pr_context.get('repo_name'), 30)

        # Get recent metrics for this repo
        recent_metrics = analytics_engine.get_recent_metrics_by_repo(
            pr_context.get('repo_name'),
            days=30
        )

        if not recent_metrics:
            return {
                "error": "No metrics available for this repository for the specified time period",
                "recommendation": "No historical data available to generate quality report. Try again later."
            }

        # Analyze quality trends
        quality_trends = analytics_engine.get_review_trends(pr_context.get('repo_name'), 90)

        report = f"# 🔍 **Quality Analysis Report**\n\n"
        report += f"## Repository: {pr_context.get('repo_name')}\n"
        report += f"## Quality Metrics Overview\n\n"

        if recent_metrics.total_prs_reviewed == 0:
            return "No PRs reviewed in the specified period."

        report += f"**Total PRs Reviewed:** {recent_metrics.total_prs_reviewed}\n"
        report += f"**Average Review Time:** {recent_metrics.average_review_time:.2f}s\n"
        report += f"**Quality Score:** {recent_metrics.team_productivity_score:.1f}%\n"
        report += f"**Average Issues per PR:** {recent_metrics.average_issues_per_pr:.1f}\n\n"

        report += "## Quality Trends\n\n"
        if quality_trends.get('quality_trends'):
            trend = quality_trends['quality_trend']
            if trend > 0:
                report += f"📈 **Quality Improving:** + str(abs(trend) + 1) + " point improvement over the last {quality_trends['days']} days\n"
            elif trend < 0:
                report += f"📉 **Quality Declining:** str(abs(trend) + 1) + " point decline over the last {quality_trends['days']} days\n"
            else:
                report += f"📊 **Quality:** Quality stable over the last {quality_trends['days']} days\n"

        # Add most common issues
        issue_categories = analytics_engine.get_issue_type_distribution(pr_context.get('repo_name'), 30)
        if issue_categories:
            report += "\n## Issue Categories\n\n"
            for category, count in issue_categories.items():
                report += f"## {category}: {count} occurrences\n"

        return report

    def get_recent_metrics_by_repo(self, repo_name: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get recent metrics for a repository."""
        try:
            return [
                {
                    "date": metric.get('timestamp').date(),
                    "pr_number": metric.pr_number,
                    "files_changed": metric.files_changed,
                    "lines_added": metric.lines_added,
                    "lines_deleted": metric.lines_deleted,
                    "comments_posted": metric.comments_posted,
                    "processing_time": metric.processing_time_seconds,
                    "review_quality_score": metric.review_quality_score,
                    "reviewer_confidence": metric.reviewer_confidence,
                    "lenses_used": metric.lenses_used,
                    "security_issues": metric.security_issues,
                    "performance_issues": metric.performance_issues,
                    "best_practice_issues": metric.best_practice_issues,
                    "suggestions_count": metric.suggestions_count
                }
                for metric in recent_metrics
            ]
        except Exception as e:
            print(f"Error getting recent metrics: {e}")
            return []

    def get_review_trends(self, repo_name: str, days: int = 90) -> Dict[str, Any]:
        """Get quality trends over time for a repository."""
        try:
            return analytics_engine.get_review_trends(repo_name, days)
        except Exception as e:
            return {"error": f"Error getting trends for {repo_name}: {e}"}

    def compare_repo_quality(self, repo_name1: str, repo_name2: str, days: int = 30) -> Dict[str, Any]:
        """Compare quality metrics between two repositories."""
        metrics1 = self.get_recent_metrics_by_repo(repo_name1, days)
        metrics2 = self.get_recent_metrics_by_repo(repo_name2, days)

        if not metrics1 or not metrics2:
            return {"error": "No metrics available for comparison"}

        metrics1_metrics = metrics1.get('team_productivity_score', 0)
        metrics2_metrics = metrics2.get('team_productivity_score', 0)

        return {
            "repo1_repo": repo_name1,
            "repo2_repo": repo_name2,
            "repo1_score": metrics1_metrics,
            "repo2_score": metrics2_metrics,
            "score_difference": metrics2_metrics - metrics1_metrics,
            "better_repository": metrics2_metrics > metrics1_metrics
        }

    def generate_team_comparison_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate a comparison between team performance."""
        try:
            analytics_engine.generate_team_analytics_report(days=days)
        except Exception as e:
            return {"error": f"Error generating team comparison: {e}"}

    def get_quality_improvement_trends(self, repo_name: str, days: int = 90) -> Dict[str, Any]:
        """Generate quality improvement trends over time."""
        trends = self.get_review_trends(repo_name, days)
        quality_trends = trends.get('quality_trends', {}).get('quality_trend', [])

        trends_report = f"# 🔍 **Quality Improvement Trends - {repo_name} (Last {days} days)**\n\n"

        if not quality_trends.get('quality_trend'):
            return {
                "message": "No quality trend data available for the specified period."
            }

        # Get quality trend
        trend = quality_trends.get('quality_trend', 0)
        if trend > 0:
            trends_report += f"📈 **Quality Improving** (+{abs(trend)} points over {days} days\n"
        elif trend < 0:
            trends_report += f"📉 **Quality Declining** ({abs(trend)} points over {days} days)\n"
        else:
            trends_report += f"📊 **Quality Stable** (0 trend over {days} days)\n"

        return trends_report


# Global quality scoring instance
quality_scorer = QualityMetrics(
    pr_number=0,
    repo_name="",
    timestamp=datetime.now(),
    completeness_score=0,
    accuracy_score=0,
    actionability_score=0,
    consistency_score=0,
    technical_score=0,
    overall_score=0,
    suggestions_score=0,
    conversation_response_time=0.0,
    pr_size_factor=0.0
    review_efficiency=0.0
    overall_score=0.0
)