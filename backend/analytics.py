# backend/analytics.py
"""
Advanced analytics system for PR review metrics and insights.
Provides comprehensive analysis of review patterns, team performance, and code quality trends.
"""

import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import os

from config import DATA_DIR


@dataclass
class ReviewMetrics:
    """Data class for review metrics."""
    pr_number: int
    repo_name: str
    timestamp: datetime
    files_changed: int
    lines_added: int
    lines_deleted: int
    comments_posted: int
    security_issues: int
    performance_issues: int
    best_practice_issues: int
    suggestions_generated: int
    processing_time_seconds: float
    review_quality_score: float
    reviewer_confidence: float


@dataclass
class TeamMetrics:
    """Data class for team performance metrics."""
    repo_name: str
    date_range: str
    total_prs_reviewed: int
    average_review_time: float
    average_issues_per_pr: float
    security_detection_rate: float
    performance_detection_rate: float
    code_quality_trend: float
    team_productivity_score: float
    most_common_issue_types: List[str]
    review_consistency_score: float


class ReviewAnalytics:
    """Advanced analytics engine for PR review metrics."""

    def __init__(self):
        self.metrics_file = DATA_DIR / "analytics" / "review_metrics.json"
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_history = self.load_metrics()

    def load_metrics(self) -> List[Dict[str, Any]]:
        """Load historical metrics from file."""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading metrics: {e}")
                return []
        return []

    def save_metrics(self, metrics: ReviewMetrics):
        """Save metrics to file."""
        try:
            # Load existing metrics
            all_metrics = self.load_metrics()

            # Add new metrics
            metrics_dict = asdict(metrics)
            metrics_dict['timestamp'] = metrics_dict['timestamp'].isoformat()
            all_metrics.append(metrics_dict)

            # Keep only last 1000 metrics to prevent file from growing too large
            if len(all_metrics) > 1000:
                all_metrics = all_metrics[-1000:]

            # Save to file
            with open(self.metrics_file, 'w') as f:
                json.dump(all_metrics, f, indent=2)

        except Exception as e:
            print(f"Error saving metrics: {e}")

    def record_review(self, pr_data: Dict[str, Any], review_results: Dict[str, Any],
                      processing_time: float = 0.0) -> ReviewMetrics:
        """Record metrics for a completed review."""

        # Extract file metrics
        files = pr_data.get('files', [])
        files_changed = len([f for f in files if f.get('status') != 'removed'])
        lines_added = sum(f.get('additions', 0) for f in files)
        lines_deleted = sum(f.get('deletions', 0) for f in files)

        # Count different types of issues
        comments_posted = len(review_results.get('comments', []))

        # Count lens-specific issues if available
        lens_results = review_results.get('lens_results', {})
        security_issues = len(lens_results.get('security', {}).get('issues', []))
        performance_issues = len(lens_results.get('performance', {}).get('issues', []))
        best_practice_issues = len(lens_results.get('best_practices', {}).get('issues', []))

        # Count suggestions
        suggestions_generated = len(review_results.get('suggestions', []))

        # Calculate quality scores
        review_quality_score = self.calculate_review_quality_score(review_results)
        reviewer_confidence = self.calculate_reviewer_confidence(review_results)

        metrics = ReviewMetrics(
            pr_number=pr_data.get('number', 0),
            repo_name=pr_data.get('repo_name', ''),
            timestamp=datetime.now(),
            files_changed=files_changed,
            lines_added=lines_added,
            lines_deleted=lines_deleted,
            comments_posted=comments_posted,
            security_issues=security_issues,
            performance_issues=performance_issues,
            best_practice_issues=best_practice_issues,
            suggestions_generated=suggestions_generated,
            processing_time_seconds=processing_time,
            review_quality_score=review_quality_score,
            reviewer_confidence=reviewer_confidence
        )

        self.save_metrics(metrics)
        return metrics

    def calculate_review_quality_score(self, review_results: Dict[str, Any]) -> float:
        """Calculate a quality score for the review (0-100)."""
        score = 0.0

        # Base score for having comments
        comments = review_results.get('comments', [])
        if comments:
            score += 30  # Base score for providing feedback

        # Bonus for detailed comments
        detailed_comments = sum(1 for comment in comments
                              if len(comment.get('body', '')) > 50)
        score += min(detailed_comments * 5, 30)  # Up to 30 points for detail

        # Security and performance coverage
        lens_results = review_results.get('lens_results', {})
        if lens_results.get('security', {}).get('issues'):
            score += 15
        if lens_results.get('performance', {}).get('issues'):
            score += 15
        if lens_results.get('best_practices', {}).get('issues'):
            score += 10

        # Suggestions (actionable feedback)
        suggestions = review_results.get('suggestions', [])
        score += min(len(suggestions) * 3, 10)  # Up to 10 points for suggestions

        return min(score, 100.0)

    def calculate_reviewer_confidence(self, review_results: Dict[str, Any]) -> float:
        """Calculate reviewer confidence based on AI response quality."""
        confidence = 70.0  # Base confidence

        # Increase confidence for detailed responses
        total_content = sum(len(comment.get('body', ''))
                           for comment in review_results.get('comments', []))

        if total_content > 1000:
            confidence += 20
        elif total_content > 500:
            confidence += 10

        # Increase confidence for multi-lens analysis
        lens_results = review_results.get('lens_results', {})
        lenses_used = len([k for k, v in lens_results.items() if v.get('issues')])
        confidence += min(lenses_used * 5, 10)

        return min(confidence, 100.0)

    def get_team_metrics(self, repo_name: str, days: int = 30) -> TeamMetrics:
        """Get team metrics for a repository over a time period."""

        # Filter metrics for the specified repo and time period
        cutoff_date = datetime.now() - timedelta(days=days)
        relevant_metrics = []

        for metric_dict in self.metrics_history:
            if (metric_dict.get('repo_name') == repo_name and
                datetime.fromisoformat(metric_dict['timestamp']) > cutoff_date):
                relevant_metrics.append(metric_dict)

        if not relevant_metrics:
            return self.create_empty_team_metrics(repo_name, days)

        # Calculate team metrics
        total_prs = len(relevant_metrics)
        avg_review_time = sum(m['processing_time_seconds'] for m in relevant_metrics) / total_prs
        avg_issues = sum((m['security_issues'] + m['performance_issues'] +
                         m['best_practice_issues']) for m in relevant_metrics) / total_prs

        # Detection rates
        total_security_issues = sum(m['security_issues'] for m in relevant_metrics)
        total_performance_issues = sum(m['performance_issues'] for m in relevant_metrics)
        security_rate = (total_security_issues / total_prs) * 100 if total_prs > 0 else 0
        performance_rate = (total_performance_issues / total_prs) * 100 if total_prs > 0 else 0

        # Quality trends
        quality_scores = [m['review_quality_score'] for m in relevant_metrics]
        quality_trend = (quality_scores[-1] - quality_scores[0]) if len(quality_scores) > 1 else 0

        # Productivity score (based on review speed and quality)
        productivity_score = ((100 - min(avg_review_time, 60)) * 0.6 +
                            sum(quality_scores) / total_prs * 0.4)

        # Most common issue types
        issue_types = []
        for m in relevant_metrics:
            if m['security_issues'] > 0:
                issue_types.extend(['security'] * m['security_issues'])
            if m['performance_issues'] > 0:
                issue_types.extend(['performance'] * m['performance_issues'])
            if m['best_practice_issues'] > 0:
                issue_types.extend(['best_practices'] * m['best_practice_issues'])

        from collections import Counter
        most_common = Counter(issue_types).most_common(3)

        # Consistency score (variance in review quality)
        if len(quality_scores) > 1:
            variance = sum((x - sum(quality_scores)/len(quality_scores))**2
                          for x in quality_scores) / len(quality_scores)
            consistency_score = max(0, 100 - variance)
        else:
            consistency_score = 100

        return TeamMetrics(
            repo_name=repo_name,
            date_range=f"Last {days} days",
            total_prs_reviewed=total_prs,
            average_review_time=avg_review_time,
            average_issues_per_pr=avg_issues,
            security_detection_rate=security_rate,
            performance_detection_rate=performance_rate,
            code_quality_trend=quality_trend,
            team_productivity_score=productivity_score,
            most_common_issue_types=[item[0] for item in most_common],
            review_consistency_score=consistency_score
        )

    def create_empty_team_metrics(self, repo_name: str, days: int) -> TeamMetrics:
        """Create empty team metrics when no data is available."""
        return TeamMetrics(
            repo_name=repo_name,
            date_range=f"Last {days} days",
            total_prs_reviewed=0,
            average_review_time=0.0,
            average_issues_per_pr=0.0,
            security_detection_rate=0.0,
            performance_detection_rate=0.0,
            code_quality_trend=0.0,
            team_productivity_score=0.0,
            most_common_issue_types=[],
            review_consistency_score=0.0
        )

    def get_top_reviewed_files(self, repo_name: str, days: int = 30, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most frequently reviewed files."""
        file_counts = {}

        cutoff_date = datetime.now() - timedelta(days=days)
        for metric_dict in self.metrics_history:
            if (metric_dict.get('repo_name') == repo_name and
                datetime.fromisoformat(metric_dict['timestamp']) > cutoff_date):
                # This would need to be enhanced to track individual files
                # For now, return a placeholder
                pass

        return []

    def generate_analytics_report(self, repo_name: str, days: int = 30) -> Dict[str, Any]:
        """Generate a comprehensive analytics report."""

        team_metrics = self.get_team_metrics(repo_name, days)

        report = {
            "repo_name": repo_name,
            "report_period": f"Last {days} days",
            "generated_at": datetime.now().isoformat(),
            "team_metrics": asdict(team_metrics),
            "key_insights": self.generate_key_insights(team_metrics),
            "recommendations": self.generate_recommendations(team_metrics)
        }

        return report

    def generate_key_insights(self, metrics: TeamMetrics) -> List[str]:
        """Generate key insights from team metrics."""
        insights = []

        if metrics.total_prs_reviewed == 0:
            insights.append("No PR reviews have been recorded yet.")
            return insights

        # Review volume insights
        if metrics.total_prs_reviewed > 50:
            insights.append(f"High review activity: {metrics.total_prs_reviewed} PRs reviewed in {metrics.date_range}")
        elif metrics.total_prs_reviewed < 10:
            insights.append(f"Low review activity: Only {metrics.total_prs_reviewed} PRs reviewed in {metrics.date_range}")

        # Performance insights
        if metrics.average_review_time > 30:
            insights.append("Reviews are taking longer than optimal (avg: {:.1f}s)".format(metrics.average_review_time))
        elif metrics.average_review_time < 10:
            insights.append("Excellent review speed (avg: {:.1f}s)".format(metrics.average_review_time))

        # Quality insights
        if metrics.team_productivity_score > 80:
            insights.append("Excellent team productivity score")
        elif metrics.team_productivity_score < 50:
            insights.append("Team productivity could be improved")

        # Issue detection insights
        if metrics.security_detection_rate > 50:
            insights.append("Strong security issue detection")
        if metrics.performance_detection_rate > 50:
            insights.append("Good performance issue detection")

        # Consistency insights
        if metrics.review_consistency_score > 80:
            insights.append("Very consistent review quality")
        elif metrics.review_consistency_score < 60:
            insights.append("Review quality could be more consistent")

        return insights

    def generate_recommendations(self, metrics: TeamMetrics) -> List[str]:
        """Generate actionable recommendations based on metrics."""
        recommendations = []

        if metrics.average_review_time > 20:
            recommendations.append("Consider optimizing review processing to reduce average review time")

        if metrics.security_detection_rate < 30:
            recommendations.append("Focus more on security issue detection in reviews")

        if metrics.performance_detection_rate < 30:
            recommendations.append("Pay more attention to performance-related issues")

        if metrics.review_consistency_score < 70:
            recommendations.append("Implement review templates or checklists for more consistent reviews")

        if metrics.team_productivity_score < 60:
            recommendations.append("Consider additional training or tools to improve review efficiency")

        if metrics.most_common_issue_types:
            most_common = metrics.most_common_issue_types[0]
            if most_common == "security":
                recommendations.append("Consider additional security training and guidelines")
            elif most_common == "performance":
                recommendations.append("Focus on performance optimization best practices")
            elif most_common == "best_practices":
                recommendations.append("Review team coding standards and guidelines")

        return recommendations

    def export_metrics_to_csv(self, repo_name: str, days: int = 30) -> str:
        """Export metrics to CSV format."""
        import csv
        from io import StringIO

        output = StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow([
            'PR Number', 'Timestamp', 'Files Changed', 'Lines Added', 'Lines Deleted',
            'Comments Posted', 'Security Issues', 'Performance Issues',
            'Best Practice Issues', 'Suggestions', 'Processing Time (s)',
            'Quality Score', 'Confidence Score'
        ])

        # Write data
        cutoff_date = datetime.now() - timedelta(days=days)
        for metric_dict in self.metrics_history:
            if (metric_dict.get('repo_name') == repo_name and
                datetime.fromisoformat(metric_dict['timestamp']) > cutoff_date):
                writer.writerow([
                    metric_dict['pr_number'],
                    metric_dict['timestamp'],
                    metric_dict['files_changed'],
                    metric_dict['lines_added'],
                    metric_dict['lines_deleted'],
                    metric_dict['comments_posted'],
                    metric_dict['security_issues'],
                    metric_dict['performance_issues'],
                    metric_dict['best_practice_issues'],
                    metric_dict['suggestions_generated'],
                    metric_dict['processing_time_seconds'],
                    metric_dict['review_quality_score'],
                    metric_dict['reviewer_confidence']
                ])

        return output.getvalue()

    def get_review_trends(self, repo_name: str, days: int = 90) -> Dict[str, Any]:
        """Analyze review trends over time."""

        cutoff_date = datetime.now() - timedelta(days=days)
        relevant_metrics = []

        for metric_dict in self.metrics_history:
            if (metric_dict.get('repo_name') == repo_name and
                datetime.fromisoformat(metric_dict['timestamp']) > cutoff_date):
                relevant_metrics.append(metric_dict)

        if not relevant_metrics:
            return {"error": "No data available for the specified period"}

        # Group by day
        daily_metrics = {}
        for metric in relevant_metrics:
            date = datetime.fromisoformat(metric['timestamp']).date()
            if date not in daily_metrics:
                daily_metrics[date] = []
            daily_metrics[date].append(metric)

        # Calculate daily averages
        trends = {
            "dates": [],
            "avg_quality_scores": [],
            "avg_processing_times": [],
            "total_prs_per_day": [],
            "security_issues_per_day": [],
            "performance_issues_per_day": []
        }

        for date in sorted(daily_metrics.keys()):
            day_metrics = daily_metrics[date]

            trends["dates"].append(date.isoformat())
            trends["avg_quality_scores"].append(
                sum(m['review_quality_score'] for m in day_metrics) / len(day_metrics)
            )
            trends["avg_processing_times"].append(
                sum(m['processing_time_seconds'] for m in day_metrics) / len(day_metrics)
            )
            trends["total_prs_per_day"].append(len(day_metrics))
            trends["security_issues_per_day"].append(
                sum(m['security_issues'] for m in day_metrics)
            )
            trends["performance_issues_per_day"].append(
                sum(m['performance_issues'] for m in day_metrics)
            )

        return trends


# Global analytics instance
analytics_engine = ReviewAnalytics()