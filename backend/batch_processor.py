# backend/batch_processor.py
"""
Batch processing system for handling multiple PRs simultaneously.
Provides queue management, prioritization, and bulk analysis capabilities.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from backend.config import DATA_DIR, MAX_DIFF_SIZE
from backend.reviewer import review_patch_line_level
from backend.review_lenses import multi_lens_review
from backend.summarizer import generate_pr_summary, format_summary_for_comment
from backend.labeler import generate_pr_labels, apply_labels_to_pr, create_missing_labels
from backend.suggestions import generate_suggestions_for_file
from backend.semantic_search import semantic_search
from backend.analytics import analytics_engine, ReviewMetrics


@dataclass
class BatchJob:
    """Represents a single batch review job."""
    job_id: str
    repo_name: str
    pr_number: int
    pr_data: Dict[str, Any]
    priority: int  # 1=high, 2=medium, 3=low
    created_at: datetime
    status: str  # 'queued', 'processing', 'completed', 'failed'
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    max_concurrent_jobs: int = 3
    max_queue_size: int = 100
    job_timeout_minutes: int = 30
    retry_attempts: int = 2
    enable_auto_prioritization: bool = True
    enable_analytics: bool = True
    default_lenses: List[str] = None


class BatchProcessor:
    """Advanced batch processing system for PR reviews."""

    def __init__(self, config: BatchConfig = None):
        self.config = config or BatchConfig()
        self.queue_file = DATA_DIR / "batch" / "job_queue.json"
        self.queue_file.parent.mkdir(parents=True, exist_ok=True)

        self.job_queue = []
        self.active_jobs = {}  # job_id -> BatchJob
        self.completed_jobs = {}
        self.failed_jobs = {}

        self.load_queue()
        self.processing = False

    def load_queue(self):
        """Load job queue from file."""
        try:
            if self.queue_file.exists():
                with open(self.queue_file, 'r') as f:
                    data = json.load(f)

                # Convert dicts back to BatchJob objects
                for job_data in data.get('queue', []):
                    job = BatchJob(
                        job_id=job_data['job_id'],
                        repo_name=job_data['repo_name'],
                        pr_number=job_data['pr_number'],
                        pr_data=job_data['pr_data'],
                        priority=job_data['priority'],
                        created_at=datetime.fromisoformat(job_data['created_at']),
                        status=job_data['status'],
                        started_at=datetime.fromisoformat(job_data['started_at']) if job_data.get('started_at') else None,
                        completed_at=datetime.fromisoformat(job_data['completed_at']) if job_data.get('completed_at') else None,
                        error_message=job_data.get('error_message'),
                        result=job_data.get('result')
                    )
                    self.job_queue.append(job)

                self.active_jobs = {
                    job_data['job_id']: BatchJob(**job_data)
                    for job_data in data.get('active_jobs', {}).values()
                }

                self.completed_jobs = {
                    job_data['job_id']: BatchJob(**job_data)
                    for job_data in data.get('completed_jobs', {}).values()
                }

                self.failed_jobs = {
                    job_data['job_id']: BatchJob(**job_data)
                    for job_data in data.get('failed_jobs', {}).values()
                }

        except Exception as e:
            print(f"Error loading batch queue: {e}")

    def save_queue(self):
        """Save job queue to file."""
        try:
            data = {
                'queue': [asdict(job) for job in self.job_queue],
                'active_jobs': {job_id: asdict(job) for job_id, job in self.active_jobs.items()},
                'completed_jobs': {job_id: asdict(job) for job_id, job in self.completed_jobs.items()},
                'failed_jobs': {job_id: asdict(job) for job_id, job in self.failed_jobs.items()},
                'last_updated': datetime.now().isoformat()
            }

            with open(self.queue_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving batch queue: {e}")

    def add_job(self, repo_name: str, pr_data: Dict[str, Any],
                priority: int = 2, job_id: str = None) -> str:
        """Add a job to the batch queue."""

        if len(self.job_queue) >= self.config.max_queue_size:
            raise Exception("Batch queue is full")

        if job_id is None:
            job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{repo_name}_{pr_data.get('number', 'unknown')}"

        job = BatchJob(
            job_id=job_id,
            repo_name=repo_name,
            pr_number=pr_data.get('number', 0),
            pr_data=pr_data,
            priority=priority,
            created_at=datetime.now(),
            status='queued'
        )

        if self.config.enable_auto_prioritization:
            self.job_queue.append(job)
            self.job_queue.sort(key=lambda x: (x.priority, x.created_at))
        else:
            self.job_queue.append(job)

        self.save_queue()
        return job_id

    def add_batch_jobs(self, repo_name: str, pr_list: List[Dict[str, Any]],
                      default_priority: int = 2) -> List[str]:
        """Add multiple jobs to the batch queue."""
        job_ids = []

        for pr_data in pr_list:
            # Auto-prioritize based on PR characteristics
            priority = self.calculate_priority(pr_data, default_priority)
            job_id = self.add_job(repo_name, pr_data, priority)
            job_ids.append(job_id)

        return job_ids

    def calculate_priority(self, pr_data: Dict[str, Any], default_priority: int) -> int:
        """Calculate priority for a PR based on various factors."""
        priority = default_priority

        # Higher priority for urgent PRs
        title = pr_data.get('title', '').lower()
        if any(keyword in title for keyword in ['urgent', 'critical', 'hotfix', 'security']):
            priority = 1  # High priority

        # Higher priority for large changes (might need more attention)
        files = pr_data.get('files', [])
        if len(files) > 10:
            priority = 1
        elif len(files) > 5:
            priority = min(priority, 2)

        # Higher priority for certain file types
        security_files = ['auth', 'security', 'password', 'token', 'key']
        performance_files = ['performance', 'cache', 'optimize', 'speed']

        for file_data in files:
            filename = file_data.get('filename', '').lower()
            if any(security_file in filename for security_file in security_files):
                priority = 1  # High priority for security changes
                break
            elif any(perf_file in filename for perf_file in performance_files):
                priority = min(priority, 2)  # Medium priority for performance

        return priority

    def get_next_job(self) -> Optional[BatchJob]:
        """Get the next job from the queue."""
        if not self.job_queue:
            return None

        job = self.job_queue.pop(0)
        job.status = 'processing'
        job.started_at = datetime.now()

        self.active_jobs[job.job_id] = job
        self.save_queue()

        return job

    def complete_job(self, job_id: str, result: Dict[str, Any]):
        """Mark a job as completed."""
        if job_id not in self.active_jobs:
            return

        job = self.active_jobs[job_id]
        job.status = 'completed'
        job.completed_at = datetime.now()
        job.result = result

        self.completed_jobs[job_id] = job
        del self.active_jobs[job_id]

        self.save_queue()

    def fail_job(self, job_id: str, error_message: str):
        """Mark a job as failed."""
        if job_id not in self.active_jobs:
            return

        job = self.active_jobs[job_id]
        job.status = 'failed'
        job.completed_at = datetime.now()
        job.error_message = error_message

        self.failed_jobs[job_id] = job
        del self.active_jobs[job_id]

        self.save_queue()

    def retry_failed_jobs(self):
        """Retry jobs that failed due to transient errors."""
        retry_jobs = []

        for job_id, job in list(self.failed_jobs.items()):
            # Retry if error seems transient or if we haven't retried much
            if ('timeout' not in job.error_message.lower() and
                'rate limit' not in job.error_message.lower()):

                job.status = 'queued'
                job.error_message = None
                job.started_at = None
                job.completed_at = None

                self.job_queue.append(job)
                retry_jobs.append(job_id)
                del self.failed_jobs[job_id]

        if retry_jobs:
            self.save_queue()
            return f"Retrying {len(retry_jobs)} failed jobs"

        return "No failed jobs to retry"

    def process_single_job(self, job: BatchJob, github_client, repo):
        """Process a single batch job."""
        start_time = time.time()

        try:
            pr = repo.get_pull(job.pr_number)
            head_sha = pr.head.sha

            # Get PR files
            files = list(pr.get_files())

            # Index repository if needed
            from backend.repo_fetcher import save_repo_snapshot
            from backend.context_indexer import index_repo

            repo_dir = save_repo_snapshot(repo, head_sha)
            index, metadata = index_repo(repo_dir, job.repo_name, head_sha)

            # Process the review
            result = self.process_pr_review(
                pr, files, job.repo_name, head_sha, index, metadata
            )

            # Record metrics if analytics is enabled
            if self.config.enable_analytics:
                processing_time = time.time() - start_time

                metrics = ReviewMetrics(
                    pr_number=job.pr_number,
                    repo_name=job.repo_name,
                    timestamp=datetime.now(),
                    files_changed=len([f for f in files if f.status != 'removed']),
                    lines_added=sum(f.additions for f in files if hasattr(f, 'additions')),
                    lines_deleted=sum(f.deletions for f in files if hasattr(f, 'deletions')),
                    comments_posted=len(result.get('comments', [])),
                    security_issues=result.get('security_issues', 0),
                    performance_issues=result.get('performance_issues', 0),
                    best_practice_issues=result.get('best_practice_issues', 0),
                    suggestions_generated=result.get('suggestions_count', 0),
                    processing_time_seconds=processing_time,
                    review_quality_score=result.get('quality_score', 0),
                    reviewer_confidence=result.get('confidence_score', 0)
                )

                analytics_engine.save_metrics(metrics)

            self.complete_job(job.job_id, result)
            return {"success": True, "job_id": job.job_id, "result": result}

        except Exception as e:
            error_msg = str(e)
            self.fail_job(job.job_id, error_msg)
            return {"success": False, "job_id": job.job_id, "error": error_msg}

    def process_pr_review(self, pr, files, repo_name: str, head_sha: str,
                          index, metadata) -> Dict[str, Any]:
        """Process a PR review with all enhanced features."""

        all_review_comments = []
        summary_blocks = []
        security_issues = 0
        performance_issues = 0
        best_practice_issues = 0
        suggestions_count = 0

        # Default lenses to use
        lenses = self.config.default_lenses or ["security", "performance", "best_practices"]

        for file in files:
            if file.status == "removed" or not file.patch:
                continue
            if len(file.patch) > MAX_DIFF_SIZE:
                continue

            # Standard review
            review_result = review_patch_line_level(
                file.patch,
                file.filename,
                repo_name,
                pr.head.ref,
                head_sha,
                index,
                metadata
            )

            if review_result and review_result.get("comments"):
                all_review_comments.extend(review_result["comments"])
                summary_blocks.append(f"### {review_result['summary']}")

            # Multi-lens analysis
            try:
                context_chunks = semantic_search(
                    file.patch, index, metadata, repo_name, head_sha, file.filename
                )

                lens_result = multi_lens_review(
                    file.patch,
                    file.filename,
                    lenses,
                    context_chunks
                )

                if lens_result.get("all_comments"):
                    for comment in lens_result["all_comments"]:
                        if comment.get("line"):
                            all_review_comments.append({
                                "path": comment["path"],
                                "body": comment["body"],
                                "line": comment["line"]
                            })

                # Count issues by type
                security_issues += len(lens_result.get("lens_results", {}).get("security", {}).get("issues", []))
                performance_issues += len(lens_result.get("lens_results", {}).get("performance", {}).get("issues", []))
                best_practice_issues += len(lens_result.get("lens_results", {}).get("best_practices", {}).get("issues", []))

            except Exception as e:
                print(f"⚠️ Error in multi-lens analysis for {file.filename}: {e}")

            # Generate suggestions
            try:
                suggestion_result = generate_suggestions_for_file(
                    file.patch, file.filename, context_chunks if 'context_chunks' in locals() else []
                )
                suggestions_count += len(suggestion_result.get("suggestions", []))

                if suggestion_result.get("github_suggestions"):
                    for suggestion in suggestion_result["github_suggestions"]:
                        all_review_comments.append({
                            "path": suggestion["path"],
                            "body": f"💡 **Code Suggestion**\n\n{suggestion['body']}\n\n**Proposed Fix:**\n```suggestion\n{suggestion['suggestion']}\n```",
                            "line": suggestion.get("position")
                        })

            except Exception as e:
                print(f"⚠️ Error generating suggestions for {file.filename}: {e}")

        # Generate PR summary and labels
        try:
            pr_summary = generate_pr_summary(files, repo_name, pr.title, pr.body)
            summary_comment = format_summary_for_comment(pr_summary)
            summary_blocks.insert(0, summary_comment)

            # Apply labels
            existing_labels = [label.name for label in pr.get_labels()]
            label_result = generate_pr_labels(files, pr.title, pr.body, existing_labels)

            if label_result.get("labels"):
                create_missing_labels(repo, label_result["labels"])
                apply_labels_to_pr(pr, label_result["labels"])

        except Exception as e:
            print(f"⚠️ Error in summary/labeling: {e}")

        # Create the review
        if all_review_comments:
            overall_summary = f"🤖 **Batch AI Review**\n\n"
            overall_summary += f"📊 **Files Analyzed:** {len([f for f in files if f.status != 'removed' and f.patch and len(f.patch) <= MAX_DIFF_SIZE])}\n"
            overall_summary += f"🔍 **Lenses Applied:** {', '.join(lenses)}\n"
            overall_summary += f"💡 **Issues Found:** {len(all_review_comments)} total\n\n"
            overall_summary += "---\n" + "\n---\n".join(summary_blocks)

            try:
                pr.create_review(
                    body=overall_summary,
                    event="COMMENT",
                    comments=all_review_comments,
                )
            except Exception as e:
                print(f"⚠️ Error creating review: {e}")
                pr.create_review(
                    body=overall_summary + f"\n\n⚠️ Could not add line-specific comments due to API limitations.",
                    event="COMMENT",
                )
        else:
            # No specific comments, just summary
            if summary_blocks:
                overall_summary = "🤖 **Batch AI Review**\n\n" + "\n---\n".join(summary_blocks)
                pr.create_review(
                    body=overall_summary,
                    event="COMMENT",
                )
            else:
                pr.create_review(
                    body="🤖 **Batch AI Review**\n\nNo issues found!",
                    event="COMMENT",
                )

        # Calculate quality scores
        quality_score = min(50 + len(all_review_comments) * 5, 100)
        confidence_score = min(70 + len(lenses) * 5, 100)

        return {
            "comments_count": len(all_review_comments),
            "security_issues": security_issues,
            "performance_issues": performance_issues,
            "best_practice_issues": best_practice_issues,
            "suggestions_count": suggestions_count,
            "quality_score": quality_score,
            "confidence_score": confidence_score,
            "lenses_used": lenses,
            "summary_blocks": summary_blocks
        }

    def start_processing(self, github_client):
        """Start the batch processing loop."""
        if self.processing:
            return "Batch processing is already running"

        self.processing = True

        def process_loop():
            while self.processing and (self.job_queue or self.active_jobs):
                # Process available jobs
                while (len(self.active_jobs) < self.config.max_concurrent_jobs and
                       self.job_queue):
                    job = self.get_next_job()
                    if not job:
                        break

                    # Check timeout
                    if (datetime.now() - job.started_at).total_seconds() >
                       self.config.job_timeout_minutes * 60):
                        self.fail_job(job.job_id, "Job timed out")
                        continue

                    # Process job in background
                    try:
                        repo = github_client.get_repo(job.repo_name)
                        result = self.process_single_job(job, github_client, repo)
                        print(f"✅ Completed batch job {job.job_id}")
                    except Exception as e:
                        print(f"❌ Failed batch job {job.job_id}: {e}")

                # Small delay to prevent busy loop
                time.sleep(1)

            print("Batch processing completed")
            self.processing = False

        # Start processing in background
        import threading
        thread = threading.Thread(target=process_loop, daemon=True)
        thread.start()

        return f"Started batch processing with {len(self.active_jobs)} active jobs"

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current status of the batch queue."""
        return {
            "queue_length": len(self.job_queue),
            "active_jobs": len(self.active_jobs),
            "completed_jobs": len(self.completed_jobs),
            "failed_jobs": len(self.failed_jobs),
            "processing": self.processing,
            "max_concurrent_jobs": self.config.max_concurrent_jobs,
            "max_queue_size": self.config.max_queue_size,
            "queued_jobs": [
                {
                    "job_id": job.job_id,
                    "repo_name": job.repo_name,
                    "pr_number": job.pr_number,
                    "priority": job.priority,
                    "created_at": job.created_at.isoformat()
                }
                for job in self.job_queue[:10]  # Show first 10 jobs
            ],
            "active_job_details": [
                {
                    "job_id": job.job_id,
                    "repo_name": job.repo_name,
                    "pr_number": job.pr_number,
                    "started_at": job.started_at.isoformat() if job.started_at else None
                }
                for job in self.active_jobs.values()
            ]
        }

    def clear_completed_jobs(self, older_than_days: int = 7):
        """Clear old completed jobs to save memory."""
        cutoff_date = datetime.now() - timedelta(days=older_than_days)

        jobs_to_remove = []
        for job_id, job in self.completed_jobs.items():
            if job.completed_at and job.completed_at < cutoff_date:
                jobs_to_remove.append(job_id)

        for job_id in jobs_to_remove:
            del self.completed_jobs[job_id]

        if jobs_to_remove:
            self.save_queue()
            return f"Cleared {len(jobs_to_remove)} completed jobs older than {older_than_days} days"

        return f"No completed jobs to clear (older than {older_than_days} days)"

    def get_job_details(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific job."""
        if job_id in self.active_jobs:
            return asdict(self.active_jobs[job_id])
        elif job_id in self.completed_jobs:
            return asdict(self.completed_jobs[job_id])
        elif job_id in self.failed_jobs:
            return asdict(self.failed_jobs[job_id])
        else:
            return None

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued job."""
        for i, job in enumerate(self.job_queue):
            if job.job_id == job_id:
                removed_job = self.job_queue.pop(i)
                removed_job.status = 'cancelled'
                self.failed_jobs[job_id] = removed_job
                self.save_queue()
                return True
        return False


# Global batch processor instance
batch_processor = BatchProcessor()