# backend/api_rate_limiter.py
"""
API rate limiting and retry logic for handling quota exceeded errors.
Provides exponential backoff, queuing, and monitoring for API calls.
"""

import time
import asyncio
import json
import threading
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
from enum import Enum

from backend.config import DATA_DIR


class RateLimitError(Exception):
    """Custom exception for rate limit exceeded."""
    pass


class QuotaExceededError(Exception):
    """Custom exception for quota exceeded."""
    pass


@dataclass
class APICall:
    """Represents an API call with its parameters."""
    id: str
    function: Callable
    args: tuple
    kwargs: dict
    created_at: datetime
    retry_count: int = 0
    max_retries: int = 3
    priority: int = 1  # 1=high, 2=medium, 3=low
    delay_until: Optional[datetime] = None


@dataclass
class APIUsageStats:
    """API usage statistics."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    quota_exceeded_count: int = 0
    rate_limited_count: int = 0
    last_reset_time: datetime = None
    daily_usage: int = 0
    hourly_usage: int = 0

    def __post_init__(self):
        if self.last_reset_time is None:
            self.last_reset_time = datetime.now()


class RateLimiter:
    """Rate limiter with exponential backoff and queue management."""

    def __init__(self, max_calls_per_minute: int = 60, max_calls_per_hour: int = 1000):
        self.max_calls_per_minute = max_calls_per_minute
        self.max_calls_per_hour = max_calls_per_hour

        # Call tracking
        self.minute_calls = deque()  # Timestamps of calls in the last minute
        self.hour_calls = deque()    # Timestamps of calls in the last hour

        # Queue for failed calls
        self.pending_queue = []
        self.processing_queue = False

        # Statistics
        self.stats = APIUsageStats()

        # Thread safety
        self.lock = threading.Lock()

        # Rate limit backoff parameters
        self.base_delay = 1.0  # Base delay in seconds
        self.max_delay = 300.0  # Maximum delay in seconds (5 minutes)
        self.backoff_multiplier = 2.0

        # Load existing stats
        self.load_stats()

    def load_stats(self):
        """Load usage statistics from file."""
        try:
            stats_file = DATA_DIR / "api_stats.json"
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    data = json.load(f)
                    self.stats = APIUsageStats(**data)
                    # Convert string timestamps back to datetime
                    if isinstance(self.stats.last_reset_time, str):
                        self.stats.last_reset_time = datetime.fromisoformat(self.stats.last_reset_time)
        except Exception as e:
            print(f"Error loading API stats: {e}")

    def save_stats(self):
        """Save usage statistics to file."""
        try:
            stats_file = DATA_DIR / "api_stats.json"
            stats_file.parent.mkdir(parents=True, exist_ok=True)

            # Convert datetime to string for JSON serialization
            stats_dict = asdict(self.stats)
            if isinstance(stats_dict['last_reset_time'], datetime):
                stats_dict['last_reset_time'] = stats_dict['last_reset_time'].isoformat()

            with open(stats_file, 'w') as f:
                json.dump(stats_dict, f, indent=2)
        except Exception as e:
            print(f"Error saving API stats: {e}")

    def is_rate_limited(self) -> bool:
        """Check if we're currently rate limited."""
        now = datetime.now()

        # Clean old calls
        self.minute_calls = deque([
            ts for ts in self.minute_calls
            if now - ts < timedelta(minutes=1)
        ])
        self.hour_calls = deque([
            ts for ts in self.hour_calls
            if now - ts < timedelta(hours=1)
        ])

        # Check limits
        return (len(self.minute_calls) >= self.max_calls_per_minute or
                len(self.hour_calls) >= self.max_calls_per_hour)

    def wait_if_needed(self):
        """Wait if we're rate limited."""
        if self.is_rate_limited():
            # Calculate wait time
            now = datetime.now()
            if len(self.minute_calls) >= self.max_calls_per_minute:
                # Wait until oldest minute call is 60 seconds old
                oldest_call = min(self.minute_calls)
                wait_time = max(0, 60 - (now - oldest_call).total_seconds())
            else:
                # Wait until oldest hour call is 3600 seconds old
                oldest_call = min(self.hour_calls)
                wait_time = max(0, 3600 - (now - oldest_call).total_seconds())

            if wait_time > 0:
                print(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)

    def record_call(self, success: bool, error_type: str = None):
        """Record an API call for statistics."""
        with self.lock:
            now = datetime.now()

            # Update call tracking
            self.minute_calls.append(now)
            self.hour_calls.append(now)

            # Update statistics
            self.stats.total_calls += 1

            if success:
                self.stats.successful_calls += 1
            else:
                self.stats.failed_calls += 1

                if error_type == "quota_exceeded":
                    self.stats.quota_exceeded_count += 1
                elif error_type == "rate_limited":
                    self.stats.rate_limited_count += 1

            # Reset daily/hourly counters if needed
            if now - self.stats.last_reset_time >= timedelta(days=1):
                self.stats.daily_usage = 0
                self.stats.last_reset_time = now

            if now.hour == 0 and now.minute == 0:
                self.stats.hourly_usage = 0

            self.stats.daily_usage += 1
            self.stats.hourly_usage += 1

            # Save stats periodically
            if self.stats.total_calls % 10 == 0:
                self.save_stats()

    def calculate_backoff_delay(self, retry_count: int) -> float:
        """Calculate exponential backoff delay."""
        delay = self.base_delay * (self.backoff_multiplier ** retry_count)
        # Add jitter to avoid thundering herd
        import random
        jitter = random.uniform(0.1, 0.3) * delay
        return min(delay + jitter, self.max_delay)

    def should_retry(self, error: Exception, retry_count: int) -> bool:
        """Determine if an error should be retried."""
        if retry_count >= 3:  # Max retries
            return False

        error_str = str(error).lower()

        # Retry on rate limit and quota errors
        if any(keyword in error_str for keyword in [
            "rate limit", "quota exceeded", "429", "too many requests",
            "quota", "billing", "plan"
        ]):
            return True

        # Retry on temporary network errors
        if any(keyword in error_str for keyword in [
            "timeout", "connection", "network", "temporary", "try again"
        ]):
            return True

        return False

    def add_to_queue(self, func: Callable, *args, priority: int = 1, **kwargs) -> str:
        """Add an API call to the queue."""
        call_id = f"call_{int(time.time())}_{len(self.pending_queue)}"

        api_call = APICall(
            id=call_id,
            function=func,
            args=args,
            kwargs=kwargs,
            created_at=datetime.now(),
            priority=priority
        )

        self.pending_queue.append(api_call)
        self.pending_queue.sort(key=lambda x: x.priority)  # Sort by priority

        return call_id

    def process_queue(self):
        """Process the pending queue."""
        if self.processing_queue:
            return

        self.processing_queue = True

        try:
            while self.pending_queue:
                api_call = self.pending_queue[0]

                # Check if call is ready to retry
                if api_call.delay_until and datetime.now() < api_call.delay_until:
                    break

                try:
                    # Wait if rate limited
                    self.wait_if_needed()

                    # Execute the call
                    result = api_call.function(*api_call.args, **api_call.kwargs)

                    # Remove from queue and record success
                    self.pending_queue.pop(0)
                    self.record_call(success=True)

                    return result

                except Exception as e:
                    if self.should_retry(e, api_call.retry_count):
                        # Calculate delay and retry
                        delay = self.calculate_backoff_delay(api_call.retry_count)
                        api_call.delay_until = datetime.now() + timedelta(seconds=delay)
                        api_call.retry_count += 1

                        # Move to back of queue if there are other calls
                        if len(self.pending_queue) > 1:
                            self.pending_queue.pop(0)
                            self.pending_queue.append(api_call)

                        print(f"API call failed (attempt {api_call.retry_count}): {e}. Retrying in {delay:.1f}s...")
                        time.sleep(delay)

                    else:
                        # Remove from queue and record failure
                        self.pending_queue.pop(0)
                        error_type = "quota_exceeded" if "quota" in str(e).lower() else "other"
                        self.record_call(success=False, error_type=error_type)

                        print(f"API call failed permanently: {e}")

        finally:
            self.processing_queue = False

    def execute_with_retry(self, func: Callable, *args, priority: int = 1, **kwargs) -> Any:
        """Execute a function with retry logic."""
        # Check if we're rate limited first
        if self.is_rate_limited():
            # Add to queue instead of executing immediately
            call_id = self.add_to_queue(func, *args, priority=priority, **kwargs)
            self.process_queue()
            return None

        try:
            # Wait if needed (for rate limiting)
            self.wait_if_needed()

            # Execute the function
            result = func(*args, **kwargs)
            self.record_call(success=True)
            return result

        except Exception as e:
            if self.should_retry(e, 0):
                # Add to queue for retry
                call_id = self.add_to_queue(func, *args, priority=priority, **kwargs)
                self.process_queue()
                return None
            else:
                # Record permanent failure
                error_type = "quota_exceeded" if "quota" in str(e).lower() else "other"
                self.record_call(success=False, error_type=error_type)
                raise e

    def get_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return {
            "total_calls": self.stats.total_calls,
            "successful_calls": self.stats.successful_calls,
            "failed_calls": self.stats.failed_calls,
            "success_rate": (self.stats.successful_calls / max(1, self.stats.total_calls)) * 100,
            "quota_exceeded_count": self.stats.quota_exceeded_count,
            "rate_limited_count": self.stats.rate_limited_count,
            "daily_usage": self.stats.daily_usage,
            "hourly_usage": self.stats.hourly_usage,
            "pending_queue_size": len(self.pending_queue),
            "minute_calls": len(self.minute_calls),
            "hour_calls": len(self.hour_calls),
            "last_reset_time": self.stats.last_reset_time.isoformat() if self.stats.last_reset_time else None
        }

    def reset_stats(self):
        """Reset usage statistics."""
        self.stats = APIUsageStats()
        self.save_stats()

    def clear_queue(self):
        """Clear the pending queue."""
        self.pending_queue.clear()

    def set_rate_limits(self, max_per_minute: int, max_per_hour: int):
        """Update rate limits."""
        self.max_calls_per_minute = max_per_minute
        self.max_calls_per_hour = max_per_hour


# Global rate limiter instance - configured for Gemini free tier (2 requests per minute)
rate_limiter = RateLimiter(max_calls_per_minute=2, max_calls_per_hour=50)


def execute_with_rate_limit(func: Callable, *args, priority: int = 1, **kwargs) -> Any:
    """Convenience function to execute with rate limiting."""
    return rate_limiter.execute_with_retry(func, *args, priority=priority, **kwargs)


def get_api_stats() -> Dict[str, Any]:
    """Get API usage statistics."""
    return rate_limiter.get_stats()


def reset_api_stats():
    """Reset API usage statistics."""
    rate_limiter.reset_stats()


def clear_api_queue():
    """Clear the API queue."""
    rate_limiter.clear_queue()


def set_api_rate_limits(max_per_minute: int, max_per_hour: int):
    """Set API rate limits."""
    rate_limiter.set_rate_limits(max_per_minute, max_per_hour)