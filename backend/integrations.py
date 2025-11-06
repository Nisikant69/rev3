# backend/integrations.py
"""
External integrations for the PR review bot.
Supports Slack, Microsoft Teams, Discord, and webhook notifications.
"""

import json
import requests
import asyncio
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
import os
import hmac
import hashlib
import base64

from config import DATA_DIR


@dataclass
class IntegrationConfig:
    """Configuration for external integrations."""
    service: str  # 'slack', 'teams', 'discord', 'webhook'
    webhook_url: str
    enabled: bool
    events: List[str]  # Events to notify about
    template: str  # Message template
    headers: Dict[str, str] = None
    secret: str = None


@dataclass
class NotificationPayload:
    """Standard payload for notifications."""
    title: str
    message: str
    url: str
    color: str  # For color coding (Slack)
    fields: List[Dict[str, Any]] = None


class ExternalIntegrations:
    """Manages external service integrations for notifications."""

    def __init__(self):
        self.integrations_file = DATA_DIR / "integrations" / "integrations.json"
        self.integrations_file.parent.mkdir(parents=True, exist_ok=True)
        self.integrations = {}
        self.notification_queue = []
        self.load_integrations()

    def load_integrations(self):
        """Load integration configurations from file."""
        try:
            if self.integrations_file.exists():
                with open(self.integrations_file, 'r') as f:
                    data = json.load(f)

                self.integrations = {
                    integration_id: IntegrationConfig(**config_data)
                    for integration_id, config_data in data.get('integrations', {}).items()
                }
        except Exception as e:
            print(f"Error loading integrations: {e}")
            self.create_default_integrations()

    def save_integrations(self):
        """Save integration configurations to file."""
        try:
            data = {
                'integrations': {
                    integration_id: asdict(config)
                    for integration_id, config in self.integrations.items()
                },
                'last_updated': datetime.now().isoformat()
            }

            with open(self.integrations_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving integrations: {e}")

    def create_default_integrations(self):
        """Create default integration configurations."""
        default_integrations = {
            "slack_default": IntegrationConfig(
                service="slack",
                webhook_url="",
                enabled=False,
                events=["review_completed", "batch_completed", "error_occurred"],
                template="🤖 **{{title}}**\n\n{{message}}",
                headers={"Content-Type": "application/json"},
                secret=""
            ),
            "teams_default": IntegrationConfig(
                service="teams",
                webhook_url="",
                enabled=False,
                events=["review_completed", "batch_completed", "error_occurred"],
                template="{{\"@type\": \"MessageCard\",\"text\": \"{{title}}\",\"text\": \"{{message}}\"}}",
                headers={"Content-Type": "application/json"}
            ),
            "discord_default": IntegrationConfig(
                service="discord",
                webhook_url="",
                enabled=False,
                events=["review_completed", "batch_completed", "error_occurred"],
                template="**{{title}}**\n\n{{message}}",
                headers={"Content-Type": "application/json"}
            ),
            "webhook_default": IntegrationConfig(
                service="webhook",
                webhook_url="",
                enabled=False,
                events=["review_completed", "batch_completed", "error_occurred"],
                template="{{\"title\":\"{{title}}\",\"message\":\"{{message}}\",\"url\":\"{{url}}\"}}",
                headers={"Content-Type": "application/json"}
            )
        }

        self.integrations = default_integrations
        self.save_integrations()

    def add_integration(self, config: IntegrationConfig, integration_id: str = None) -> str:
        """Add a new integration."""
        try:
            if integration_id is None:
                integration_id = f"{config.service}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Validate required fields
            if not config.webhook_url:
                raise ValueError("webhook_url is required for integrations")

            # Check for duplicate
            if integration_id in self.integrations:
                raise ValueError(f"Integration with ID '{integration_id}' already exists")

            self.integrations[integration_id] = config
            self.save_integrations()

            return f"Added {config.service.title()} integration with ID '{integration_id}'"

        except Exception as e:
            return f"Error adding integration: {e}"

    def remove_integration(self, integration_id: str) -> str:
        """Remove an integration."""
        try:
            if integration_id not in self.integrations:
                return f"Integration with ID '{integration_id}' not found"

            integration_name = self.integrations[integration_id].service.title
            del self.integrations[integration_id]
            self.save_integrations()

            return f"Removed {integration_name} integration"

        except Exception as e:
            return f"Error removing integration: {e}"

    def send_notification(self, integration_id: str, payload: NotificationPayload) -> bool:
        """Send a notification through the specified integration."""
        if integration_id not in self.integrations:
            print(f"Integration '{integration_id}' not found")
            return False

        config = self.integrations[integration_id]
        if not config.enabled:
            return False

        try:
            return self.send_by_service(config, payload)
        except Exception as e:
            print(f"Error sending notification via {config.service}: {e}")
            return False

    def send_by_service(self, config: IntegrationConfig, payload: NotificationPayload) -> bool:
        """Send notification using the appropriate service method."""
        if config.service == "slack":
            return self.send_slack_notification(config, payload)
        elif config.service == "teams":
            return self.send_teams_notification(config, payload)
        elif config.service == "discord":
            return self.send_discord_notification(config, payload)
        elif config.service == "webhook":
            return self.send_webhook_notification(config, payload)
        else:
            print(f"Unsupported service: {config.service}")
            return False

    def send_slack_notification(self, config: IntegrationConfig, payload: NotificationPayload) -> bool:
        """Send notification to Slack."""
        try:
            # Create Slack message
            slack_payload = {
                "text": payload.title,
                "attachments": [
                    {
                        "color": payload.color or "good",
                        "fields": payload.fields or []
                    }
                ]
            }

            # Add URL if provided
            if payload.url:
                slack_payload["attachments"][0]["actions"] = [
                    {
                        "type": "button",
                        "text": "View PR",
                        "url": payload.url
                    }
                ]

            # Send request
            response = requests.post(
                config.webhook_url,
                json=slack_payload,
                headers=config.headers,
                timeout=10
            )

            return response.status_code == 200

        except Exception as e:
            print(f"Error sending Slack notification: {e}")
            return False

    def send_teams_notification(self, config: IntegrationConfig, payload: NotificationPayload) -> bool:
        """Send notification to Microsoft Teams."""
        try:
            # Create Teams message card
            teams_payload = {
                "@type": "MessageCard",
                "@context": "https://schema.org/extensions",
                "theme": "default",
                "summary": payload.title,
                "sections": [
                    {
                        "activityTitle": payload.title,
                        "text": payload.message,
                        "potentialAction": [
                            {
                                "@type": "OpenUri",
                                "name": "View Pull Request",
                                "url": payload.url
                            }
                        ] if payload.url else []
                    }
                ]
            }

            # Send request
            response = requests.post(
                config.webhook_url,
                json=teams_payload,
                headers=config.headers,
                timeout=10
            )

            return response.status_code == 200

        except Exception as e:
            print(f"Error sending Teams notification: {e}")
            return False

    def send_discord_notification(self, config: IntegrationConfig, payload: NotificationPayload) -> bool:
        """Send notification to Discord."""
        try:
            # Create Discord embed message
            discord_payload = {
                "embeds": [
                    {
                        "title": payload.title,
                        "description": payload.message,
                        "color": self.get_discord_color(payload.color),
                        "url": payload.url
                    }
                ]
            }

            # Send request
            response = requests.post(
                config.webhook_url,
                json=discord_payload,
                headers=config.headers,
                timeout=10
            )

            return response.status_code == 200

        except Exception as e:
            print(f"Error sending Discord notification: {e}")
            return False

    def send_webhook_notification(self, config: IntegrationConfig, payload: NotificationPayload) -> bool:
        """Send notification to generic webhook."""
        try:
            # Create webhook payload
            webhook_payload = {
                "title": payload.title,
                "message": payload.message,
                "url": payload.url,
                "timestamp": datetime.now().isoformat(),
                "source": "ai-pr-review-bot"
            }

            # Add signature if secret is provided
            headers = config.headers or {}
            if config.secret:
                signature = self.generate_signature(json.dumps(webhook_payload), config.secret)
                headers["X-Signature"] = signature

            # Send request
            response = requests.post(
                config.webhook_url,
                json=webhook_payload,
                headers=headers,
                timeout=10
            )

            return response.status_code == 200

        except Exception as e:
            print(f"Error sending webhook notification: {e}")
            return False

    def generate_signature(self, payload: str, secret: str) -> str:
        """Generate HMAC signature for webhook verification."""
        signature = hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    def get_discord_color(self, color: str) -> int:
        """Convert color name to Discord color code."""
        color_map = {
            "good": 0x28a745,  # green
            "warning": 0xffaa00,  # yellow
            "danger": 0xdc3545, # red
            "info": 0x17a2b8,  # blue
            "primary": 0x3498db, # blue
            "secondary": 0x9b59c6,  # grey
        }
        return color_map.get(color.lower(), 0x36a64f)  # Default grey

    def queue_notification(self, integration_id: str, payload: NotificationPayload) -> str:
        """Queue a notification for later sending."""
        try:
            queued_notification = {
                "integration_id": integration_id,
                "payload": asdict(payload),
                "queued_at": datetime.now().isoformat(),
                "attempts": 0
            }

            self.notification_queue.append(queued_notification)
            return f"Queued notification for {integration_id}"

        except Exception as e:
            return f"Error queueing notification: {e}"

    def process_notification_queue(self) -> int:
        """Process queued notifications."""
        processed_count = 0
        failed_notifications = []

        while self.notification_queue:
            notification_data = self.notification_queue.pop(0)

            try:
                success = self.send_notification(
                    notification_data["integration_id"],
                    NotificationPayload(**notification_data["payload"])
                )

                if success:
                    processed_count += 1
                else:
                    # Retry logic
                    notification_data["attempts"] += 1
                    if notification_data["attempts"] < 3:
                        self.notification_queue.append(notification_data)
                    else:
                        failed_notifications.append(notification_data)

            except Exception as e:
                print(f"Error processing notification: {e}")
                failed_notifications.append(notification_data)

        return processed_count

    def create_notification_payload(self, title: str, message: str, url: str = "",
                               color: str = "good", fields: List[Dict[str, Any]] = None) -> NotificationPayload:
        """Create a standardized notification payload."""
        return NotificationPayload(
            title=title,
            message=message,
            url=url,
            color=color,
            fields=fields
        )

    def trigger_all_integrations(self, title: str, message: str, url: str = "",
                             color: str = "good") -> Dict[str, Any]:
        """Trigger notifications across all enabled integrations."""
        results = {}

        for integration_id, config in self.integrations.items():
            if config.enabled:
                try:
                    payload = self.create_notification_payload(title, message, url, color)
                    success = self.send_notification(integration_id, payload)
                    results[integration_id] = {
                        "success": success,
                        "service": config.service,
                        "webhook_url": config.webhook_url
                    }

                    if success:
                        results[integration_id]["sent_at"] = datetime.now().isoformat()
                    else:
                        results[integration_id]["error"] = "Failed to send"

                except Exception as e:
                    results[integration_id] = {
                        "success": False,
                        "service": config.service,
                        "error": str(e)
                    }

        return results


class ReviewNotifications:
    """Manages automated review notifications."""

    def __init__(self):
        self.integrations = ExternalIntegrations()
        self.enabled_integrations = [
            integration_id for integration_id, config in self.integrations.integrations.items() if config.enabled
        ]

    def notify_review_completed(self, repo_name: str, pr_number: int,
                             review_summary: str, pr_url: str,
                             issues_count: int, lenses_used: List[str] = None) -> Dict[str, Any]:
        """Notify about a completed review."""
        title = f"✅ Review Complete - PR #{pr_number} in {repo_name}"
        message = f"Review completed with {issues_count} issues found."

        if lenses_used:
            message += f"\nLenses applied: {', '.join(lenses_used)}"

        return self.trigger_all_integrations(
            title, message, pr_url
        )

    def notify_batch_completed(self, repo_name: str, job_count: int,
                               total_issues: int, processing_time: float) -> Dict[str, Any]:
        """Notify about batch processing completion."""
        title = f"📋 Batch Review Complete - {repo_name}"
        message = f"Successfully reviewed {job_count} PRs with {total_issues} total issues."

        if processing_time > 0:
            message += f"\nProcessing time: {processing_time:.1f}s"

        return self.trigger_all_integrations(
            title, message
        )

    def notify_error_occurred(self, repo_name: str, pr_number: int,
                          error_message: str, pr_url: str = "") -> Dict[str, Any]:
        """Notify about an error during review."""
        title = f"❌ Review Error - PR #{pr_number} in {repo_name}"
        message = f"An error occurred during review: {error_message}"

        if pr_url:
            message += f"\n[View PR]({pr_url})"

        return self.trigger_all_integrations(
            title, message, pr_url, color="danger"
        )

    def notify_security_issue(self, repo_name: str, pr_number: int,
                             issue_description: str, pr_url: str = "",
                             severity: str = "high") -> Dict[str, Any]:
        """Notify about a critical security issue."""
        title = f"🚨 Security Issue - PR #{pr_number} in {repo_name}"
        message = f"Critical security issue detected: {issue_description}"

        return self.trigger_all_integrations(
            title, message, pr_url, color="danger"
        )

    def notify_performance_issue(self, repo_name: str, pr_number: int,
                                issue_description: str, pr_url: str = "") -> Dict[str, Any]:
        """Notify about a performance issue."""
        title = f"⚡ Performance Issue - PR #{pr_number} in {repo_name}"
        message = f"Performance issue detected: {issue_description}"

        return self.trigger_all_integrations(
            title, message, pr_url, color="warning"
        )

    def notify_suggestion_applied(self, repo_name: str, pr_number: int,
                                    suggestion_count: int, pr_url: str = "") -> Dict[str, Any]:
        """Notify when code suggestions are available."""
        title = f"💡 Suggestions Available - PR #{pr_number} in {repo_name}"
        message = f"{suggestion_count} code suggestions are available for one-click application."

        return self.trigger_all_integrations(
            title, message, pr_url, color="good"
        )


# Global notification manager instance
notifications = ReviewNotifications()