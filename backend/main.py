# main.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request, HTTPException
import hmac, hashlib, os, json, traceback
from github import Github, GithubException
import requests

from reviewer import review_patch_line_level
from auth import get_installation_token
from repo_fetcher import save_repo_snapshot
from context_indexer import index_repo
from config import GITHUB_WEBHOOK_SECRET as WEBHOOK_SECRET, MAX_DIFF_SIZE
from review_lenses import multi_lens_review, get_available_lenses
from summarizer import generate_pr_summary, format_summary_for_comment
from labeler import generate_pr_labels, apply_labels_to_pr, create_missing_labels
from suggestions import generate_suggestions_for_file
import re

app = FastAPI()


def is_review_command(comment: str) -> bool:
    """Check if a comment contains a review command."""
    comment_lower = comment.lower().strip()
    review_patterns = [
        r"^/review\s*$",
        r"^/review\s+",
        r"@review-bot",
        r"review\s+please",
        r"can\s+you\s+review"
    ]
    return any(re.match(pattern, comment_lower) for pattern in review_patterns)


def parse_review_command(comment: str) -> Dict[str, Any]:
    """Parse a review command to extract options."""
    comment_lower = comment.lower().strip()

    # Default options
    options = {
        "lenses": ["security", "performance", "best_practices"],
        "include_summary": True,
        "include_labels": True,
        "force_reindex": False
    }

    # Parse specific options
    if "security" in comment_lower:
        options["lenses"] = ["security"]
    elif "performance" in comment_lower:
        options["lenses"] = ["performance"]
    elif "best-practices" in comment_lower or "best_practices" in comment_lower:
        options["lenses"] = ["best_practices"]
    elif "lens:" in comment_lower:
        # Extract custom lens specification
        lens_match = re.search(r"lens:\s*([a-z_,\s]+)", comment_lower)
        if lens_match:
            lens_str = lens_match.group(1).replace(" ", "")
            options["lenses"] = [lens.strip() for lens in lens_str.split(",") if lens.strip()]

    if "no-summary" in comment_lower:
        options["include_summary"] = False

    if "no-labels" in comment_lower:
        options["include_labels"] = False

    if "reindex" in comment_lower:
        options["force_reindex"] = True

    return options


def handle_manual_review_request(payload: Dict[str, Any], comment: str):
    """Handle a manual review request from a PR comment."""
    try:
        # Extract PR information
        issue = payload.get("issue", {})
        repo_full_name = payload.get("repository", {}).get("full_name")
        pr_number = issue.get("number")
        comment_user = payload.get("comment", {}).get("user", {}).get("login")

        # Get GitHub connection
        installation_id = payload.get("installation", {}).get("id")
        access_token = get_installation_token(installation_id)
        g = Github(login_or_token=access_token)

        repo = g.get_repo(repo_full_name)
        pr = repo.get_pull(pr_number)

        print(f"🚀 Manual review requested by {comment_user} for PR #{pr_number}")

        # Parse review command options
        options = parse_review_command(comment)

        # Acknowledge the request
        pr.create_issue_comment(
            f"🤖 **AI Review Started**\n\n"
            f"Review requested by @{comment_user} with options: {', '.join(options['lenses'])} lenses.\n\n"
            f"⏳ Processing... This may take a few moments for large PRs."
        )

        # Perform the review
        perform_enhanced_review(pr, repo, options)

        # Post completion message
        pr.create_issue_comment(
            f"✅ **AI Review Complete**\n\n"
            f"Review completed with {', '.join(options['lenses'])} analysis.\n\n"
            f"Please check the review comments above for detailed feedback."
        )

    except Exception as e:
        print(f"🔥 Error in manual review: {e}")
        try:
            pr.create_issue_comment(
                f"❌ **AI Review Failed**\n\n"
                f"Sorry, I encountered an error while processing your review request:\n\n"
                f"```\n{str(e)}\n```\n\n"
                f"Please try again or contact the repository maintainers."
            )
        except:
            pass


def perform_enhanced_review(pr, repo, options: Dict[str, Any] = None):
    """Perform an enhanced review with multiple lenses."""
    if options is None:
        options = {
            "lenses": ["security", "performance", "best_practices"],
            "include_summary": True,
            "include_labels": True,
            "force_reindex": False
        }

    try:
        # Get PR information
        files = list(pr.get_files())
        head_sha = pr.head.sha
        repo_name = repo.full_name
        ref = pr.head.ref

        # Index repository (or use existing)
        if options.get("force_reindex", False):
            print("🔄 Forcing reindex...")
            # Clear existing index if needed
            import shutil
            from config import INDEX_DIR
            index_path = INDEX_DIR / f"{repo_name.replace('/', '__')}_{head_sha}.faiss"
            metadata_path = INDEX_DIR / f"{repo_name.replace('/', '__')}_{head_sha}_meta.json"
            if index_path.exists():
                index_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()

        repo_dir = save_repo_snapshot(repo, head_sha)
        index, metadata = index_repo(repo_dir, repo_name, head_sha)

        all_review_comments = []
        summary_blocks = []
        lens_summaries = []

        # Process each file with requested lenses
        for file in files:
            if file.status == "removed" or not file.patch:
                continue
            if len(file.patch) > MAX_DIFF_SIZE:
                continue

            # Run standard review
            review_result = review_patch_line_level(
                file.patch,
                file.filename,
                repo_name,
                ref,
                head_sha,
                index,
                metadata
            )

            # Run multi-lens analysis if requested
            if options.get("lenses"):
                # Get context for enhanced analysis
                from semantic_search import semantic_search
                context_chunks = semantic_search(
                    file.patch, index, metadata, repo_name, head_sha, file.filename
                )

                lens_result = multi_lens_review(
                    file.patch,
                    file.filename,
                    options["lenses"],
                    context_chunks
                )

                if lens_result.get("all_comments"):
                    # Convert lens comments to GitHub format
                    for comment in lens_result["all_comments"]:
                        if "line" in comment and comment["line"]:
                            # Ensure line is an integer, not a string
                            line_num = int(comment["line"]) if isinstance(comment["line"], str) else comment["line"]
                            comment_data = {
                                "path": comment["path"],
                                "body": comment["body"],
                                "line": line_num
                            }
                            # Don't include position field when using line
                            all_review_comments.append(comment_data)
                        else:
                            # Add as summary comment if no line number
                            summary_blocks.append(f"### {lens_result.get('summary', 'Multi-lens analysis')}")

                    lens_summaries.append(lens_result.get("summary", ""))

            # Add standard review comments
            if review_result and review_result.get("comments"):
                all_review_comments.extend(review_result["comments"])
                summary_blocks.append(f"### {review_result['summary']}")

        # Create PR summary if requested
        if options.get("include_summary", True):
            pr_summary = generate_pr_summary(files, repo_name, pr.title, pr.body)
            summary_comment = format_summary_for_comment(pr_summary)
            summary_blocks.insert(0, summary_comment)  # Add summary first

        # Apply labels if requested
        if options.get("include_labels", True):
            try:
                # Create missing labels first
                existing_labels = [label.name for label in pr.get_labels()]
                desired_labels = generate_pr_labels(files, pr.title, pr.body, existing_labels)

                if desired_labels.get("labels"):
                    create_missing_labels(repo, desired_labels["labels"])
                    apply_labels_to_pr(pr, desired_labels["labels"])

                    print(f"✅ Applied labels: {', '.join(desired_labels['labels'])}")
            except Exception as e:
                print(f"⚠️ Error applying labels: {e}")

        # Create the review
        if all_review_comments:
            # Generate overall summary
            overall_summary = f"🤖 **AI Code Review (Multi-Lens Analysis)**\n\n"
            overall_summary += f"Lenses used: {', '.join(options['lenses'])}\n\n"
            overall_summary += f"Found {len(all_review_comments)} issue{'s' if len(all_review_comments) != 1 else ''} across {len([f for f in files if f.status != 'removed' and f.patch and len(f.patch) <= MAX_DIFF_SIZE])} file{'s' if len(files) != 1 else ''}.\n\n"

            if lens_summaries:
                overall_summary += "## Lens Analysis Summary\n\n" + "\n".join(lens_summaries) + "\n\n"

            overall_summary += "\n---\n".join(summary_blocks)

            try:
                pr.create_review(
                    body=overall_summary,
                    event="COMMENT",
                    comments=all_review_comments,
                )
                print(f"✅ Created enhanced review with {len(all_review_comments)} comments using {', '.join(options['lenses'])} lenses")
            except GithubException as e:
                print(f"⚠️ Error creating review: {e}")
                # Fallback to comment-only review
                pr.create_review(
                    body=overall_summary + f"\n\n⚠️ Could not add line-specific comments due to API limitations.",
                    event="COMMENT",
                )
        else:
            # No specific comments, just summary
            if summary_blocks:
                overall_summary = "🤖 **AI Code Review**\n\n" + "\n---\n".join(summary_blocks)
                pr.create_review(
                    body=overall_summary,
                    event="COMMENT",
                )
            else:
                pr.create_review(
                    body="🤖 **AI Code Review**\n\nNo issues found!",
                    event="COMMENT",
                )

    except Exception as e:
        print(f"🔥 Error in enhanced review: {e}")
        raise

@app.post("/api/webhook")
async def github_webhook(request: Request):
    # --- Signature verification ---
    signature_header = request.headers.get("X-Hub-Signature-256")
    if not signature_header:
        raise HTTPException(status_code=400, detail="Missing signature header")
    body = await request.body()
    hash_object = hmac.new(WEBHOOK_SECRET.encode("utf-8"), body, hashlib.sha256)
    if not hmac.compare_digest(signature_header.split("=")[1], hash_object.hexdigest()):
        raise HTTPException(status_code=400, detail="Invalid signature")

    payload = await request.json()
    event = request.headers.get("X-GitHub-Event")

    if event == "pull_request":
        action = payload.get("action")
        if action in ["opened", "reopened", "synchronize"]:
            try:
                installation_id = payload["installation"]["id"]
                access_token = get_installation_token(installation_id)
                g = Github(login_or_token=access_token)

                repo_name = payload["repository"]["full_name"]
                repo = g.get_repo(repo_name)
                pr_number = payload["pull_request"]["number"]
                pr = repo.get_pull(pr_number)

                head_sha = payload["pull_request"]["head"]["sha"]
                ref = payload["pull_request"]["head"]["ref"]

                # --- Snapshot + Index (done once) ---
                repo_dir = save_repo_snapshot(repo, head_sha)
                index, metadata = index_repo(repo_dir, repo_name, head_sha)

                files = list(pr.get_files())
                all_review_comments = []
                action_flag = "COMMENT"
                summary_blocks = []

                for file in files:
                    if file.status == "removed" or not file.patch:
                        continue
                    if len(file.patch) > MAX_DIFF_SIZE:
                        continue

                    # Use the enhanced review system with all features
                    review_result = review_patch_line_level(
                        file.patch,
                        file.filename,
                        repo_name,
                        ref,
                        head_sha,
                        index,
                        metadata
                    )

                    if review_result and review_result.get("comments"):
                        # Add line-specific comments
                        all_review_comments.extend(review_result["comments"])
                        summary_blocks.append(f"### {review_result['summary']}")
                    elif review_result and review_result.get("summary"):
                        # No line comments but have summary
                        summary_blocks.append(f"### {review_result['summary']}")

                    # Generate multi-lens analysis for comprehensive review
                    lens_options = ["security", "performance", "best_practices"]
                    try:
                        # Get context for enhanced analysis
                        from semantic_search import semantic_search
                        context_chunks = semantic_search(
                            file.patch, index, metadata, repo_name, head_sha, file.filename
                        )

                        lens_result = multi_lens_review(
                            file.patch,
                            file.filename,
                            lens_options,
                            context_chunks
                        )

                        if lens_result.get("all_comments"):
                            # Convert lens comments to GitHub format
                            for comment in lens_result["all_comments"]:
                                if "line" in comment and comment["line"]:
                                    # Ensure line is an integer, not a string
                                    line_num = int(comment["line"]) if isinstance(comment["line"], str) else comment["line"]
                                    comment_data = {
                                        "path": comment["path"],
                                        "body": comment["body"],
                                        "line": line_num
                                    }
                                    # Don't include position field when using line
                                    all_review_comments.append(comment_data)
                                else:
                                    # Add as summary comment if no line number
                                    summary_blocks.append(f"### 🔍 {lens_result.get('summary', 'Multi-lens analysis')}")

                    except Exception as e:
                        print(f"⚠️ Error in multi-lens analysis for {file.filename}: {e}")

                    # Generate code suggestions if enabled
                    try:
                        suggestion_result = generate_suggestions_for_file(
                            file.patch, file.filename, context_chunks if 'context_chunks' in locals() else []
                        )

                        if suggestion_result.get("github_suggestions"):
                            # Add suggestions as review comments
                            for suggestion in suggestion_result["github_suggestions"]:
                                # Get the line number and ensure it's an integer
                                line_num = suggestion.get("line") or suggestion.get("position")
                                if line_num:
                                    line_num = int(line_num) if isinstance(line_num, str) else line_num
                                    comment_data = {
                                        "path": suggestion["path"],
                                        "body": f"💡 **Code Suggestion**\n\n{suggestion['body']}\n\n**Proposed Fix:**\n```suggestion\n{suggestion['suggestion']}\n```",
                                        "line": line_num
                                    }
                                    # Don't include position field when using line
                                    all_review_comments.append(comment_data)

                            summary_blocks.append(f"### 💡 Found {len(suggestion_result['github_suggestions'])} code suggestion{'s' if len(suggestion_result['github_suggestions']) != 1 else ''}")

                    except Exception as e:
                        print(f"⚠️ Error generating suggestions for {file.filename}: {e}")

                # Generate PR summary and labels
                try:
                    pr_summary = generate_pr_summary(files, repo_name, pr.title, pr.body)
                    summary_comment = format_summary_for_comment(pr_summary)
                    summary_blocks.insert(0, summary_comment)  # Add summary first

                    # Apply automatic labels
                    existing_labels = [label.name for label in pr.get_labels()]
                    label_result = generate_pr_labels(files, pr.title, pr.body, existing_labels)

                    if label_result.get("labels"):
                        create_missing_labels(repo, label_result["labels"])
                        apply_labels_to_pr(pr, label_result["labels"])
                        print(f"✅ Applied labels: {', '.join(label_result['labels'])}")

                except Exception as e:
                    print(f"⚠️ Error in summary/labeling: {e}")

                # Create comprehensive review with all features
                if all_review_comments:
                    # Generate enhanced overall summary
                    overall_summary = f"🤖 **AI Code Review - Comprehensive Analysis**\n\n"
                    overall_summary += f"📊 **Files Analyzed:** {len([f for f in files if f.status != 'removed' and f.patch and len(f.patch) <= MAX_DIFF_SIZE])}\n"
                    overall_summary += f"🔍 **Lenses Applied:** Security, Performance, Best Practices\n"
                    overall_summary += f"💡 **Issues Found:** {len(all_review_comments)} total\n\n"
                    overall_summary += "---\n" + "\n---\n".join(summary_blocks)

                    try:
                        pr.create_review(
                            body=overall_summary,
                            event="COMMENT",
                            comments=all_review_comments,
                        )
                        print(f"✅ Created review with {len(all_review_comments)} line-specific comments")
                    except GithubException as e:
                        print(f"⚠️ Error creating review with line comments: {e}")
                        # Fallback to comment-only review
                        pr.create_review(
                            body=overall_summary + f"\n\n⚠️ Could not add line-specific comments due to API limitations.",
                            event="COMMENT",
                        )
                elif summary_blocks:
                    # No line comments but have summaries
                    overall_summary = "🤖 **AI Code Review**\n\n" + "\n---\n".join(summary_blocks)
                    pr.create_review(
                        body=overall_summary,
                        event="COMMENT",
                    )
                else:
                    pr.create_review(
                        body="🤖 **AI Code Review**\n\nNo issues found!",
                        event="COMMENT",
                    )

            except (GithubException, requests.exceptions.RequestException) as e:
                print("⚠️ GitHub/Network error:", repr(e))
                traceback.print_exc()
                raise HTTPException(status_code=502, detail=f"GitHub API error: {e}")
            except Exception as e:
                print("🔥 Internal error in webhook:", repr(e))
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    elif event == "issue_comment":
        # Handle on-demand review triggers from PR comments
        action = payload.get("action")
        if action in ["created"]:
            try:
                comment = payload.get("comment", {}).get("body", "")
                issue = payload.get("issue", {})

                # Check if this is a PR comment (issues in PRs are also issues)
                if issue.get("pull_request"):
                    if is_review_command(comment):
                        handle_manual_review_request(payload, comment)
                    else:
                        # Check if it's a conversational query
                        from conversation import handle_conversational_comment, is_bot_mentioned
                        if is_bot_mentioned(comment, "review-bot[bot]"):  # Default bot username
                            try:
                                # Get PR context for conversation
                                pr_number = issue.get("number")
                                pr_context = {
                                    "pr_number": pr_number,
                                    "recent_reviews": [],
                                    "files": []
                                }

                                # Get bot's previous comments
                                pr = repo.get_pull(pr_number)
                                issue_comments = pr.get_issue_comments()
                                bot_comments = [c for c in issue_comments if c.user.login == "review-bot[bot]"]

                                # Generate conversational response
                                response = handle_conversational_comment(
                                    payload.get("comment", {}),
                                    pr_context,
                                    "review-bot[bot]",
                                    repo_name,
                                    pr.head.sha,
                                    index,
                                    metadata
                                )

                                if response:
                                    pr.create_issue_comment(response)
                                    print(f"💬 Responded to conversational query on PR #{pr_number}")

                            except Exception as e:
                                print(f"⚠️ Error handling conversational comment: {e}")

            except Exception as e:
                print(f"⚠️ Error handling issue comment: {e}")
                traceback.print_exc()

    return {"status": "success"}