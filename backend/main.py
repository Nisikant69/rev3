# main.py
from fastapi import FastAPI, Request, HTTPException
import hmac, hashlib, os, json, traceback
from github import Github, GithubException
import requests

from backend.reviewer import review_patch_line_level
from backend.auth import get_installation_token
from backend.repo_fetcher import save_repo_snapshot
from backend.context_indexer import index_repo
from backend.config import GITHUB_WEBHOOK_SECRET as WEBHOOK_SECRET, MAX_DIFF_SIZE

app = FastAPI()

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

                files = pr.get_files()
                all_review_comments = []
                action_flag = "COMMENT"
                summary_blocks = []

                for file in files:
                    if file.status == "removed" or not file.patch:
                        continue
                    if len(file.patch) > MAX_DIFF_SIZE:
                        continue

                    # Use the new line-level review system
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

                # Create review with line-specific comments
                if all_review_comments:
                    # Generate overall summary
                    overall_summary = f"🤖 **AI Code Review**\n\nFound {len(all_review_comments)} issue{'s' if len(all_review_comments) != 1 else ''} across {len([f for f in files if f.status != 'removed' and f.patch and len(f.patch) <= MAX_DIFF_SIZE])} file{'s' if len(files) != 1 else ''}.\n\n"
                    overall_summary += "\n---\n".join(summary_blocks)

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

    return {"status": "success"}