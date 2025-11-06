# backend/conversation.py
"""
Conversational follow-up system for AI review bot Q&A.
Allows users to ask questions about review comments and get detailed explanations.
"""

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import google.generativeai as genai
from typing import List, Dict, Any, Optional, Tuple
from backend.utils import detect_language_from_filename
from backend.config import GEMINI_API_KEY
from backend.semantic_search import semantic_search, get_function_context, get_class_context
from backend.context_indexer import load_dependency_graph, get_related_files
from backend.api_rate_limiter import execute_with_rate_limit

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)


class ConversationManager:
    """Manages conversational interactions with the AI review bot."""

    def __init__(self):
        self.model = genai.GenerativeModel("gemini-2.5-pro")
        self.conversation_history = {}  # Store conversation history per PR

    def is_conversational_query(self, comment: str, bot_username: str = None) -> bool:
        """
        Check if a comment is a conversational query directed at the bot.

        Args:
            comment: The comment text
            bot_username: The bot's GitHub username (optional)

        Returns:
            True if this is a conversational query
        """
        comment_lower = comment.lower().strip()

        # Check for direct mentions
        if bot_username:
            if f"@{bot_username.lower()}" in comment_lower:
                return True

        # Check for question patterns
        question_indicators = [
            "why", "what", "how", "can you", "could you", "would you",
            "explain", "clarify", "elaborate", "more detail"
        ]

        return any(indicator in comment_lower for indicator in question_indicators)

    def is_reply_to_bot(self, comment: Dict[str, Any], bot_comments: List[Dict[str, Any]]) -> bool:
        """
        Check if a comment is a reply to a bot comment.

        Args:
            comment: The comment data
            bot_comments: List of bot comments

        Returns:
            True if this is a reply to a bot comment
        """
        # Check if comment is replying to a bot comment
        if comment.get("in_reply_to_id"):
            reply_to_id = comment["in_reply_to_id"]
            return any(bot_comment.get("id") == reply_to_id for bot_comment in bot_comments)

        return False

    def extract_context_from_question(self, comment: str, pr_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant context from a user's question.

        Args:
            comment: The user's comment/question
            pr_context: PR context including files and previous reviews

        Returns:
            Extracted context for answering the question
        """
        comment_lower = comment.lower()

        context = {
            "question_type": "general",
            "referenced_file": None,
            "referenced_function": None,
            "referenced_line": None,
            "question_text": comment.strip()
        }

        # Extract file references
        import re
        file_patterns = [
            r"in (\S+\.(py|js|ts|java|cpp|c|go|rs|rb|php|swift|kt|scala|html|css|sql|sh|yaml|yml|json|xml|md|txt))",
            r"file (\S+\.(py|js|ts|java|cpp|c|go|rs|rb|php|swift|kt|scala|html|css|sql|sh|yaml|yml|json|xml|md|txt))",
            r"(\S+\.(py|js|ts|java|cpp|c|go|rs|rb|php|swift|kt|scala|html|css|sql|sh|yaml|yml|json|xml|md|txt))"
        ]

        for pattern in file_patterns:
            match = re.search(pattern, comment_lower)
            if match:
                context["referenced_file"] = match.group(1)
                break

        # Extract function/method references
        func_patterns = [
            r"function (\w+)",
            r"method (\w+)",
            r"(\w+)\(\)",
            r"in (\w+)\("
        ]

        for pattern in func_patterns:
            match = re.search(pattern, comment_lower)
            if match:
                context["referenced_function"] = match.group(1)
                break

        # Extract line references
        line_patterns = [
            r"line (\d+)",
            r"(\d+)th line",
            r"at line (\d+)"
        ]

        for pattern in line_patterns:
            match = re.search(pattern, comment_lower)
            if match:
                context["referenced_line"] = match.group(1)
                break

        # Determine question type
        if "why" in comment_lower:
            context["question_type"] = "why"
        elif "how" in comment_lower:
            context["question_type"] = "how"
        elif "what" in comment_lower:
            context["question_type"] = "what"
        elif "fix" in comment_lower or "solve" in comment_lower:
            context["question_type"] = "solution"

        return context

    def generate_response(self, question: str, context: Dict[str, Any], pr_context: Dict[str, Any],
                          repo_name: str = None, commit_sha: str = None, index=None, metadata=None) -> str:
        """
        Generate a response to a user's question.

        Args:
            question: The user's question
            context: Extracted context from the question
            pr_context: PR context
            repo_name: Repository name
            commit_sha: Commit SHA
            index: Search index
            metadata: Search metadata

        Returns:
            Generated response
        """
        prompt = self.create_conversation_prompt(question, context, pr_context, repo_name, commit_sha, index, metadata)

        try:
            response = self.model.generate_content(prompt)
            if response and response.text:
                return self.format_response(response.text, context)
        except Exception as e:
            print(f"Error generating conversation response: {e}")

        return self.generate_fallback_response(question, context)

    def create_conversation_prompt(self, question: str, context: Dict[str, Any], pr_context: Dict[str, Any],
                                   repo_name: str, commit_sha: str, index, metadata) -> str:
        """Create a prompt for generating conversational responses."""
        # Build context information
        relevant_code = ""
        additional_context = ""

        if context.get("referenced_file") and index and metadata:
            # Search for relevant code chunks
            query = context["referenced_file"]
            if context.get("referenced_function"):
                query += f" {context['referenced_function']}"

            try:
                search_results = semantic_search(query, index, metadata, repo_name, commit_sha)
                if search_results:
                    relevant_code = "\n".join([f"File: {result['file']}\n```{result['content'][:500]}```" for result in search_results[:3]])
            except Exception as e:
                print(f"Error searching for context: {e}")

        # Get additional context about the original review
        original_reviews = pr_context.get("recent_reviews", [])
        if original_reviews:
            additional_context = "Original review comments:\n"
            for review in original_reviews[-3:]:  # Last 3 reviews
                additional_context += f"- {review.get('body', '')[:200]}...\n"

        prompt = f"""
You are an AI code review assistant answering a developer's question about a code review you provided.

Question: {question}

Context:
- Question type: {context.get('question_type', 'general')}
- Referenced file: {context.get('referenced_file', 'None')}
- Referenced function: {context.get('referenced_function', 'None')}
- Referenced line: {context.get('referenced_line', 'None')}

{additional_context}

{relevant_code}

Guidelines for your response:
1. Be helpful and educational
2. Explain the reasoning behind your original review comment
3. Provide specific examples when possible
4. Suggest concrete solutions or alternatives
5. Keep your response concise but thorough
6. Use a friendly, professional tone

If the question asks "why" something is an issue, explain the security, performance, or maintainability implications.
If the question asks "how" to fix something, provide specific code examples.
If the question asks "what" something means, provide a clear explanation.

Your response should be informative and help the developer understand and improve their code.
"""

        return prompt

    def format_response(self, response: str, context: Dict[str, Any]) -> str:
        """Format the AI response for GitHub comments."""
        response_lines = response.strip().split('\n')
        formatted_response = f"🤖 **Answer to your question**\n\n"

        # Add contextual reference if available
        if context.get("referenced_file"):
            formatted_response += f"**Regarding:** {context['referenced_file']}"
            if context.get("referenced_function"):
                formatted_response += f" → {context['referenced_function']}()"
            if context.get("referenced_line"):
                formatted_response += f" (Line {context['referenced_line']})"
            formatted_response += "\n\n"

        # Add the response content
        formatted_response += "\n".join(response_lines)

        # Add a helpful closing
        formatted_response += "\n\n---\n\n*💡 This response is generated based on the code context and review analysis. Feel free to ask follow-up questions!*"

        return formatted_response

    def generate_fallback_response(self, question: str, context: Dict[str, Any]) -> str:
        """Generate a fallback response when AI generation fails."""
        return f"""🤖 **Answer to your question**\n\n
I'm sorry, I encountered an issue generating a detailed response to your question about: "{question[:100]}..."

**Context:**
- File: {context.get('referenced_file', 'Not specified')}
- Function: {context.get('referenced_function', 'Not specified')}
- Line: {context.get('referenced_line', 'Not specified')}

**General Guidance:**
- If you're asking about a specific issue I mentioned, it's typically based on security, performance, or code quality best practices
- For implementation details, consider the language-specific patterns and frameworks you're using
- Feel free to ask more specific questions or provide more context

Please try rephrasing your question or contact the repository maintainers for more detailed assistance.

*💡 This is a fallback response. The detailed analysis may be temporarily unavailable.*
"""

    def should_respond_to_comment(self, comment: Dict[str, Any], bot_username: str = None,
                                  bot_comments: List[Dict[str, Any]] = None) -> bool:
        """
        Determine if the bot should respond to a comment.

        Args:
            comment: The comment to evaluate
            bot_username: The bot's username
            bot_comments: Previous bot comments

        Returns:
            True if the bot should respond
        """
        comment_text = comment.get("body", "")

        # Check if it's a direct question to the bot
        if self.is_conversational_query(comment_text, bot_username):
            return True

        # Check if it's a reply to a bot comment
        if bot_comments and self.is_reply_to_bot(comment, bot_comments):
            return True

        return False

    def get_conversation_context(self, pr_number: int, repo_name: str) -> Dict[str, Any]:
        """
        Get conversation context for a specific PR.

        Args:
            pr_number: The PR number
            repo_name: The repository name

        Returns:
            Conversation context including history
        """
        context_key = f"{repo_name}_{pr_number}"
        return self.conversation_history.get(context_key, {
            "questions_asked": [],
            "topics_discussed": [],
            "last_interaction": None
        })

    def update_conversation_history(self, pr_number: int, repo_name: str, question: str, response: str):
        """
        Update conversation history for a PR.

        Args:
            pr_number: The PR number
            repo_name: The repository name
            question: The user's question
            response: The bot's response
        """
        context_key = f"{repo_name}_{pr_number}"

        if context_key not in self.conversation_history:
            self.conversation_history[context_key] = {
                "questions_asked": [],
                "topics_discussed": [],
                "last_interaction": None
            }

        history = self.conversation_history[context_key]
        history["questions_asked"].append(question)
        history["last_interaction"] = {
            "question": question,
            "response": response,
            "timestamp": json.dumps({"timestamp": "now"})  # Simplified timestamp
        }

        # Keep only last 10 interactions to avoid memory issues
        if len(history["questions_asked"]) > 10:
            history["questions_asked"] = history["questions_asked"][-10:]

    def create_conversation_summary(self, pr_number: int, repo_name: str) -> str:
        """
        Create a summary of the conversation for the PR.

        Args:
            pr_number: The PR number
            repo_name: The repository name

        Returns:
            Conversation summary
        """
        context = self.get_conversation_context(pr_number, repo_name)
        questions = context.get("questions_asked", [])

        if not questions:
            return "No conversational interactions recorded."

        summary = f"## 📝 Conversation Summary\n\n"
        summary += f"**Total questions asked:** {len(questions)}\n\n"

        if questions:
            summary += "**Recent questions:**\n"
            for i, question in enumerate(questions[-5:], 1):  # Last 5 questions
                summary += f"{i}. {question[:100]}{'...' if len(question) > 100 else ''}\n"

        return summary


def handle_conversational_comment(comment: Dict[str, Any], pr_context: Dict[str, Any],
                                  bot_username: str = None, repo_name: str = None,
                                  commit_sha: str = None, index=None, metadata=None) -> Optional[str]:
    """
    Handle a conversational comment and generate a response.

    Args:
        comment: The comment data
        pr_context: PR context
        bot_username: The bot's username
        repo_name: Repository name
        commit_sha: Commit SHA
        index: Search index
        metadata: Search metadata

    Returns:
        Generated response or None
    """
    conversation_manager = ConversationManager()

    # Check if we should respond
    if not conversation_manager.should_respond_to_comment(comment, bot_username):
        return None

    comment_text = comment.get("body", "")

    # Extract context from the question
    context = conversation_manager.extract_context_from_question(comment_text, pr_context)

    # Generate response
    response = conversation_manager.generate_response(
        comment_text,
        context,
        pr_context,
        repo_name,
        commit_sha,
        index,
        metadata
    )

    # Update conversation history
    pr_number = pr_context.get("pr_number", 0)
    if pr_number and repo_name:
        conversation_manager.update_conversation_history(pr_number, repo_name, comment_text, response)

    return response


def is_bot_mentioned(comment_text: str, bot_username: str) -> bool:
    """
    Check if the bot is mentioned in a comment.

    Args:
        comment_text: The comment text
        bot_username: The bot's username

    Returns:
        True if bot is mentioned
    """
    if not bot_username:
        return False

    return f"@{bot_username.lower()}" in comment_text.lower()


def extract_question_intent(comment_text: str) -> str:
    """
    Extract the intent of a user's question.

    Args:
        comment_text: The comment text

    Returns:
        Question intent (why, how, what, etc.)
    """
    comment_lower = comment_text.lower()

    if "why" in comment_lower:
        return "why"
    elif "how" in comment_lower:
        return "how"
    elif "what" in comment_lower:
        return "what"
    elif "fix" in comment_lower or "solve" in comment_lower:
        return "solution"
    elif "explain" in comment_lower or "clarify" in comment_lower:
        return "explanation"
    else:
        return "general"