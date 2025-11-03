# AI PR Review Bot - Setup Guide

## Overview

This enhanced PR review bot provides line-level code reviews with intelligent analysis, automatic labeling, and contextual understanding. It now supports:

- **Line-Level Comments**: Precise feedback anchored to exact code positions
- **Intelligent Summarization**: AI-powered PR descriptions and change analysis
- **Auto-Labeling**: Smart categorization based on file patterns and content
- **Enhanced Context**: Dependency-aware analysis with function-level chunking
- **Large PR Handling**: Intelligent chunking to manage context windows
- **Multiple Review Lenses**: Security, performance, and best practices analysis

## Prerequisites

1. **Python 3.11+** installed
2. **Git** installed
3. **GitHub Account** with repository access
4. **ngrok** account (free tier sufficient for local testing)
5. **Google Cloud Project** with Gemini API enabled

## Step 1: GitHub App Setup

### 1.1 Create GitHub App
1. Go to GitHub Settings → Developer settings → GitHub Apps → New GitHub App
2. **App name**: `AI PR Review Bot` (or your preferred name)
3. **Homepage URL**: `https://your-repo-url.com`
4. **Webhook URL**: `https://your-ngrok-url.ngrok.io/api/webhook` (update later with ngrok URL)

### 1.2 Configure Permissions
- **Repository permissions**:
  - **Pull requests**: Read & write
  - **Contents**: Read
  - **Issues**: Read & write (for conversations)
  - **Metadata**: Read

### 1.3 Subscribe to Events
- **Pull request**
- **Issue comment** (for conversational features)

### 1.4 Generate Private Key
1. Click "Generate a private key"
2. Download and save the `.pem` file securely
3. Note the **App ID** displayed

### 1.5 Install App
1. Install on your target repositories
2. Note the **Installation ID** (visible in URL after installation)

## Step 2: Google Gemini API Setup

### 2.1 Create Google Cloud Project
1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create new project or select existing

### 2.2 Enable Gemini API
1. Go to APIs & Services → Library
2. Search for "Generative AI API" or "Gemini API"
3. Click Enable

### 2.3 Create API Key
1. Go to APIs & Services → Credentials
2. Click "Create Credentials" → API Key
3. Copy and secure the API key

## Step 3: Local Development Setup

### 3.1 Clone and Setup Repository
```bash
git clone <your-repository-url>
cd rev3/backend
```

### 3.2 Install Dependencies
```bash
pip install -r requirements.txt
```

### 3.3 Create Environment File
```bash
cp .env.example .env
```

### 3.4 Configure .env File
Edit `.env` with your configuration:

```bash
# GitHub Configuration
GITHUB_WEBHOOK_SECRET=your_random_webhook_secret_here_change_this_in_production
GITHUB_APP_ID=your_github_app_id_here
GITHUB_PRIVATE_KEY=./path/to/your/private_key.pem

# AI Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Processing Limits (adjust based on your needs)
MAX_DIFF_SIZE=15000
MAX_TOKENS_PER_REQUEST=8000
TOP_K=5
MIN_CHUNK_LINES=10

# Feature Flags
ENABLE_AUTO_LABELING=true
ENABLE_CODE_SUGGESTIONS=true
ENABLE_CONVERSATION=true
ENABLE_SUMMARIZATION=true
PREFER_FUNCTION_BOUNDARIES=true

# ngrok Configuration (optional)
NGROK_AUTH_TOKEN=your_ngrok_auth_token_here
```

### 3.5 Verify Setup
```bash
# Test GitHub App authentication
python -c "from backend.auth import get_installation_token; print('Auth working')"

# Test Gemini API
python -c "import google.generativeai as genai; print('Gemini working')"
```

## Step 4: ngrok Setup

### 4.1 Install ngrok
```bash
# macOS
brew install ngrok

# Windows
# Download from https://ngrok.com/download

# Linux
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar xvzf ngrok-v3-stable-linux-amd64.tgz
```

### 4.2 Configure ngrok (Optional)
```bash
# Add auth token for persistent sessions
ngrok config add-authtoken YOUR_NGROK_AUTH_TOKEN
```

### 4.3 Start ngrok
```bash
ngrok http 8000
```

### 4.4 Copy ngrok URL
- ngrok will display a URL like `https://abc123.ngrok.io`
- Copy this URL for GitHub webhook configuration

## Step 5: Update GitHub Webhook

1. Go back to GitHub App settings
2. **Update Webhook URL**: Paste your ngrok URL with `/api/webhook`
   - Example: `https://abc123.ngrok.io/api/webhook`
3. **Set Webhook Secret**: Use the same value as `GITHUB_WEBHOOK_SECRET` in your .env
4. **Save changes**

## Step 6: Run the Application

### 6.1 Start the FastAPI Server
```bash
cd rev3/backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 6.2 Verify Webhook Reception
- Check console output for webhook events
- Create a test PR in your repository
- Watch for "Webhook received" messages

### 6.3 Test Line-Level Reviews
- Create a PR with code changes
- The bot should post line-specific comments instead of a single large comment

## Step 7: Production Deployment

### Option A: Docker Deployment (Recommended)

#### 7.1 Create Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/indexes data/repo_cache

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 7.2 Build and Run
```bash
docker build -t pr-review-bot .
docker run -d -p 8000:8000 --env-file .env pr-review-bot
```

### Option B: Cloud Service Deployment

#### 7.1 Create start.sh
```bash
#!/bin/bash
uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
chmod +x start.sh
```

#### 7.2 Deploy to Your Preferred Cloud
- Update `requirements.txt` with all dependencies
- Configure environment variables in cloud service dashboard
- Deploy using Vercel, Render, Railway, or similar

## Key Features Now Available

### ✅ Line-Level Comments
- Comments are now anchored to exact line positions in diffs
- AI provides specific feedback for individual code lines
- Developers see feedback right next to their code

### ✅ Intelligent Summarization
- AI generates PR descriptions and summaries
- Automatic change categorization (Added, Fixed, Changed, etc.)
- Risk assessment and testing recommendations

### ✅ Auto-Labeling
- Automatic PR labeling based on file patterns and content
- Labels for languages, file types, and change categories
- Size-based labeling (small, medium, large PRs)

### ✅ Enhanced Context Understanding
- Function-level code chunking for better analysis
- Dependency-aware context retrieval
- Cross-file relationship analysis

### ✅ Large PR Handling
- Intelligent chunking to manage AI context limits
- Processing by function boundaries when possible
- Graceful handling of very large PRs

### ✅ Multi-Lens Analysis Framework
- Modular system for different review perspectives
- Security, performance, and best practices lenses
- Extensible architecture for custom review criteria

## Configuration Options

### Processing Limits
- `MAX_DIFF_SIZE`: Maximum diff size to process (default: 15000)
- `MAX_TOKENS_PER_REQUEST`: AI token limit per request (default: 8000)
- `TOP_K`: Number of context chunks to retrieve (default: 5)
- `MIN_CHUNK_LINES`: Minimum lines per chunk (default: 10)

### Feature Flags
- `ENABLE_AUTO_LABELING`: Enable automatic PR labeling (default: true)
- `ENABLE_CODE_SUGGESTIONS`: Enable code suggestions (default: true)
- `ENABLE_CONVERSATION`: Enable conversational follow-up (default: true)
- `ENABLE_SUMMARIZATION`: Enable AI summarization (default: true)
- `PREFER_FUNCTION_BOUNDARIES`: Use function-level chunking (default: true)

## Troubleshooting

### Common Issues

**Issue 1: Line-level comments not appearing**
- **Solution**: Check diff parser is correctly mapping line numbers to diff positions

**Issue 2: GitHub App permissions error**
- **Solution**: Verify the App has proper permissions and is installed on the repository

**Issue 3: Gemini API quota exceeded**
- **Solution**: Enable billing in Google Cloud Console or implement rate limiting

**Issue 4: Large PRs cause timeout**
- **Solution**: The bot now handles large PRs with intelligent chunking

**Issue 5: ngrok connection issues**
- **Solution**: Use a paid ngrok plan for persistent URLs, or deploy to a cloud service

### Debug Commands

```bash
# Test webhook payload manually
curl -X POST http://localhost:8000/api/webhook \
  -H "Content-Type: application/json" \
  -H "X-Hub-Signature-256: sha256=your_signature" \
  -d '{"zen":"Non-blocking is better than blocking."}'

# Check GitHub App authentication
python -c "from backend.auth import build_app_jwt; print('JWT Token:', build_app_jwt())"

# Test semantic search
python -c "from backend.semantic_search import semantic_search; print('Search working')"
```

## Performance Optimization

### Recommended Settings

**For Small Teams (< 50 developers):**
- `MAX_DIFF_SIZE = 10000`
- `MAX_TOKENS_PER_REQUEST = 6000`
- Process all PR types automatically

**For Medium Teams (50-200 developers):**
- `MAX_DIFF_SIZE = 15000` (default)
- `MAX_TOKENS_PER_REQUEST = 8000` (default)
- Use manual triggers for draft PRs

**For Large Teams (> 200 developers):**
- `MAX_DIFF_SIZE = 20000`
- `MAX_TOKENS_PER_REQUEST = 10000`
- Implement queuing system for concurrent PRs

## Monitoring and Maintenance

### Add Logging
```python
# Add to main.py for better debugging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log key events
logger.info(f"Webhook received: {event} {action}")
logger.info(f"Review generated for PR #{pr_number}")
```

### Monitor Resources
- Check GitHub API rate limits
- Monitor Gemini API usage in Google Cloud Console
- Watch application logs for errors

### Regular Maintenance
- Clear old cached indexes: `rm -rf data/indexes/*`
- Update dependencies regularly
- Monitor for GitHub API changes
- Backup configuration files

## Security Considerations

### Data Protection
1. **API Keys**: Never commit API keys to version control
2. **Code Privacy**: Consider using on-premise AI models for sensitive code
3. **Access Control**: Limit GitHub App permissions to minimum required

### Auditing and Compliance
1. **Review Logging**: Log all review activities for audit trails
2. **Data Handling**: Ensure compliance with data protection regulations
3. **Access Logs**: Monitor who is accessing the system

## Next Steps

The core enhancement is now complete! Your PR review bot now provides:

1. **Actionable line-level comments** instead of large issue comments
2. **Intelligent PR analysis** with summaries and labeling
3. **Enhanced context understanding** with dependency-aware analysis
4. **Robust handling of large PRs** with intelligent chunking

Future enhancements could include:
- On-demand review triggers (/review commands)
- Code suggestions with GitHub's suggestion API
- Conversational follow-up capabilities
- Additional review lenses for specific domains

**Generated with Compyle**