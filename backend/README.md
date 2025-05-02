# RAG ChatBot Backend

A comprehensive backend system for a Retrieval-Augmented Generation (RAG) chatbot with web crawling capabilities.

## Features

- Web crawler that performs 1-hop crawling from seed URLs
- Content scraper and cleaner for extracting meaningful text from web pages
- Chunking system that divides content into appropriate segments for vectorization
- Vector database integration using PostgreSQL with pgvector extension
- Support for multiple LLM providers:
  - OpenAI
  - Anthropic
  - Google Gemini
  - Together AI
- FastAPI-based REST API with OpenAI-compatible endpoints

## Prerequisites

- Python 3.9+
- PostgreSQL 14+ with pgvector extension
- API keys for at least one of the supported LLM providers

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/ragchatbot.git
cd ragchatbot
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up PostgreSQL with pgvector:

```sql
CREATE EXTENSION vector;
```

5. Create a `.env` file with your configuration:

```env
# Database settings
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ragchatbot
DB_USER=postgres
DB_PASSWORD=your_password

# LLM API keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GEMINI_API_KEY=your_gemini_key
TOGETHER_API_KEY=your_together_key
```

## Running the Application

Start the FastAPI server:

```bash
python main.py
```

The API will be available at http://localhost:8000.

## API Endpoints

### 1. Build Index

Crawls and indexes web pages from the provided URLs.

```http
POST /build_index
Content-Type: application/json

{
  "urls": ["https://example.com", "https://another-site.com"]
}
```

Response:

```json
{
  "status": "success"
}
```

### 2. Chat

Sends a message to the chatbot and gets a response.

#### Basic Usage:

```http
POST /chat
Content-Type: application/json

{
  "prompt": "What is machine learning?"
}
```

Response:

```json
{
  "response": "Machine learning is a branch of artificial intelligence..."
}
```

#### Advanced Usage (OpenAI-compatible):

```http
POST /api/chat
Content-Type: application/json

{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"}
  ],
  "model": "gpt-4",
  "temperature": 0.7
}
```

## Architecture

The system consists of the following components:

1. **Crawler**: Fetches web pages and extracts links for 1-hop crawling.
2. **Scraper**: Extracts and cleans content from HTML pages.
3. **Processor**: Cleans and chunks text for efficient indexing.
4. **Database**: Stores web pages, text chunks, and their vector embeddings.
5. **LLM Integration**: Connects with various model providers for chat and embeddings.
6. **API**: Provides endpoints for indexing and chat functionality.

## Configuration

You can customize the behavior by modifying `config.py`. Key settings include:

- Database connection parameters
- Crawler settings (depth, timeout, headers)
- Chunking parameters (size, overlap)
- Default LLM provider and model settings

## License

MIT

