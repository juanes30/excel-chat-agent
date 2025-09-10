# Excel Chat Agent Backend

Excel Chat Agent backend service built with FastAPI, Ollama, ChromaDB, and LangChain.

## Features

- Process Excel files with automatic metadata extraction
- Semantic search using ChromaDB vectors
- Local LLM integration with Ollama
- Real-time chat with WebSockets
- Data visualization recommendations

## Development

```bash
# Install dependencies
uv pip install -e ".[dev]"

# Run tests
uv run pytest

# Start server
uv run uvicorn app.main:app --reload
```