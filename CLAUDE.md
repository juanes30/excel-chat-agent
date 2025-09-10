# Excel Chat Agent with Ollama and LangChain - Using UV and SuperClaude

## Project Overview
Create a complete chat agent system that can analyze and answer questions about Excel files using local LLM (Ollama), vector search (ChromaDB), and a modern React interface with WebSockets for real-time communication. This project uses UV for Python dependency management and SuperClaude framework for structured development.

## Tech Stack
- **Python Package Manager**: UV (fast, modern Python package manager)
- **Backend**: Python, FastAPI, LangChain, ChromaDB, Ollama
- **Frontend**: React.js, Tailwind CSS, WebSockets
- **LLM**: Ollama (local, no API costs)
- **Vector DB**: ChromaDB (open source)
- **Cache**: In-memory (Redis optional)
- **Development Framework**: SuperClaude for structured workflows

## Prerequisites
```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or on macOS/Linux with Homebrew
brew install uv

# Install SuperClaude
pipx install SuperClaude && SuperClaude install

# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3  # or mistral, phi, etc.
```

## Project Structure
```
excel-chat-agent/
├── backend/
│   ├── .python-version          # Python version for UV
│   ├── pyproject.toml           # UV project configuration
│   ├── uv.lock                  # UV lock file
│   ├── .venv/                   # UV virtual environment
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── models/
│   │   │   └── schemas.py
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── excel_processor.py
│   │   │   ├── vector_store.py
│   │   │   ├── llm_service.py
│   │   │   └── cache_service.py
│   │   └── websocket/
│   │       └── connection.py
│   ├── data/
│   │   └── excel_files/
│   ├── chroma_db/
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_excel_processor.py
│   │   ├── test_vector_store.py
│   │   └── test_llm_service.py
│   └── .env
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── components/
│   │   │   └── ExcelChatAgent.jsx
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.js
│   │   └── index.css
│   ├── package.json
│   ├── tailwind.config.js
│   └── postcss.config.js
├── docker-compose.yml
├── README.md
├── claude.md
└── start.sh
```

## SuperClaude Workflow Commands

Use these commands in Claude Code after installing SuperClaude:

### Initial Planning
```bash
/sc:brainstorm "Excel chat agent with Ollama, ChromaDB, FastAPI WebSockets and UV"
/sc:design "system architecture" --api --ddd --think-hard
@agent-system-architect "design integration between Excel processor, vector store and LLM"
```

### Backend Implementation
```bash
/sc:implement "UV Python project setup with FastAPI" --python
/sc:implement "Excel processor service with pandas" --python --tdd
/sc:implement "ChromaDB vector store service" --python --focus performance
/sc:implement "Ollama LLM service with LangChain" --python --async
/sc:implement "FastAPI WebSocket server" --api --websocket --security
```

### Frontend Implementation
```bash
/sc:implement "React chat component with WebSocket" --react --typescript
/sc:build --react --tailwind --responsive
```

### Testing & Optimization
```bash
/sc:test --coverage --e2e
/sc:analyze . --focus performance --think
/sc:troubleshoot "any issues" --think-hard --seq
```

## Implementation Steps

### Step 1: Initialize Backend with UV
Create the backend project structure using UV for modern Python dependency management.

Create `backend/pyproject.toml`:
```toml
[project]
name = "excel-chat-agent"
version = "0.1.0"
description = "Excel Chat Agent with Ollama and ChromaDB"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.104.1",
    "uvicorn[standard]>=0.24.0",
    "pandas>=2.1.3",
    "openpyxl>=3.1.2",
    "chromadb>=0.4.18",
    "sentence-transformers>=2.2.2",
    "langchain>=0.1.0",
    "langchain-community>=0.1.0",
    "ollama>=0.1.7",
    "websockets>=12.0",
    "python-multipart>=0.0.6",
    "pydantic>=2.5.0",
    "redis>=5.0.1",
    "numpy>=1.24.3",
    "python-jose[cryptography]>=3.3.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.3",
    "pytest-asyncio>=0.21.1",
    "pytest-cov>=4.1.0",
    "black>=23.11.0",
    "ruff>=0.1.6",
    "mypy>=1.7.0",
    "pre-commit>=3.5.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = [
    "pytest>=7.4.3",
    "pytest-asyncio>=0.21.1",
    "pytest-cov>=4.1.0",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.black]
line-length = 100
target-version = ["py311"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
```

Initialize UV project:
```bash
cd backend
echo "3.11" > .python-version
uv venv
uv pip sync
uv pip install -e ".[dev]"
```

### Step 2: Excel Processor Service
Create `backend/app/services/excel_processor.py` with:
- Class `ExcelProcessor` that processes Excel files efficiently
- Use UV-installed pandas for DataFrame operations
- Methods to extract metadata, statistics, and create text representations
- Support for multiple sheets with proper memory management
- Caching mechanism using Python's built-in lru_cache
- Methods: `process_excel_file()`, `process_all_files()`, `query_data()`
- Generate unique hash for each file using hashlib
- Extract column types, sample data, and statistics
- Handle large Excel files efficiently with chunking if needed

### Step 3: Vector Store Service
Create `backend/app/services/vector_store.py` with:
- Class `VectorStoreService` using ChromaDB
- Local embeddings with SentenceTransformer ('all-MiniLM-L6-v2')
- Async methods for better performance: `add_excel_data()`, `search()`, `search_by_file()`, `search_by_sheet()`
- Store documents with metadata (file_name, sheet_name, row/column counts)
- Implement batch processing for large datasets
- Semantic search capabilities with relevance scoring
- Clear and reindex functionality with progress tracking
- Connection pooling for ChromaDB

### Step 4: LLM Service with Ollama
Create `backend/app/services/llm_service.py` with:
- Class `LLMService` integrating Ollama and LangChain
- Multiple specialized prompts (data_analysis, sql_query, summary, comparison)
- Async/await pattern throughout for better concurrency
- Conversation memory management with Redis cache option
- Streaming response support for WebSocket using async generators
- Methods: `generate_response()`, `generate_streaming_response()`, `analyze_data_request()`
- Chart recommendation system for data visualization
- Response caching with TTL using Redis or in-memory
- Token counting and optimization
- Retry logic with exponential backoff for Ollama calls

### Step 5: Pydantic Models
Create `backend/app/models/schemas.py` with comprehensive type hints and validation:
```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

class ChartType(str, Enum):
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    HEATMAP = "heatmap"

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    file_filter: Optional[str] = None
    sheet_filter: Optional[str] = None
    max_results: Optional[int] = Field(default=5, ge=1, le=20)
    include_statistics: bool = False
    streaming: bool = True

class ChartData(BaseModel):
    type: ChartType
    data: List[Dict[str, Any]]
    config: Dict[str, Any]
    title: Optional[str] = None
    description: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float = Field(..., ge=0.0, le=1.0)
    chart_data: Optional[ChartData] = None
    timestamp: datetime
    processing_time_ms: Optional[int] = None
    tokens_used: Optional[int] = None

class FileInfo(BaseModel):
    file_name: str
    file_hash: str
    total_sheets: int
    total_rows: int
    total_columns: int
    file_size_mb: float
    last_modified: datetime
    
    @validator('file_size_mb')
    def validate_file_size(cls, v):
        if v > 100:
            raise ValueError('File size exceeds 100MB limit')
        return v

class WebSocketMessage(BaseModel):
    type: str
    content: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None
    session_id: Optional[str] = None

class SystemStats(BaseModel):
    total_files: int
    total_documents: int
    cache_size: int
    model_name: str
    vector_store_size: int
    uptime_seconds: int
```

### Step 6: FastAPI Main Application with UV
Create `backend/app/main.py` with modern async patterns:
- FastAPI app with CORS middleware configured for localhost:3000
- WebSocket endpoint at `/ws` for real-time chat with auto-reconnect
- REST endpoints with proper error handling and validation:
  - `GET /` - Health check with system metrics
  - `GET /api/files` - List available files with pagination
  - `POST /api/upload` - Upload new Excel file with virus scanning
  - `POST /api/query` - Query data with intelligent caching
  - `GET /api/stats` - Detailed system statistics
  - `POST /api/reindex` - Reindex all files with progress tracking
  - `DELETE /api/cache` - Clear cache
- ConnectionManager class for WebSocket management with heartbeat
- Startup event to process and index existing Excel files
- Graceful shutdown handling
- Response caching with TTL and smart invalidation
- Rate limiting per client
- Request ID tracking for debugging
- Structured logging with correlation IDs

### Step 7: Testing Setup with UV
Create comprehensive tests using pytest (installed via UV):

`backend/tests/test_excel_processor.py`:
```python
import pytest
import pandas as pd
from pathlib import Path
from app.services.excel_processor import ExcelProcessor

@pytest.fixture
def excel_processor():
    return ExcelProcessor("./test_data")

@pytest.fixture
def sample_excel_file(tmp_path):
    df = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'Score': [85.5, 92.0, 78.5]
    })
    file_path = tmp_path / "test.xlsx"
    df.to_excel(file_path, index=False)
    return file_path

@pytest.mark.asyncio
async def test_process_excel_file(excel_processor, sample_excel_file):
    result = await excel_processor.process_excel_file(sample_excel_file)
    assert result['file_name'] == 'test.xlsx'
    assert 'Sheet1' in result['sheets']
    assert result['total_rows'] == 3

# Add more comprehensive tests
```

Run tests with UV:
```bash
uv run pytest tests/ -v --cov=app --cov-report=html
```

### Step 8: Frontend Setup
Create `frontend/package.json` with modern React setup:
```json
{
  "name": "excel-chat-frontend",
  "version": "0.1.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1",
    "lucide-react": "^0.294.0",
    "recharts": "^2.10.3",
    "axios": "^1.6.2",
    "web-vitals": "^2.1.4",
    "@tanstack/react-query": "^5.0.0",
    "zustand": "^4.4.6"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "eslintConfig": {
    "extends": ["react-app"]
  },
  "browserslist": {
    "production": [">0.2%", "not dead", "not op_mini all"],
    "development": ["last 1 chrome version", "last 1 firefox version", "last 1 safari version"]
  },
  "devDependencies": {
    "tailwindcss": "^3.3.6",
    "autoprefixer": "^10.4.16",
    "postcss": "^8.4.32",
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "typescript": "^5.3.0"
  }
}
```

### Step 9: Tailwind Configuration
Create `frontend/tailwind.config.js`:
```javascript
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      animation: {
        'bounce': 'bounce 1s infinite',
        'pulse': 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'spin': 'spin 1s linear infinite',
      },
      colors: {
        'primary': '#3B82F6',
        'secondary': '#10B981',
        'accent': '#F59E0B',
      }
    },
  },
  plugins: [],
}
```

Create `frontend/postcss.config.js`:
```javascript
module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
```

### Step 10: React Chat Component with TypeScript
Create `frontend/src/components/ExcelChatAgent.tsx` with:
- Complete chat interface with sidebar and main chat area
- WebSocket connection with auto-reconnect and exponential backoff
- File upload with drag-and-drop and progress tracking
- Real-time message streaming with token-by-token display
- File selection for filtered queries
- System statistics display with auto-refresh
- Message history with user/assistant/system messages
- Loading states, error boundaries, and retry logic
- Responsive design with Tailwind CSS
- Icons from lucide-react
- State management with Zustand
- API calls with React Query for caching
- Virtual scrolling for large message lists
- Keyboard shortcuts for power users

### Step 11: Main React App
Create `frontend/src/App.tsx`:
```tsx
import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import ExcelChatAgent from './components/ExcelChatAgent';
import './App.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      cacheTime: 1000 * 60 * 10, // 10 minutes
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <div className="App">
        <ExcelChatAgent />
      </div>
    </QueryClientProvider>
  );
}

export default App;
```

### Step 12: Startup Script with UV
Create `start.sh` with UV support:
```bash
#!/bin/bash
set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting Excel Chat Agent with UV...${NC}"

# Check UV installation
if ! command -v uv &> /dev/null; then
    echo -e "${RED}UV is not installed. Installing...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Start Ollama
echo -e "${YELLOW}Starting Ollama...${NC}"
if ! pgrep -x "ollama" > /dev/null; then
    ollama serve &
    OLLAMA_PID=$!
    sleep 5
fi

# Check if model exists
if ! ollama list | grep -q "llama3"; then
    echo -e "${YELLOW}Pulling llama3 model...${NC}"
    ollama pull llama3
fi

# Backend setup with UV
echo -e "${YELLOW}Setting up backend with UV...${NC}"
cd backend

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    uv venv
fi

# Install/update dependencies
uv pip sync
uv pip install -e ".[dev]"

# Run database migrations if needed
# uv run alembic upgrade head

# Start backend
echo -e "${GREEN}Starting backend...${NC}"
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend
sleep 5

# Frontend setup
echo -e "${YELLOW}Setting up frontend...${NC}"
cd ../frontend

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    npm install
fi

# Start frontend
echo -e "${GREEN}Starting frontend...${NC}"
npm start &
FRONTEND_PID=$!

echo -e "${GREEN}✅ System started successfully!${NC}"
echo -e "📊 Frontend: http://localhost:3000"
echo -e "🔧 Backend API: http://localhost:8000"
echo -e "📚 API Docs: http://localhost:8000/docs"
echo -e "🤖 Ollama: http://localhost:11434"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down services...${NC}"
    kill $FRONTEND_PID 2>/dev/null
    kill $BACKEND_PID 2>/dev/null
    kill $OLLAMA_PID 2>/dev/null
    echo -e "${GREEN}Services stopped.${NC}"
    exit 0
}

# Set trap for cleanup
trap cleanup INT TERM

# Wait for all background processes
wait
```

### Step 13: Docker Compose with UV (Optional)
Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    command: serve
    networks:
      - excel-chat-network

  backend:
    build: 
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app
      - ./data:/app/data
      - uv-cache:/root/.cache/uv
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_HOST=redis
      - UV_SYSTEM_PYTHON=1
    depends_on:
      - ollama
      - redis
    command: uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    networks:
      - excel-chat-network

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - excel-chat-network

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app
      - /app/node_modules
    depends_on:
      - backend
    networks:
      - excel-chat-network

volumes:
  ollama_data:
  redis_data:
  uv-cache:

networks:
  excel-chat-network:
    driver: bridge
```

Create `backend/Dockerfile`:
```dockerfile
FROM python:3.11-slim

# Install UV
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY app/ ./app/

# Install dependencies with UV
RUN uv venv && \
    uv pip sync

# Expose port
EXPOSE 8000

# Run with UV
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 14: Development Workflow with UV

#### Daily Development Commands:
```bash
# Backend development with UV
cd backend

# Add new dependency
uv add pandas scikit-learn

# Add dev dependency
uv add --dev pytest-benchmark

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=app

# Format code
uv run black app/
uv run ruff check app/

# Type checking
uv run mypy app/

# Start dev server
uv run uvicorn app.main:app --reload

# Run specific script
uv run python scripts/migrate_data.py
```

#### Production Build:
```bash
# Build with UV
cd backend
uv build

# Create requirements.txt for compatibility
uv pip freeze > requirements.txt

# Package for distribution
uv build --wheel
```

### Step 15: README with UV Instructions
Create `README.md` with complete UV-based setup instructions:

```markdown
# Excel Chat Agent

A modern chat interface for querying Excel files using local LLM (Ollama), vector search (ChromaDB), and real-time WebSocket communication.

## Prerequisites

- Python 3.11+
- Node.js 18+
- UV (Python package manager)
- Ollama

## Quick Start with UV

### 1. Install UV
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and Setup
```bash
git clone <your-repo>
cd excel-chat-agent

# Backend setup with UV
cd backend
uv venv
uv pip sync
uv pip install -e ".[dev]"

# Frontend setup
cd ../frontend
npm install
```

### 3. Start Services
```bash
# From project root
./start.sh
```

### Development with UV

```bash
# Add dependencies
uv add package-name

# Run tests
uv run pytest

# Start backend
uv run uvicorn app.main:app --reload

# Update dependencies
uv pip sync
```

## Features

- 📊 Process Excel files with automatic metadata extraction
- 🔍 Semantic search using ChromaDB vectors
- 🤖 Local LLM integration with Ollama
- ⚡ Real-time chat with WebSockets
- 📈 Data visualization recommendations
- 🚀 Fast dependency management with UV
- 🔒 Secure file upload and processing

## Testing

```bash
cd backend
uv run pytest tests/ -v --cov=app
```

## Production Deployment

```bash
# Build with UV
uv build

# Or create traditional requirements
uv pip freeze > requirements.txt
```
```

## Key Features to Implement with UV

1. **Dependency Management**
   - Use UV for fast, deterministic dependency resolution
   - Lock file ensures reproducible builds
   - Parallel installation for speed
   - Built-in virtual environment management

2. **Development Workflow**
   - `uv run` for consistent script execution
   - `uv add` for adding dependencies
   - `uv sync` for team synchronization
   - Integration with pre-commit hooks

3. **Performance Optimizations**
   - UV's Rust-based resolver is 10-100x faster than pip
   - Aggressive caching reduces redundant downloads
   - Parallel operations throughout
   - Minimal memory footprint

4. **Testing Strategy**
   - Use `uv run pytest` for consistent test execution
   - Coverage reports with pytest-cov
   - Parallel test execution with pytest-xdist
   - Integration tests for WebSocket connections

## Testing Instructions with UV

1. Place test Excel files in `backend/data/excel_files/`
2. Start the system: `./start.sh`
3. Open http://localhost:3000
4. Run tests: `cd backend && uv run pytest`
5. Check coverage: `uv run pytest --cov=app --cov-report=html`

## Important Notes

- UV provides 10-100x faster dependency resolution than pip
- All Python dependencies are locked in `uv.lock` for reproducibility
- Use `uv sync` to ensure all team members have identical environments
- UV works seamlessly with existing pip packages
- The `.venv` is automatically created and managed by UV
- UV caches packages globally, saving disk space across projects

## Error Handling

- UV will automatically retry failed downloads
- Clear UV cache if needed: `uv cache clean`
- Check UV version: `uv --version`
- Update UV: `uv self-update`

## Performance Optimization with UV

- UV's parallel downloads speed up initial setup
- Reuse cached packages across projects
- Fast environment creation with copy-on-write (CoW) where supported
- Minimal overhead compared to traditional pip

## Security Considerations

- UV verifies package checksums automatically
- Lock file ensures no surprise updates
- Audit dependencies: `uv pip audit`
- Private registry support for enterprise packages

## SuperClaude Integration Points

When using SuperClaude commands, specify UV usage:
- `/sc:implement "Python service with UV" --python --modern`
- `/sc:test --python --uv`
- `@agent-python-expert "optimize UV dependency management"`

This system provides a complete, production-ready solution using UV for fast, reliable Python dependency management, making development and deployment significantly more efficient.