#!/bin/bash

# Excel Chat Agent Backend Development Server Startup Script

set -e

echo "🚀 Starting Excel Chat Agent Backend (Development Mode)"
echo ""

# Check if we're in the backend directory
if [[ ! -f "pyproject.toml" ]]; then
    echo "❌ Error: Please run this script from the backend directory"
    echo "   cd backend && ./start_dev.sh"
    exit 1
fi

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "❌ UV is not installed. Please install UV first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create data directory if it doesn't exist
mkdir -p data/excel_files
echo "📁 Data directory: $(pwd)/data/excel_files"

# Create chroma database directory
mkdir -p chroma_db
echo "🔍 Vector DB directory: $(pwd)/chroma_db"

# Check if virtual environment exists
if [[ ! -d ".venv" ]]; then
    echo "🔧 Creating virtual environment..."
    uv venv
fi

# Install/update dependencies
echo "📦 Installing dependencies..."
uv pip install -e ".[dev]"

# Create .env file if it doesn't exist
if [[ ! -f ".env" ]]; then
    echo "⚙️ Creating .env file from template..."
    cp .env.example .env
fi

echo ""
echo "✅ Backend setup complete!"
echo ""
echo "📊 Starting FastAPI development server..."
echo "   - API: http://localhost:8000"
echo "   - Docs: http://localhost:8000/docs"
echo "   - WebSocket: ws://localhost:8000/ws/{session_id}"
echo ""
echo "💡 Tips:"
echo "   - Place Excel files in: $(pwd)/data/excel_files/"
echo "   - Check server logs for processing updates"
echo "   - Use Ctrl+C to stop the server"
echo ""

# Start the development server
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level info