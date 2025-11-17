#!/bin/bash
# Start the backend server

cd "$(dirname "$0")"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

echo "Starting backend server on http://localhost:8000"
echo "API docs available at http://localhost:8000/docs"
echo ""
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
