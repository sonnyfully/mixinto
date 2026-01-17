#!/bin/bash
# Development script to run both Flask backend and React frontend

echo "Starting MixInto development environment..."
echo ""
echo "This will start:"
echo "  1. Flask backend on a free port (will be displayed)"
echo "  2. React frontend dev server on http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Start Flask backend in background
cd "$(dirname "$0")/.."
python3 scripts/start_web.py &
FLASK_PID=$!

# Wait a moment for Flask to start
sleep 2

# Start React frontend
cd frontend
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

npm run dev &
REACT_PID=$!

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping servers..."
    kill $FLASK_PID 2>/dev/null
    kill $REACT_PID 2>/dev/null
    exit
}

trap cleanup SIGINT SIGTERM

# Wait for both processes
wait
