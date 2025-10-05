#!/bin/bash
# Start Simple SeaSeeAI System

echo "🚀 Starting Simple SeaSeeAI System"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed or not in PATH"
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs data

# Check if simple_api.py exists
if [ ! -f "simple_api.py" ]; then
    echo "❌ simple_api.py not found. Please create it first."
    exit 1
fi

echo "🌐 Starting API Server..."
echo "💡 The API server will start in the background"
echo ""

# Start API server in background
python simple_api.py &
API_PID=$!

echo "API Server started with PID: $API_PID"
echo "⏳ Waiting for API to start..."
sleep 3

# Check if API is running
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ API Server is running on http://localhost:8000"
else
    echo "❌ API Server failed to start. Check the output above for errors."
    exit 1
fi

echo ""
echo "📊 Starting Dashboard..."
echo "🎯 Dashboard will open at: http://localhost:8501"
echo ""
echo "💡 Press Ctrl+C to stop both servers"
echo ""

# Start dashboard using Python module (works even if streamlit command not in PATH)
python -m streamlit run simple_dashboard.py

# Cleanup when user presses Ctrl+C
echo ""
echo "🛑 Stopping servers..."
kill $API_PID 2>/dev/null || true
echo "✅ System stopped"
