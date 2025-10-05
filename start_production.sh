#!/bin/bash
# SeaSeeAI Production Startup Script

set -e

echo "🚀 Starting SeaSeeAI Production System"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs data models config

# Check if models exist
if [ ! -f "models/best_traisformer.pth" ]; then
    echo "❌ Model file not found. Training models..."
    python src/training/train_traisformer.py
fi

# Generate default config if not exists
if [ ! -f "config/production.yaml" ]; then
    echo "📝 Generating production configuration..."
    python src/utils/production_config.py
fi

# Build and start services
echo "🐳 Building and starting Docker services..."
docker-compose up --build -d

echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
echo "🏥 Checking service health..."

# Check API
if curl -s http://localhost:8000/health | grep -q "healthy"; then
    echo "✅ API is healthy"
else
    echo "❌ API health check failed"
    docker-compose logs seaseeai-api
    exit 1
fi

# Check Dashboard
if curl -s http://localhost:8501 > /dev/null; then
    echo "✅ Dashboard is running"
else
    echo "⚠️ Dashboard might be starting up..."
fi

echo ""
echo "🎉 SeaSeeAI Production System Started Successfully!"
echo ""
echo "📊 Access Points:"
echo "   API Server: http://localhost:8000"
echo "   API Documentation: http://localhost:8000/docs"
echo "   Dashboard: http://localhost:8501"
echo "   Grafana: http://localhost:3000 (admin/admin)"
echo "   Prometheus: http://localhost:9090"
echo ""
echo "📋 Useful Commands:"
echo "   View logs: docker-compose logs -f"
echo "   Stop services: docker-compose down"
echo "   Restart services: docker-compose restart"
echo "   Scale API: docker-compose up -d --scale seaseeai-api=3"
echo ""
