#!/bin/bash
# Production deployment script for SeaSeeAI

set -e

echo "🚀 Starting SeaSeeAI Production Deployment"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs data models config monitoring

# Check if model exists
if [ ! -f "models/best_traisformer.pth" ]; then
    echo "❌ Model file not found. Please ensure models/best_traisformer.pth exists."
    echo "💡 You can train the model by running: python src/training/train_traisformer.py"
    exit 1
fi

# Check if config exists
if [ ! -f "config/production.yaml" ]; then
    echo "📝 Creating default production configuration..."
    python src/utils/production_config.py
fi

# Build and start services
echo "🐳 Building Docker images..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

echo "⏳ Waiting for services to start..."
sleep 30

# Check if API is healthy
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API is healthy"
else
    echo "❌ API is not responding. Check logs with: docker-compose logs seaseeai-api"
    exit 1
fi

# Check if dashboard is accessible
if curl -f http://localhost:8501 > /dev/null 2>&1; then
    echo "✅ Dashboard is accessible"
else
    echo "⚠️ Dashboard might be starting up. Check with: docker-compose logs seaseeai-dashboard"
fi

echo ""
echo "🎉 Deployment completed successfully!"
echo ""
echo "📊 Access your services:"
echo "   API: http://localhost:8000"
echo "   Dashboard: http://localhost:8501"
echo "   API Documentation: http://localhost:8000/docs"
echo "   Grafana: http://localhost:3000 (admin/admin)"
echo ""
echo "📋 Useful commands:"
echo "   View logs: docker-compose logs -f"
echo "   Stop services: docker-compose down"
echo "   Update services: docker-compose pull && docker-compose up -d"
echo ""
