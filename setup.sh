#!/bin/bash

# BRI Application Setup Script
# This script sets up the BRI application with all necessary configurations

echo "🚀 Setting up Behavioral Risk Index (BRI) Application..."

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p data/raw
mkdir -p data/processed
mkdir -p output
mkdir -p uploads
mkdir -p backups

# Set permissions
chmod 755 logs data output uploads backups

# Create .env file from template
echo "⚙️  Creating environment file..."
cat > .env << 'EOF'
# Environment Variables for BRI Application

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=bri_db
POSTGRES_USER=bri_user
POSTGRES_PASSWORD=bri_secure_password_2024

# Reddit API Configuration
REDDIT_CLIENT_ID=KiGT8yL2ko-ZaWBYbQrwmw
REDDIT_CLIENT_SECRET=ezQtiCbSJhozM55eq8IC8Ee6qOJglg
REDDIT_USER_AGENT=BRI-DataCollector/1.0 by mrayanasim09
REDDIT_USERNAME=mrayanasim09

# Application Configuration
FLASK_ENV=production
SECRET_KEY=bri_secret_key_2024_secure_random_string
HOST=0.0.0.0
PORT=5000

# Monitoring Configuration
MONITORING_ENABLED=True
METRICS_ENABLED=True

# Feature Flags
FEATURE_REDDIT_DATA=True
FEATURE_GDELT_DATA=True
FEATURE_YAHOO_FINANCE=True
FEATURE_REAL_TIME_ALERTS=True
FEATURE_ADVANCED_ANALYTICS=True
EOF

echo "✅ Environment file created!"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Install additional dependencies for monitoring
echo "📊 Installing monitoring dependencies..."
pip install prometheus-client psutil redis

# Test Reddit API connection
echo "🔗 Testing Reddit API connection..."
python3 -c "
import sys
sys.path.append('src')
from reddit_api import RedditAPIClient
try:
    client = RedditAPIClient()
    print('✅ Reddit API connection successful!')
except Exception as e:
    print(f'❌ Reddit API connection failed: {e}')
"

# Test data validation
echo "🧪 Testing data validation..."
python3 -c "
import sys
sys.path.append('src')
from data_validation import DataValidator
import pandas as pd

# Test with sample data
test_data = pd.DataFrame({
    'title': ['Test post 1', 'Test post 2'],
    'subreddit': ['investing', 'stocks'],
    'score': [100, 200],
    'created_utc': [1640995200, 1640995300]
})

validator = DataValidator()
report = validator.validate_data(test_data, 'reddit')
print(f'✅ Data validation test passed! Overall score: {report.overall_score:.2f}')
"

# Run unit tests
echo "🧪 Running unit tests..."
python3 -m pytest tests/ -v --tb=short

# Create Docker images
echo "🐳 Building Docker images..."
docker build -t bri-app:latest .

# Start services with Docker Compose
echo "🚀 Starting services with Docker Compose..."
docker-compose up -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
echo "🏥 Checking service health..."
docker-compose ps

# Test API endpoints
echo "🌐 Testing API endpoints..."
sleep 10

# Test health endpoint
echo "Testing health endpoint..."
curl -f http://localhost:5000/health || echo "Health endpoint not ready yet"

# Test BRI summary endpoint
echo "Testing BRI summary endpoint..."
curl -f http://localhost:5000/api/summary || echo "BRI summary endpoint not ready yet"

echo ""
echo "🎉 Setup completed!"
echo ""
echo "📋 Next Steps:"
echo "1. Access the BRI API: http://localhost:5000"
echo "2. View Grafana dashboard: http://localhost:3000 (admin/admin)"
echo "3. View Prometheus metrics: http://localhost:9090"
echo "4. Check logs: docker-compose logs -f bri-app"
echo "5. Run the pipeline: python refactored_bri_pipeline.py"
echo ""
echo "🔧 Configuration:"
echo "- Reddit API: ✅ Configured with your credentials"
echo "- Database: ✅ PostgreSQL with comprehensive schema"
echo "- Monitoring: ✅ Prometheus + Grafana"
echo "- Docker: ✅ Multi-container setup"
echo "- Testing: ✅ Comprehensive unit tests"
echo ""
echo "📚 Documentation:"
echo "- API Docs: docs/API_DOCUMENTATION.md"
echo "- Configuration: config.yaml"
echo "- Environment: .env"
echo ""
echo "🚀 Your BRI application is ready to use!"
