# 🚀 BRI Quick Start Guide

## Prerequisites
- Docker and Docker Compose installed
- Python 3.11+ (for local development)
- Git

## 🎯 Quick Setup (5 minutes)

### 1. Run the Setup Script
```bash
./setup.sh
```

This will:
- ✅ Configure Reddit API with your credentials
- ✅ Install all dependencies
- ✅ Run unit tests
- ✅ Build Docker images
- ✅ Start all services
- ✅ Test API endpoints

### 2. Access Your Services

| Service | URL | Credentials |
|---------|-----|-------------|
| **BRI API** | http://localhost:5000 | - |
| **Grafana Dashboard** | http://localhost:3000 | admin/admin |
| **Prometheus Metrics** | http://localhost:9090 | - |
| **API Documentation** | http://localhost:5000/docs | - |

### 3. Test the API
```bash
# Health check
curl http://localhost:5000/health

# BRI summary
curl http://localhost:5000/api/summary

# BRI time series
curl http://localhost:5000/api/bri_chart
```

## 🔧 Manual Setup (Alternative)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install prometheus-client psutil redis
```

### 2. Start with Docker Compose
```bash
docker-compose up -d
```

### 3. Run the Pipeline
```bash
python refactored_bri_pipeline.py --start-date 2024-01-01 --end-date 2024-01-31
```

## 📊 Monitoring

### Grafana Dashboard
- **URL**: http://localhost:3000
- **Username**: admin
- **Password**: admin
- **Dashboard**: BRI Monitoring Dashboard

### Prometheus Metrics
- **URL**: http://localhost:9090
- **Metrics**: bri_*, node_*, postgres_*

## 🧪 Testing

### Run Unit Tests
```bash
python -m pytest tests/ -v
```

### Test Reddit API
```bash
python -c "
from src.reddit_api import RedditAPIClient
client = RedditAPIClient()
posts = client.collect_subreddit_data('investing', limit=5)
print(f'Collected {len(posts)} posts')
"
```

### Test Data Validation
```bash
python -c "
from src.data_validation import DataValidator
import pandas as pd

test_data = pd.DataFrame({
    'title': ['Test post'],
    'subreddit': ['investing'],
    'score': [100],
    'created_utc': [1640995200]
})

validator = DataValidator()
report = validator.validate_data(test_data, 'reddit')
print(f'Data quality score: {report.overall_score:.2f}')
"
```

## 🚀 Production Deployment

### 1. Update Environment Variables
```bash
# Edit .env file with production values
nano .env
```

### 2. Deploy with Docker Compose
```bash
docker-compose -f docker-compose.yml up -d
```

### 3. Scale Services
```bash
docker-compose up -d --scale bri-app=3
```

## 📈 Usage Examples

### Collect Data
```python
from src.reddit_api import RedditAPIClient
from src.data_validation import DataValidator

# Collect Reddit data
client = RedditAPIClient()
posts = client.collect_subreddit_data('investing', limit=100)

# Validate data quality
validator = DataValidator()
report = validator.validate_data(posts, 'reddit')
print(f"Data quality: {report.overall_score:.2f}")
```

### Run BRI Pipeline
```python
from src.pipeline_phases import *

# Initialize phases
phase1 = Phase1DataCollection(data_collector, gdelt_processor)
phase2 = Phase2DataPreprocessing(text_preprocessor)
phase3 = Phase3FeatureEngineering()
phase4 = Phase4BRICalculation()

# Collect data
market_data = phase1.collect_market_data('2024-01-01', '2024-01-31')
reddit_data = phase1.collect_reddit_data('2024-01-01', '2024-01-31')

# Process data
reddit_clean = phase2.clean_reddit_text(reddit_data)
sentiment_data = phase2.perform_sentiment_analysis(reddit_clean, gdelt_clean)

# Calculate BRI
features = phase3.create_behavioral_features(sentiment_data, gdelt_clean, reddit_clean)
bri_data = phase4.calculate_bri(features)
```

## 🔍 Troubleshooting

### Common Issues

1. **Reddit API Rate Limits**
   - Solution: Wait and retry, or reduce request frequency

2. **Database Connection Issues**
   - Solution: Check PostgreSQL is running and credentials are correct

3. **Docker Build Failures**
   - Solution: Check Docker is running and has enough resources

4. **Memory Issues**
   - Solution: Increase Docker memory limit or reduce batch sizes

### Check Logs
```bash
# Application logs
docker-compose logs -f bri-app

# All services
docker-compose logs -f

# Specific service
docker-compose logs -f postgres
```

### Restart Services
```bash
# Restart all
docker-compose restart

# Restart specific service
docker-compose restart bri-app
```

## 📚 Documentation

- **API Documentation**: `docs/API_DOCUMENTATION.md`
- **Configuration**: `config.yaml`
- **Database Schema**: `database/init.sql`
- **Docker Setup**: `Dockerfile`, `docker-compose.yml`
- **Monitoring**: `monitoring/` directory

## 🆘 Support

If you encounter issues:
1. Check the logs: `docker-compose logs -f`
2. Verify configuration: `config.yaml` and `.env`
3. Test individual components
4. Check service health: `docker-compose ps`

## 🎉 Success!

Your BRI application is now running with:
- ✅ Real Reddit API integration
- ✅ Comprehensive data validation
- ✅ Production-ready monitoring
- ✅ Docker containerization
- ✅ Database persistence
- ✅ Unit tests
- ✅ API documentation

**Next**: Start collecting data and building your Behavioral Risk Index!
