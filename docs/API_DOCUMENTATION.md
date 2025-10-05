# 📚 **BRI Dashboard API Documentation**

## 🎯 **Overview**

The Enhanced BRI Dashboard provides a comprehensive REST API for accessing behavioral risk analysis data, advanced analytics, and real-time market insights. This documentation covers all available endpoints, request/response formats, and usage examples.

---

## 🔗 **Base URL**

```
Production: https://your-domain.com
Development: http://localhost:5000
```

---

## 📊 **Core Endpoints**

### **1. Dashboard Summary**
Get comprehensive summary statistics and current BRI metrics.

**Endpoint:** `GET /api/summary`

**Response:**
```json
{
  "current_bri": 38.5,
  "risk_level": "Moderate",
  "mean_bri": 42.3,
  "std_bri": 8.7,
  "min_bri": 15.2,
  "max_bri": 78.9,
  "correlation": 0.872,
  "r_squared": 0.761,
  "trend": "Rising",
  "data_points": 1096,
  "last_updated": "2024-01-15 14:30:25"
}
```

**Status Codes:**
- `200 OK` - Success
- `500 Internal Server Error` - Server error

---

### **2. BRI Time Series Chart**
Get interactive BRI time series data with risk-based coloring.

**Endpoint:** `GET /api/bri_chart`

**Response:**
```json
{
  "data": [
    {
      "x": ["2022-01-01", "2022-01-02", ...],
      "y": [35.2, 38.7, ...],
      "mode": "lines+markers",
      "name": "BRI",
      "line": {"color": "#38A169", "width": 2},
      "marker": {"size": 4, "color": ["#38A169", "#D69E2E", ...]}
    }
  ],
  "layout": {
    "title": "Behavioral Risk Index (BRI) Over Time",
    "xaxis": {"title": "Date"},
    "yaxis": {"title": "BRI (0-100)"},
    "height": 500
  }
}
```

---

## 🔥 **Advanced Analytics Endpoints**

### **3. Risk Heatmap**
Get risk heatmap visualization showing risk levels over time.

**Endpoint:** `GET /api/risk_heatmap`

**Response:**
```json
{
  "data": [
    {
      "z": [[35.2, 38.7, 42.1], [33.8, 36.4, 39.2], ...],
      "x": ["Risk Level"],
      "y": ["2022-01", "2022-02", ...],
      "type": "heatmap",
      "colorscale": [[0, "#38A169"], [0.5, "#D69E2E"], [1, "#E53E3E"]]
    }
  ],
  "layout": {
    "title": "Risk Heatmap - Monthly BRI Levels",
    "height": 400
  }
}
```

---

### **4. Volatility Clustering**
Get volatility clustering analysis with market regime detection.

**Endpoint:** `GET /api/volatility_clustering`

**Response:**
```json
{
  "figure": {
    "data": [...],
    "layout": {...}
  },
  "cluster_stats": {
    "Low Volatility": {
      "bri_volatility": {"mean": 2.1, "std": 0.8},
      "bri_abs_change": {"mean": 0.05, "std": 0.02}
    },
    "Moderate Volatility": {
      "bri_volatility": {"mean": 5.3, "std": 1.2},
      "bri_abs_change": {"mean": 0.12, "std": 0.04}
    },
    "High Volatility": {
      "bri_volatility": {"mean": 12.7, "std": 3.1},
      "bri_abs_change": {"mean": 0.28, "std": 0.09}
    }
  },
  "cluster_labels": {
    "Low Volatility": 245,
    "Moderate Volatility": 678,
    "High Volatility": 173
  }
}
```

---

### **5. Early Warning System**
Get early warning system visualization and spike detection.

**Endpoint:** `GET /api/early_warning`

**Response:**
```json
{
  "figure": {
    "data": [...],
    "layout": {...}
  },
  "warning_stats": {
    "total_spikes": 23,
    "spike_frequency": 2.1,
    "avg_spike_magnitude": 67.3,
    "max_spike": 89.2,
    "recent_spikes": 3
  },
  "spike_events": {
    "2023-03-15": {
      "BRI": 78.5,
      "rolling_threshold": 65.2
    },
    "2023-10-22": {
      "BRI": 82.1,
      "rolling_threshold": 68.7
    }
  }
}
```

---

### **6. Confidence Intervals & Forecasting**
Get BRI forecasting with confidence intervals.

**Endpoint:** `GET /api/confidence_intervals`

**Response:**
```json
{
  "figure": {
    "data": [...],
    "layout": {...}
  },
  "predictions": {
    "2024-01-16": 39.2,
    "2024-01-17": 41.1,
    "2024-01-18": 38.7,
    "2024-01-19": 42.3,
    "2024-01-20": 40.8,
    "2024-01-21": 39.5,
    "2024-01-22": 41.7
  },
  "confidence_intervals": {
    "upper": {
      "2024-01-16": 45.8,
      "2024-01-17": 47.2,
      "2024-01-18": 44.9,
      "2024-01-19": 48.1,
      "2024-01-20": 46.5,
      "2024-01-21": 45.2,
      "2024-01-22": 47.8
    },
    "lower": {
      "2024-01-16": 32.6,
      "2024-01-17": 35.0,
      "2024-01-18": 32.5,
      "2024-01-19": 36.5,
      "2024-01-20": 35.1,
      "2024-01-21": 33.8,
      "2024-01-22": 35.6
    }
  },
  "prediction_accuracy": {
    "trend_magnitude": 0.15,
    "confidence_level": 0.95,
    "prediction_horizon": 7
  }
}
```

---

### **7. Comprehensive Analytics Summary**
Get all advanced analytics in a single request.

**Endpoint:** `GET /api/analytics_summary`

**Response:**
```json
{
  "risk_heatmap": {...},
  "volatility_clustering": {...},
  "early_warning": {...},
  "confidence_intervals": {...}
}
```

---

## 🛠️ **Usage Examples**

### **JavaScript/Fetch API**
```javascript
// Get summary statistics
async function getSummary() {
  try {
    const response = await fetch('/api/summary');
    const data = await response.json();
    console.log('Current BRI:', data.current_bri);
    console.log('Risk Level:', data.risk_level);
  } catch (error) {
    console.error('Error:', error);
  }
}

// Get risk heatmap
async function getRiskHeatmap() {
  try {
    const response = await fetch('/api/risk_heatmap');
    const data = await response.json();
    Plotly.newPlot('heatmap-container', data.data, data.layout);
  } catch (error) {
    console.error('Error:', error);
  }
}

// Get early warning alerts
async function getEarlyWarning() {
  try {
    const response = await fetch('/api/early_warning');
    const data = await response.json();
    
    if (data.warning_stats.recent_spikes > 0) {
      alert(`Warning: ${data.warning_stats.recent_spikes} recent risk spikes detected!`);
    }
  } catch (error) {
    console.error('Error:', error);
  }
}
```

### **Python/Requests**
```python
import requests
import json

# Get summary statistics
def get_summary():
    response = requests.get('http://localhost:5000/api/summary')
    if response.status_code == 200:
        data = response.json()
        print(f"Current BRI: {data['current_bri']}")
        print(f"Risk Level: {data['risk_level']}")
        return data
    else:
        print(f"Error: {response.status_code}")
        return None

# Get volatility clustering
def get_volatility_clustering():
    response = requests.get('http://localhost:5000/api/volatility_clustering')
    if response.status_code == 200:
        data = response.json()
        print(f"Cluster distribution: {data['cluster_labels']}")
        return data
    else:
        print(f"Error: {response.status_code}")
        return None

# Get forecasting predictions
def get_forecast():
    response = requests.get('http://localhost:5000/api/confidence_intervals')
    if response.status_code == 200:
        data = response.json()
        print("7-day BRI forecast:")
        for date, prediction in data['predictions'].items():
            print(f"{date}: {prediction:.1f}")
        return data
    else:
        print(f"Error: {response.status_code}")
        return None
```

### **cURL Examples**
```bash
# Get summary statistics
curl -X GET http://localhost:5000/api/summary

# Get risk heatmap
curl -X GET http://localhost:5000/api/risk_heatmap

# Get early warning system
curl -X GET http://localhost:5000/api/early_warning

# Get comprehensive analytics
curl -X GET http://localhost:5000/api/analytics_summary
```

---

## 📈 **Data Models**

### **BRI Data Structure**
```json
{
  "date": "2024-01-15",
  "BRI": 38.5,
  "sentiment_volatility": 0.15,
  "media_herding": 0.23,
  "news_tone": 0.42,
  "event_density": 0.18,
  "polarity_skew": 0.12
}
```

### **Risk Level Classification**
```json
{
  "low": {
    "threshold": 30,
    "description": "Stable market conditions, low behavioral risk",
    "color": "#38A169"
  },
  "moderate": {
    "threshold": 60,
    "description": "Normal market volatility, moderate risk",
    "color": "#D69E2E"
  },
  "high": {
    "threshold": 100,
    "description": "Elevated stress, high behavioral risk",
    "color": "#E53E3E"
  }
}
```

---

## ⚠️ **Error Handling**

### **Error Response Format**
```json
{
  "error": "Error message description",
  "code": "ERROR_CODE",
  "timestamp": "2024-01-15T14:30:25Z"
}
```

### **Common Error Codes**
- `400 Bad Request` - Invalid request parameters
- `404 Not Found` - Endpoint or resource not found
- `500 Internal Server Error` - Server-side error
- `503 Service Unavailable` - Service temporarily unavailable

---

## 🔒 **Rate Limiting**

- **Rate Limit:** 100 requests per hour per IP
- **Headers:** 
  - `X-RateLimit-Limit`: 100
  - `X-RateLimit-Remaining`: 95
  - `X-RateLimit-Reset`: 1642248625

---

## 📊 **Response Times**

| Endpoint | Average Response Time | 95th Percentile |
|----------|----------------------|-----------------|
| `/api/summary` | 50ms | 100ms |
| `/api/bri_chart` | 200ms | 400ms |
| `/api/risk_heatmap` | 300ms | 600ms |
| `/api/volatility_clustering` | 500ms | 1000ms |
| `/api/early_warning` | 400ms | 800ms |
| `/api/confidence_intervals` | 600ms | 1200ms |
| `/api/analytics_summary` | 1500ms | 3000ms |

---

## 🚀 **Best Practices**

### **1. Caching**
- Cache responses for 5 minutes to reduce server load
- Use ETags for conditional requests
- Implement client-side caching for static data

### **2. Error Handling**
- Always check response status codes
- Implement exponential backoff for retries
- Handle network timeouts gracefully

### **3. Performance**
- Use pagination for large datasets
- Request only needed data fields
- Implement request debouncing for real-time updates

### **4. Security**
- Validate all input parameters
- Use HTTPS in production
- Implement API key authentication for sensitive endpoints

---

## 🔄 **Webhook Integration**

### **Webhook Configuration**
```json
{
  "url": "https://your-app.com/webhook/bri-alerts",
  "events": ["risk_spike", "threshold_breach", "anomaly_detected"],
  "secret": "your-webhook-secret"
}
```

### **Webhook Payload**
```json
{
  "event": "risk_spike",
  "timestamp": "2024-01-15T14:30:25Z",
  "data": {
    "bri_value": 78.5,
    "threshold": 65.2,
    "risk_level": "High",
    "spike_magnitude": 13.3
  }
}
```

---

## 📞 **Support**

For API support and questions:
- **Documentation:** [GitHub Repository](https://github.com/your-repo/bri-dashboard)
- **Issues:** [GitHub Issues](https://github.com/your-repo/bri-dashboard/issues)
- **Email:** support@your-domain.com

---

## 📝 **Changelog**

### **Version 2.0.0** (Current)
- ✅ Added advanced analytics endpoints
- ✅ Implemented risk heatmap visualization
- ✅ Added volatility clustering analysis
- ✅ Created early warning system
- ✅ Added confidence intervals and forecasting
- ✅ Enhanced error handling and documentation

### **Version 1.0.0**
- ✅ Basic BRI calculation and visualization
- ✅ Summary statistics endpoint
- ✅ Time series chart endpoint
- ✅ Professional dashboard interface
