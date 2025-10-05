# Behavioral Risk Index (BRI) API Documentation

## Overview

The BRI API provides comprehensive endpoints for accessing behavioral risk data, market analysis, and real-time monitoring capabilities. This RESTful API is built with Flask and provides JSON responses for all endpoints.

## Base URL

```
http://localhost:5000/api
```

## Authentication

Currently, the API does not require authentication. For production deployment, consider implementing API key authentication.

## Rate Limiting

- **Rate Limit**: 100 requests per hour per IP
- **Burst Limit**: 10 requests per minute
- **Headers**: Rate limit information is included in response headers

## Endpoints

### 1. Summary Data

#### GET `/api/summary`

Returns comprehensive BRI summary statistics and current market state.

**Response:**
```json
{
  "current_bri": 45.2,
  "mean_bri": 42.1,
  "std_bri": 8.3,
  "min_bri": 12.4,
  "max_bri": 89.7,
  "risk_level": "Medium",
  "trend": "Rising",
  "last_updated": "2024-01-15 14:30:00",
  "correlation": -0.194,
  "r_squared": 0.038,
  "data_points": 1257
}
```

**Status Codes:**
- `200 OK`: Success
- `500 Internal Server Error`: Server error

---

### 2. BRI Time Series

#### GET `/api/bri_chart`

Returns BRI time series data with risk-based coloring and moving averages.

**Response:**
```json
{
  "data": [
    {
      "x": ["2024-01-01", "2024-01-02", ...],
      "y": [45.2, 47.1, ...],
      "type": "scatter",
      "mode": "lines+markers",
      "name": "BRI",
      "line": {"color": "#38A169", "width": 2}
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

**Status Codes:**
- `200 OK`: Success
- `500 Internal Server Error`: Server error

---

### 3. Correlation Analysis

#### GET `/api/correlation`

Returns BRI-VIX correlation data and statistics.

**Response:**
```json
{
  "correlation": -0.194,
  "r_squared": 0.038,
  "data_points": 1257,
  "pearson_p_value": 0.001,
  "spearman_correlation": -0.187
}
```

#### GET `/api/correlation_chart`

Returns scatter plot data for BRI vs VIX correlation analysis.

**Response:**
```json
{
  "data": [
    {
      "x": [45.2, 47.1, ...],
      "y": [18.3, 19.1, ...],
      "type": "scatter",
      "mode": "markers",
      "name": "BRI vs VIX"
    }
  ],
  "layout": {
    "title": "BRI vs VIX Correlation (r = -0.194)",
    "xaxis": {"title": "Behavioral Risk Index (BRI)"},
    "yaxis": {"title": "VIX (Volatility Index)"}
  }
}
```

---

### 4. Feature Analysis

#### GET `/api/features`

Returns feature importance and component analysis.

**Response:**
```json
{
  "sent_vol": 45.2,
  "news_tone": 38.7,
  "herding": 52.1,
  "polarity_skew": 41.3,
  "event_density": 35.8
}
```

#### GET `/api/feature_chart`

Returns feature importance visualization data.

**Response:**
```json
{
  "data": [
    {
      "x": ["Sentiment Volatility", "News Tone", "Media Herding", "Polarity Skew", "Event Density"],
      "y": [45.2, 38.7, 52.1, 41.3, 35.8],
      "type": "bar",
      "marker": {"color": ["#2ECC71", "#3498DB", "#E74C3C", "#F1C40F", "#34495E"]}
    }
  ],
  "layout": {
    "title": "Feature Importance (Average Scores)",
    "xaxis": {"title": "Features"},
    "yaxis": {"title": "Average Score"}
  }
}
```

---

### 5. Distribution Analysis

#### GET `/api/distribution_chart`

Returns BRI distribution histogram data.

**Response:**
```json
{
  "data": [
    {
      "x": [12.4, 15.2, 18.1, ...],
      "type": "histogram",
      "nbinsx": 30,
      "marker": {"color": "#3498DB", "opacity": 0.7}
    }
  ],
  "layout": {
    "title": "BRI Distribution",
    "xaxis": {"title": "BRI Value"},
    "yaxis": {"title": "Frequency"}
  }
}
```

---

### 6. Advanced Analytics

#### GET `/api/box_plots`

Returns box plot analysis for BRI components.

**Response:**
```json
{
  "data": [
    {
      "y": [45.2, 47.1, ...],
      "type": "box",
      "name": "BRI Distribution",
      "boxpoints": "outliers"
    }
  ],
  "layout": {
    "title": "Feature Distribution Analysis",
    "yaxis": {"title": "Score"}
  }
}
```

#### GET `/api/violin_plots`

Returns violin plot analysis for BRI distribution.

**Response:**
```json
{
  "data": [
    {
      "y": [45.2, 47.1, ...],
      "type": "violin",
      "name": "BRI Distribution",
      "box_visible": true,
      "meanline_visible": true
    }
  ],
  "layout": {
    "title": "BRI Distribution (Violin Plot)",
    "yaxis": {"title": "BRI Value"}
  }
}
```

---

### 7. Forecasting

#### GET `/api/forecasting_comparison`

Returns forecasting comparison data with confidence intervals.

**Response:**
```json
{
  "data": [
    {
      "x": ["2024-01-01", "2024-01-02", ...],
      "y": [45.2, 47.1, ...],
      "type": "scatter",
      "mode": "lines",
      "name": "Actual BRI"
    },
    {
      "x": ["2024-01-15", "2024-01-16", ...],
      "y": [48.1, 49.3, ...],
      "type": "scatter",
      "mode": "lines",
      "name": "Forecast",
      "line": {"dash": "dash"}
    }
  ],
  "layout": {
    "title": "BRI Forecasting Comparison",
    "xaxis": {"title": "Date"},
    "yaxis": {"title": "BRI Value"}
  }
}
```

#### GET `/api/confidence_intervals`

Returns confidence interval analysis for BRI predictions.

**Response:**
```json
{
  "data": [
    {
      "x": ["2024-01-01", "2024-01-02", ...],
      "y": [45.2, 47.1, ...],
      "type": "scatter",
      "mode": "lines",
      "name": "BRI"
    },
    {
      "x": ["2024-01-01", "2024-01-02", ...],
      "y": [52.1, 54.3, ...],
      "type": "scatter",
      "mode": "lines",
      "name": "Upper 95% CI",
      "fill": "tonexty"
    }
  ],
  "layout": {
    "title": "BRI with 95% Confidence Intervals",
    "xaxis": {"title": "Date"},
    "yaxis": {"title": "BRI Value"}
  }
}
```

---

### 8. Model Performance

#### GET `/api/model_performance`

Returns model performance comparison data.

**Response:**
```json
{
  "data": [
    {
      "x": ["Random Forest", "XGBoost", "LSTM", "ARIMA"],
      "y": [0.789, 0.812, 0.756, 0.698],
      "type": "bar",
      "name": "R² Score",
      "marker": {"color": "#3182CE"}
    }
  ],
  "layout": {
    "title": "Model Performance Comparison",
    "xaxis": {"title": "Model"},
    "yaxis": {"title": "Score"}
  }
}
```

---

### 9. Risk Analysis

#### GET `/api/risk_heatmap`

Returns risk heatmap data for temporal risk analysis.

**Response:**
```json
{
  "data": [
    {
      "z": [[45.2, 47.1], [48.3, 46.8], ...],
      "type": "heatmap",
      "colorscale": [[0, "#38A169"], [0.5, "#D69E2E"], [1, "#E53E3E"]],
      "showscale": true
    }
  ],
  "layout": {
    "title": "BRI Risk Heatmap",
    "xaxis": {"title": "Risk Categories"},
    "yaxis": {"title": "Time Period"}
  }
}
```

#### GET `/api/early_warning`

Returns early warning system data with risk thresholds.

**Response:**
```json
{
  "data": [
    {
      "x": ["2024-01-01", "2024-01-02", ...],
      "y": [45.2, 47.1, ...],
      "type": "scatter",
      "mode": "lines",
      "name": "BRI"
    },
    {
      "x": ["2024-01-01", "2024-01-02", ...],
      "y": [75.0, 75.0, ...],
      "type": "scatter",
      "mode": "lines",
      "name": "High Risk Threshold",
      "line": {"dash": "dash"}
    }
  ],
  "layout": {
    "title": "BRI Early Warning System",
    "xaxis": {"title": "Date"},
    "yaxis": {"title": "BRI Level"}
  }
}
```

---

### 10. Statistical Validation

#### GET `/api/statistical_validation_report`

Returns statistical validation results.

**Response:**
```json
{
  "data": [
    {
      "x": ["Correlation", "Stationarity", "Granger Causality", "Normality", "Heteroskedasticity"],
      "y": [0.001, 0.012, 0.0001, 0.0004, 0.0012],
      "type": "bar",
      "marker": {"color": ["#38A169", "#D69E2E", "#38A169", "#38A169", "#38A169"]}
    }
  ],
  "layout": {
    "title": "Statistical Validation Results",
    "xaxis": {"title": "Statistical Test"},
    "yaxis": {"title": "P-Value"}
  }
}
```

---

## Error Handling

### Standard Error Response

```json
{
  "error": "Error message",
  "code": "ERROR_CODE",
  "timestamp": "2024-01-15T14:30:00Z"
}
```

### Common Error Codes

- `INVALID_REQUEST`: Invalid request parameters
- `DATA_NOT_FOUND`: Requested data not available
- `PROCESSING_ERROR`: Error during data processing
- `RATE_LIMIT_EXCEEDED`: Rate limit exceeded
- `SERVER_ERROR`: Internal server error

---

## Rate Limiting

### Headers

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642248600
```

### Status Codes

- `429 Too Many Requests`: Rate limit exceeded
- `503 Service Unavailable`: Service temporarily unavailable

---

## Data Formats

### Date Format
All dates are returned in ISO 8601 format: `YYYY-MM-DDTHH:MM:SSZ`

### Number Format
- BRI values: 0-100 scale
- Correlations: -1.0 to 1.0
- Percentages: 0.0 to 100.0

### Risk Levels
- `Low`: 0-30
- `Medium`: 30-60
- `High`: 60-100

---

## Examples

### Python Client

```python
import requests
import json

# Get BRI summary
response = requests.get('http://localhost:5000/api/summary')
data = response.json()
print(f"Current BRI: {data['current_bri']}")

# Get correlation data
response = requests.get('http://localhost:5000/api/correlation')
correlation = response.json()
print(f"BRI-VIX Correlation: {correlation['correlation']}")
```

### JavaScript Client

```javascript
// Get BRI summary
fetch('/api/summary')
  .then(response => response.json())
  .then(data => {
    console.log(`Current BRI: ${data.current_bri}`);
  });

// Get correlation data
fetch('/api/correlation')
  .then(response => response.json())
  .then(data => {
    console.log(`BRI-VIX Correlation: ${data.correlation}`);
  });
```

---

## Changelog

### Version 2.0.0 (2024-01-15)
- Added comprehensive API documentation
- Enhanced error handling
- Added rate limiting
- Improved response formats

### Version 1.0.0 (2024-01-01)
- Initial API release
- Basic BRI endpoints
- Chart data endpoints

---

## Support

For API support and questions:
- **Documentation**: [GitHub Repository](https://github.com/mrayanasim09/Behavioral_Risk_Index)
- **Issues**: [GitHub Issues](https://github.com/mrayanasim09/Behavioral_Risk_Index/issues)
- **Email**: [Your Email]

---

*Last Updated: January 15, 2024*