# 🌐 **Behavioral Risk Index (BRI) Web Application**

## 🚀 **Production-Ready Web Dashboard**

A comprehensive web application for real-time monitoring and analysis of the Behavioral Risk Index (BRI) - a novel measure of market behavioral risk patterns.

---

## ✨ **Features**

### **📊 Real-Time Dashboard**
- **Live BRI Monitoring**: Current BRI value with risk level classification
- **Interactive Charts**: Plotly-powered visualizations
- **Statistical Analysis**: Comprehensive BRI statistics and trends
- **Responsive Design**: Mobile-friendly Bootstrap interface

### **📈 Advanced Visualizations**
- **BRI Time Series**: Historical BRI with 7-day moving average
- **BRI-VIX Correlation**: Scatter plot with regression analysis
- **Feature Importance**: Bar chart of behavioral risk components
- **Distribution Analysis**: Histogram with statistical overlays

### **🎯 Risk Management**
- **Risk Level Classification**: Low, Medium, High, Extreme
- **Trend Analysis**: 30-day directional trend
- **Correlation Metrics**: BRI-VIX relationship strength
- **Statistical Thresholds**: Mean ± 1σ volatility bands

---

## 🛠 **Technology Stack**

### **Backend**
- **Flask**: Python web framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualizations

### **Frontend**
- **Bootstrap 5**: Responsive UI framework
- **Plotly.js**: Interactive charting library
- **Font Awesome**: Icons and graphics
- **Vanilla JavaScript**: Dynamic functionality

### **Deployment**
- **Render**: Cloud hosting platform
- **Gunicorn**: WSGI HTTP server
- **Python 3.9+**: Runtime environment

---

## 🚀 **Deployment on Render**

### **Prerequisites**
- GitHub repository with the code
- Render account (free tier available)
- Python 3.9+ environment

### **Deployment Steps**

#### **1. Repository Setup**
```bash
# Ensure all files are in the repository
git add .
git commit -m "Add web application"
git push origin main
```

#### **2. Render Configuration**

**Service Type**: Web Service

**Build Command**:
```bash
pip install -r requirements.txt
```

**Start Command**:
```bash
gunicorn app:app
```

**Root Directory**: Leave empty (uses repository root)

**Environment**: Python 3

#### **3. Environment Variables**
No additional environment variables required for basic functionality.

#### **4. Advanced Configuration**

**Auto-Deploy**: Enable for automatic deployments on code changes

**Health Check Path**: `/` (optional)

**Instance Type**: Starter (free) or higher for production

---

## 📁 **File Structure**

```
Behavioral_Risk_Index/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── Procfile             # Render deployment configuration
├── templates/
│   └── index.html       # Main dashboard template
├── output/              # Data output directory
│   ├── research_grade/  # Research pipeline results
│   └── complete/        # Complete pipeline results
└── README_WEB_APP.md    # This documentation
```

---

## 🔧 **Local Development**

### **Installation**
```bash
# Clone repository
git clone <your-repo-url>
cd Behavioral_Risk_Index

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

### **Access**
- **Local**: http://localhost:5000
- **Production**: https://your-app-name.onrender.com

---

## 📊 **API Endpoints**

### **Summary Data**
- `GET /api/summary` - BRI summary statistics
- `GET /api/correlation` - BRI-VIX correlation data
- `GET /api/features` - Feature importance data

### **Charts**
- `GET /api/bri_chart` - BRI time series chart
- `GET /api/correlation_chart` - BRI-VIX correlation chart
- `GET /api/feature_chart` - Feature importance chart
- `GET /api/distribution_chart` - BRI distribution chart

---

## 🎯 **Data Sources**

### **Primary Data**
- **BRI Time Series**: From research pipeline output
- **Market Data**: VIX, S&P500, and other indicators
- **Feature Data**: Behavioral risk components

### **Fallback Data**
- **Sample Data**: Generated for demonstration if no real data available
- **Realistic Patterns**: Simulated BRI with market-like behavior

---

## 🔄 **Auto-Refresh**

### **Dashboard Updates**
- **Manual Refresh**: Browser refresh or page reload
- **Auto-Refresh**: Every 5 minutes (configurable)
- **Real-Time**: API calls for live data updates

### **Data Updates**
- **Pipeline Integration**: Connects to research pipeline output
- **File Monitoring**: Automatically loads new data files
- **Error Handling**: Graceful fallback to sample data

---

## 🎨 **Customization**

### **Styling**
- **Bootstrap Themes**: Easy color scheme changes
- **CSS Variables**: Customizable styling
- **Responsive Design**: Mobile-first approach

### **Charts**
- **Plotly Configuration**: Interactive chart settings
- **Color Schemes**: Customizable chart colors
- **Layout Options**: Flexible chart arrangements

### **Features**
- **Risk Thresholds**: Adjustable risk level boundaries
- **Update Intervals**: Configurable refresh rates
- **Chart Types**: Additional visualization options

---

## 🚨 **Troubleshooting**

### **Common Issues**

#### **Data Not Loading**
- Check if `output/research_grade/` or `output/complete/` directories exist
- Verify data files are properly formatted
- Check application logs for errors

#### **Charts Not Displaying**
- Ensure JavaScript is enabled
- Check browser console for errors
- Verify Plotly.js is loading correctly

#### **Deployment Issues**
- Check Render build logs
- Verify all dependencies are in requirements.txt
- Ensure Procfile is correctly formatted

### **Debug Mode**
```bash
# Run in debug mode locally
export FLASK_DEBUG=1
python app.py
```

---

## 📈 **Performance**

### **Optimization**
- **Lazy Loading**: Charts load on demand
- **Caching**: API responses cached for 5 minutes
- **Compression**: Gzip compression enabled
- **CDN**: Static assets served via CDN

### **Scalability**
- **Horizontal Scaling**: Multiple instances supported
- **Database Ready**: Easy integration with databases
- **API Rate Limiting**: Configurable rate limits
- **Monitoring**: Built-in health checks

---

## 🔒 **Security**

### **Security Features**
- **Input Validation**: All inputs validated
- **XSS Protection**: Content Security Policy headers
- **CSRF Protection**: Cross-site request forgery prevention
- **Error Handling**: Secure error messages

### **Best Practices**
- **Environment Variables**: Sensitive data in environment
- **HTTPS Only**: Secure connections required
- **Regular Updates**: Dependencies kept current
- **Security Headers**: Comprehensive security headers

---

## 📚 **Documentation**

### **Additional Resources**
- **API Documentation**: Complete endpoint documentation
- **User Guide**: Step-by-step usage instructions
- **Developer Guide**: Customization and extension guide
- **Troubleshooting**: Common issues and solutions

### **Support**
- **GitHub Issues**: Bug reports and feature requests
- **Documentation**: Comprehensive guides and tutorials
- **Community**: User community and discussions

---

## 🎉 **Success Metrics**

### **Deployment Ready**
- ✅ **Production Code**: Clean, optimized codebase
- ✅ **Dependencies**: Minimal, secure dependencies
- ✅ **Configuration**: Render-ready configuration
- ✅ **Documentation**: Complete deployment guide

### **User Experience**
- ✅ **Responsive Design**: Mobile and desktop friendly
- ✅ **Fast Loading**: Optimized performance
- ✅ **Interactive Charts**: Engaging visualizations
- ✅ **Real-Time Data**: Live updates and monitoring

### **Technical Excellence**
- ✅ **Error Handling**: Robust error management
- ✅ **Security**: Production-ready security
- ✅ **Scalability**: Horizontal scaling support
- ✅ **Maintainability**: Clean, documented code

---

## 🚀 **Ready for Deployment!**

Your Behavioral Risk Index web application is now **production-ready** and can be deployed on Render with the following configuration:

**Start Command**: `gunicorn app:app`
**Root Directory**: (leave empty)
**Environment**: Python 3

**The application will automatically:**
- Load BRI data from your research pipeline
- Generate interactive visualizations
- Provide real-time risk monitoring
- Update every 5 minutes automatically

**Deploy now and start monitoring behavioral risk patterns in real-time!** 🎉

---

*Last Updated: October 5, 2024 - 05:30 UTC*
*Status: ✅ PRODUCTION READY - Web application ready for Render deployment*
