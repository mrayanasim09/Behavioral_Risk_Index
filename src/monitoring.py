"""
Comprehensive Monitoring and Alerting System

This module provides monitoring, alerting, and observability features for the BRI application.
It includes metrics collection, health checks, performance monitoring, and alerting.
"""

import logging
import time
import psutil
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import queue
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import yaml
import os
from functools import wraps

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    source: str
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'level': self.level.value,
            'title': self.title,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'metadata': self.metadata or {}
        }

@dataclass
class Metric:
    """Metric data structure"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    labels: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'type': self.metric_type.value,
            'timestamp': self.timestamp.isoformat(),
            'labels': self.labels or {}
        }

class MetricsCollector:
    """Metrics collection and management"""
    
    def __init__(self):
        self.metrics = {}
        self.lock = threading.Lock()
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = {
                    'type': MetricType.COUNTER,
                    'value': 0.0,
                    'labels': labels or {}
                }
            self.metrics[name]['value'] += value
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric"""
        with self.lock:
            self.metrics[name] = {
                'type': MetricType.GAUGE,
                'value': value,
                'labels': labels or {}
            }
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram metric"""
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = {
                    'type': MetricType.HISTOGRAM,
                    'values': [],
                    'labels': labels or {}
                }
            self.metrics[name]['values'].append(value)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        with self.lock:
            return self.metrics.copy()
    
    def clear_metrics(self):
        """Clear all metrics"""
        with self.lock:
            self.metrics.clear()

class HealthChecker:
    """Health check system"""
    
    def __init__(self):
        self.checks = {}
        self.results = {}
    
    def register_check(self, name: str, check_func: Callable[[], bool], 
                      timeout: int = 30, critical: bool = False):
        """Register a health check"""
        self.checks[name] = {
            'function': check_func,
            'timeout': timeout,
            'critical': critical
        }
    
    def run_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {}
        
        for name, check in self.checks.items():
            try:
                start_time = time.time()
                result = check['function']()
                duration = time.time() - start_time
                
                results[name] = {
                    'status': 'healthy' if result else 'unhealthy',
                    'duration': duration,
                    'critical': check['critical'],
                    'timestamp': datetime.now().isoformat()
                }
                
            except Exception as e:
                results[name] = {
                    'status': 'error',
                    'error': str(e),
                    'critical': check['critical'],
                    'timestamp': datetime.now().isoformat()
                }
        
        self.results = results
        return results
    
    def get_overall_health(self) -> str:
        """Get overall health status"""
        if not self.results:
            return 'unknown'
        
        critical_failed = any(
            result['status'] != 'healthy' and result['critical']
            for result in self.results.values()
        )
        
        if critical_failed:
            return 'critical'
        
        unhealthy_count = sum(
            1 for result in self.results.values()
            if result['status'] != 'healthy'
        )
        
        if unhealthy_count == 0:
            return 'healthy'
        elif unhealthy_count < len(self.results) / 2:
            return 'degraded'
        else:
            return 'unhealthy'

class AlertManager:
    """Alert management and notification system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._load_config()
        self.alerts = []
        self.alert_queue = queue.Queue()
        self.notification_handlers = []
        self._setup_notification_handlers()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load alert configuration"""
        return {
            'email': {
                'enabled': os.getenv('ALERT_EMAIL_ENABLED', 'false').lower() == 'true',
                'smtp_server': os.getenv('ALERT_SMTP_SERVER', 'smtp.gmail.com'),
                'smtp_port': int(os.getenv('ALERT_SMTP_PORT', '587')),
                'username': os.getenv('ALERT_EMAIL_USERNAME', ''),
                'password': os.getenv('ALERT_EMAIL_PASSWORD', ''),
                'to_addresses': os.getenv('ALERT_EMAIL_TO', '').split(',')
            },
            'webhook': {
                'enabled': os.getenv('ALERT_WEBHOOK_ENABLED', 'false').lower() == 'true',
                'url': os.getenv('ALERT_WEBHOOK_URL', ''),
                'headers': json.loads(os.getenv('ALERT_WEBHOOK_HEADERS', '{}'))
            },
            'slack': {
                'enabled': os.getenv('ALERT_SLACK_ENABLED', 'false').lower() == 'true',
                'webhook_url': os.getenv('ALERT_SLACK_WEBHOOK', ''),
                'channel': os.getenv('ALERT_SLACK_CHANNEL', '#alerts')
            }
        }
    
    def _setup_notification_handlers(self):
        """Setup notification handlers based on configuration"""
        if self.config['email']['enabled']:
            self.notification_handlers.append(self._send_email_alert)
        
        if self.config['webhook']['enabled']:
            self.notification_handlers.append(self._send_webhook_alert)
        
        if self.config['slack']['enabled']:
            self.notification_handlers.append(self._send_slack_alert)
    
    def create_alert(self, level: AlertLevel, title: str, message: str, 
                    source: str, metadata: Optional[Dict[str, Any]] = None) -> Alert:
        """Create a new alert"""
        alert = Alert(
            id=f"{source}_{int(time.time())}",
            level=level,
            title=title,
            message=message,
            timestamp=datetime.now(),
            source=source,
            metadata=metadata
        )
        
        self.alerts.append(alert)
        self.alert_queue.put(alert)
        
        # Send notifications
        self._send_notifications(alert)
        
        return alert
    
    def _send_notifications(self, alert: Alert):
        """Send notifications for an alert"""
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Failed to send notification: {e}")
    
    def _send_email_alert(self, alert: Alert):
        """Send email alert"""
        if not self.config['email']['enabled']:
            return
        
        try:
            msg = MimeMultipart()
            msg['From'] = self.config['email']['username']
            msg['To'] = ', '.join(self.config['email']['to_addresses'])
            msg['Subject'] = f"[{alert.level.value.upper()}] {alert.title}"
            
            body = f"""
            Alert: {alert.title}
            Level: {alert.level.value.upper()}
            Source: {alert.source}
            Time: {alert.timestamp.isoformat()}
            
            Message:
            {alert.message}
            
            Metadata:
            {json.dumps(alert.metadata, indent=2) if alert.metadata else 'None'}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(self.config['email']['smtp_server'], self.config['email']['smtp_port'])
            server.starttls()
            server.login(self.config['email']['username'], self.config['email']['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent: {alert.title}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def _send_webhook_alert(self, alert: Alert):
        """Send webhook alert"""
        if not self.config['webhook']['enabled']:
            return
        
        try:
            payload = alert.to_dict()
            headers = self.config['webhook']['headers']
            
            response = requests.post(
                self.config['webhook']['url'],
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            logger.info(f"Webhook alert sent: {alert.title}")
            
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    def _send_slack_alert(self, alert: Alert):
        """Send Slack alert"""
        if not self.config['slack']['enabled']:
            return
        
        try:
            color_map = {
                AlertLevel.INFO: 'good',
                AlertLevel.WARNING: 'warning',
                AlertLevel.ERROR: 'danger',
                AlertLevel.CRITICAL: 'danger'
            }
            
            payload = {
                'channel': self.config['slack']['channel'],
                'attachments': [{
                    'color': color_map.get(alert.level, 'good'),
                    'title': alert.title,
                    'text': alert.message,
                    'fields': [
                        {'title': 'Level', 'value': alert.level.value.upper(), 'short': True},
                        {'title': 'Source', 'value': alert.source, 'short': True},
                        {'title': 'Time', 'value': alert.timestamp.isoformat(), 'short': False}
                    ]
                }]
            }
            
            response = requests.post(
                self.config['slack']['webhook_url'],
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            logger.info(f"Slack alert sent: {alert.title}")
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    def get_alerts(self, level: Optional[AlertLevel] = None, 
                  source: Optional[str] = None, 
                  limit: int = 100) -> List[Alert]:
        """Get alerts with optional filtering"""
        filtered_alerts = self.alerts
        
        if level:
            filtered_alerts = [a for a in filtered_alerts if a.level == level]
        
        if source:
            filtered_alerts = [a for a in filtered_alerts if a.source == source]
        
        return filtered_alerts[-limit:]

class PerformanceMonitor:
    """Performance monitoring and profiling"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.start_time = time.time()
    
    def monitor_function(self, func_name: str = None):
        """Decorator to monitor function performance"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                name = func_name or func.__name__
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    # Record success metrics
                    self.metrics_collector.increment_counter(f"{name}_calls")
                    self.metrics_collector.record_histogram(f"{name}_duration", duration)
                    
                    return result
                    
                except Exception as e:
                    duration = time.time() - start_time
                    
                    # Record error metrics
                    self.metrics_collector.increment_counter(f"{name}_errors")
                    self.metrics_collector.record_histogram(f"{name}_error_duration", duration)
                    
                    raise
            
            return wrapper
        return decorator
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_free = disk.free
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info().rss
            process_cpu = process.cpu_percent()
            
            # Uptime
            uptime = time.time() - self.start_time
            
            return {
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count
                },
                'memory': {
                    'percent': memory_percent,
                    'available_bytes': memory_available,
                    'process_memory_bytes': process_memory
                },
                'disk': {
                    'percent': disk_percent,
                    'free_bytes': disk_free
                },
                'process': {
                    'cpu_percent': process_cpu,
                    'memory_bytes': process_memory
                },
                'uptime_seconds': uptime
            }
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {}
    
    def get_application_metrics(self) -> Dict[str, Any]:
        """Get application-specific metrics"""
        return self.metrics_collector.get_metrics()

class MonitoringDashboard:
    """Monitoring dashboard and reporting"""
    
    def __init__(self, health_checker: HealthChecker, 
                 alert_manager: AlertManager, 
                 performance_monitor: PerformanceMonitor):
        self.health_checker = health_checker
        self.alert_manager = alert_manager
        self.performance_monitor = performance_monitor
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        return {
            'timestamp': datetime.now().isoformat(),
            'health': {
                'overall': self.health_checker.get_overall_health(),
                'checks': self.health_checker.run_checks()
            },
            'alerts': {
                'recent': [alert.to_dict() for alert in self.alert_manager.get_alerts(limit=10)],
                'counts': self._get_alert_counts()
            },
            'performance': {
                'system': self.performance_monitor.get_system_metrics(),
                'application': self.performance_monitor.get_application_metrics()
            }
        }
    
    def _get_alert_counts(self) -> Dict[str, int]:
        """Get alert counts by level"""
        counts = {}
        for level in AlertLevel:
            counts[level.value] = len(self.alert_manager.get_alerts(level=level))
        return counts
    
    def generate_report(self) -> str:
        """Generate monitoring report"""
        data = self.get_dashboard_data()
        
        report = f"""
# BRI Monitoring Report
Generated: {data['timestamp']}

## Health Status
Overall: {data['health']['overall']}

### Health Checks
"""
        
        for check_name, check_result in data['health']['checks'].items():
            status = check_result['status']
            critical = check_result.get('critical', False)
            report += f"- {check_name}: {status} {'(CRITICAL)' if critical else ''}\n"
        
        report += f"""
## Alerts
Recent Alerts: {len(data['alerts']['recent'])}
Alert Counts: {data['alerts']['counts']}

## Performance
System CPU: {data['performance']['system'].get('cpu', {}).get('percent', 'N/A')}%
System Memory: {data['performance']['system'].get('memory', {}).get('percent', 'N/A')}%
Uptime: {data['performance']['system'].get('uptime_seconds', 0):.0f} seconds
"""
        
        return report

# Global monitoring instances
metrics_collector = MetricsCollector()
health_checker = HealthChecker()
alert_manager = AlertManager()
performance_monitor = PerformanceMonitor()
monitoring_dashboard = MonitoringDashboard(health_checker, alert_manager, performance_monitor)

# Register default health checks
def check_database_connection():
    """Check database connection"""
    try:
        from database import get_database_manager
        db = get_database_manager()
        db.execute_query("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False

def check_reddit_api():
    """Check Reddit API connectivity"""
    try:
        response = requests.get("https://www.reddit.com", timeout=10)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Reddit API health check failed: {e}")
        return False

def check_gdelt_api():
    """Check GDELT API connectivity"""
    try:
        response = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", timeout=10)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"GDELT API health check failed: {e}")
        return False

# Register health checks
health_checker.register_check("database", check_database_connection, critical=True)
health_checker.register_check("reddit_api", check_reddit_api, critical=False)
health_checker.register_check("gdelt_api", check_gdelt_api, critical=False)

# Example usage
if __name__ == "__main__":
    # Test monitoring system
    print("Testing monitoring system...")
    
    # Test metrics collection
    metrics_collector.increment_counter("test_counter")
    metrics_collector.set_gauge("test_gauge", 42.0)
    metrics_collector.record_histogram("test_histogram", 1.5)
    
    print("Metrics:", metrics_collector.get_metrics())
    
    # Test health checks
    health_results = health_checker.run_checks()
    print("Health checks:", health_results)
    print("Overall health:", health_checker.get_overall_health())
    
    # Test alerts
    alert = alert_manager.create_alert(
        AlertLevel.INFO,
        "Test Alert",
        "This is a test alert",
        "monitoring_test"
    )
    print("Created alert:", alert.to_dict())
    
    # Test performance monitoring
    system_metrics = performance_monitor.get_system_metrics()
    print("System metrics:", system_metrics)
    
    # Test dashboard
    dashboard_data = monitoring_dashboard.get_dashboard_data()
    print("Dashboard data keys:", list(dashboard_data.keys()))
