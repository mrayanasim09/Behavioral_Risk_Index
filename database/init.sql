-- Database initialization script for Behavioral Risk Index (BRI)
-- This script creates the necessary tables and indexes for the BRI application

-- Create database if it doesn't exist
CREATE DATABASE IF NOT EXISTS bri_db;

-- Use the database
\c bri_db;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS bri;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Set search path
SET search_path TO bri, public;

-- Create tables

-- Market data table
CREATE TABLE IF NOT EXISTS market_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    date DATE NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    open DECIMAL(10, 2),
    high DECIMAL(10, 2),
    low DECIMAL(10, 2),
    close DECIMAL(10, 2),
    volume BIGINT,
    adjusted_close DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- GDELT events table
CREATE TABLE IF NOT EXISTS gdelt_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    global_event_id BIGINT UNIQUE NOT NULL,
    date DATE NOT NULL,
    actor1_name VARCHAR(255),
    actor2_name VARCHAR(255),
    event_code VARCHAR(10),
    goldstein_scale DECIMAL(5, 2),
    avg_tone DECIMAL(8, 2),
    num_mentions INTEGER,
    num_sources INTEGER,
    num_articles INTEGER,
    source_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Reddit posts table
CREATE TABLE IF NOT EXISTS reddit_posts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    reddit_id VARCHAR(20) UNIQUE NOT NULL,
    date DATE NOT NULL,
    title TEXT NOT NULL,
    selftext TEXT,
    subreddit VARCHAR(50) NOT NULL,
    score INTEGER DEFAULT 0,
    num_comments INTEGER DEFAULT 0,
    author VARCHAR(50),
    url TEXT,
    is_self BOOLEAN DEFAULT FALSE,
    upvote_ratio DECIMAL(3, 2),
    flair_text VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sentiment data table
CREATE TABLE IF NOT EXISTS sentiment_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    date DATE NOT NULL,
    source VARCHAR(20) NOT NULL, -- 'reddit', 'gdelt', 'news'
    sentiment DECIMAL(5, 4) NOT NULL, -- -1 to 1
    confidence DECIMAL(5, 4) NOT NULL, -- 0 to 1
    engagement INTEGER DEFAULT 0,
    subreddit VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Behavioral features table
CREATE TABLE IF NOT EXISTS behavioral_features (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    date DATE NOT NULL,
    sentiment_volatility DECIMAL(10, 6),
    news_tone DECIMAL(10, 6),
    media_herding DECIMAL(10, 6),
    polarity_skew DECIMAL(10, 6),
    event_density DECIMAL(10, 6),
    engagement_index DECIMAL(10, 6),
    fear_index DECIMAL(10, 6),
    overconfidence_index DECIMAL(10, 6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- BRI time series table
CREATE TABLE IF NOT EXISTS bri_timeseries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    date DATE NOT NULL UNIQUE,
    bri DECIMAL(10, 6) NOT NULL,
    bri_normalized DECIMAL(5, 2) NOT NULL, -- 0 to 100
    risk_level VARCHAR(20) NOT NULL, -- 'Low', 'Medium', 'High'
    trend VARCHAR(20), -- 'Rising', 'Falling', 'Stable'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- VIX data table
CREATE TABLE IF NOT EXISTS vix_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    date DATE NOT NULL UNIQUE,
    vix DECIMAL(8, 4) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Validation results table
CREATE TABLE IF NOT EXISTS validation_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    date DATE NOT NULL,
    correlation_bri_vix DECIMAL(8, 6),
    correlation_bri_market DECIMAL(8, 6),
    r_squared DECIMAL(8, 6),
    data_points INTEGER,
    validation_score DECIMAL(5, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Backtest results table
CREATE TABLE IF NOT EXISTS backtest_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_name VARCHAR(255) NOT NULL,
    event_date DATE NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    bri_before DECIMAL(10, 6),
    bri_after DECIMAL(10, 6),
    bri_change DECIMAL(10, 6),
    market_impact DECIMAL(10, 6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model performance table
CREATE TABLE IF NOT EXISTS model_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(50) NOT NULL,
    metric_value DECIMAL(10, 6) NOT NULL,
    evaluation_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- API usage tracking table
CREATE TABLE IF NOT EXISTS api_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    endpoint VARCHAR(100) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER NOT NULL,
    response_time_ms INTEGER,
    user_agent TEXT,
    ip_address INET,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance

-- Market data indexes
CREATE INDEX IF NOT EXISTS idx_market_data_date ON market_data(date);
CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data(symbol);
CREATE INDEX IF NOT EXISTS idx_market_data_date_symbol ON market_data(date, symbol);

-- GDELT events indexes
CREATE INDEX IF NOT EXISTS idx_gdelt_events_date ON gdelt_events(date);
CREATE INDEX IF NOT EXISTS idx_gdelt_events_global_event_id ON gdelt_events(global_event_id);
CREATE INDEX IF NOT EXISTS idx_gdelt_events_goldstein_scale ON gdelt_events(goldstein_scale);

-- Reddit posts indexes
CREATE INDEX IF NOT EXISTS idx_reddit_posts_date ON reddit_posts(date);
CREATE INDEX IF NOT EXISTS idx_reddit_posts_subreddit ON reddit_posts(subreddit);
CREATE INDEX IF NOT EXISTS idx_reddit_posts_score ON reddit_posts(score);
CREATE INDEX IF NOT EXISTS idx_reddit_posts_date_subreddit ON reddit_posts(date, subreddit);

-- Sentiment data indexes
CREATE INDEX IF NOT EXISTS idx_sentiment_data_date ON sentiment_data(date);
CREATE INDEX IF NOT EXISTS idx_sentiment_data_source ON sentiment_data(source);
CREATE INDEX IF NOT EXISTS idx_sentiment_data_date_source ON sentiment_data(date, source);

-- Behavioral features indexes
CREATE INDEX IF NOT EXISTS idx_behavioral_features_date ON behavioral_features(date);

-- BRI time series indexes
CREATE INDEX IF NOT EXISTS idx_bri_timeseries_date ON bri_timeseries(date);
CREATE INDEX IF NOT EXISTS idx_bri_timeseries_risk_level ON bri_timeseries(risk_level);
CREATE INDEX IF NOT EXISTS idx_bri_timeseries_trend ON bri_timeseries(trend);

-- VIX data indexes
CREATE INDEX IF NOT EXISTS idx_vix_data_date ON vix_data(date);

-- Validation results indexes
CREATE INDEX IF NOT EXISTS idx_validation_results_date ON validation_results(date);

-- Backtest results indexes
CREATE INDEX IF NOT EXISTS idx_backtest_results_event_date ON backtest_results(event_date);
CREATE INDEX IF NOT EXISTS idx_backtest_results_event_type ON backtest_results(event_type);

-- Model performance indexes
CREATE INDEX IF NOT EXISTS idx_model_performance_model_name ON model_performance(model_name);
CREATE INDEX IF NOT EXISTS idx_model_performance_evaluation_date ON model_performance(evaluation_date);

-- API usage indexes
CREATE INDEX IF NOT EXISTS idx_api_usage_endpoint ON api_usage(endpoint);
CREATE INDEX IF NOT EXISTS idx_api_usage_created_at ON api_usage(created_at);
CREATE INDEX IF NOT EXISTS idx_api_usage_status_code ON api_usage(status_code);

-- Create views for common queries

-- BRI summary view
CREATE OR REPLACE VIEW bri_summary AS
SELECT 
    date,
    bri,
    bri_normalized,
    risk_level,
    trend,
    CASE 
        WHEN bri_normalized < 30 THEN 'Low'
        WHEN bri_normalized < 60 THEN 'Medium'
        ELSE 'High'
    END as calculated_risk_level
FROM bri_timeseries
ORDER BY date DESC;

-- Correlation analysis view
CREATE OR REPLACE VIEW correlation_analysis AS
SELECT 
    v.date,
    v.vix,
    b.bri,
    b.bri_normalized,
    m.close as market_close,
    v.vix - LAG(v.vix) OVER (ORDER BY v.date) as vix_change,
    b.bri - LAG(b.bri) OVER (ORDER BY b.date) as bri_change
FROM vix_data v
JOIN bri_timeseries b ON v.date = b.date
LEFT JOIN market_data m ON v.date = m.date AND m.symbol = '^GSPC'
ORDER BY v.date DESC;

-- Sentiment analysis view
CREATE OR REPLACE VIEW sentiment_analysis AS
SELECT 
    date,
    source,
    AVG(sentiment) as avg_sentiment,
    STDDEV(sentiment) as sentiment_volatility,
    COUNT(*) as record_count,
    AVG(confidence) as avg_confidence
FROM sentiment_data
GROUP BY date, source
ORDER BY date DESC, source;

-- Create functions

-- Function to calculate BRI statistics
CREATE OR REPLACE FUNCTION calculate_bri_stats(start_date DATE, end_date DATE)
RETURNS TABLE (
    avg_bri DECIMAL(10, 6),
    std_bri DECIMAL(10, 6),
    min_bri DECIMAL(10, 6),
    max_bri DECIMAL(10, 6),
    count_days INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        AVG(bri) as avg_bri,
        STDDEV(bri) as std_bri,
        MIN(bri) as min_bri,
        MAX(bri) as max_bri,
        COUNT(*)::INTEGER as count_days
    FROM bri_timeseries
    WHERE date BETWEEN start_date AND end_date;
END;
$$ LANGUAGE plpgsql;

-- Function to get risk level distribution
CREATE OR REPLACE FUNCTION get_risk_level_distribution(start_date DATE, end_date DATE)
RETURNS TABLE (
    risk_level VARCHAR(20),
    count_days INTEGER,
    percentage DECIMAL(5, 2)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        b.risk_level,
        COUNT(*)::INTEGER as count_days,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
    FROM bri_timeseries b
    WHERE b.date BETWEEN start_date AND end_date
    GROUP BY b.risk_level
    ORDER BY count_days DESC;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for updated_at timestamps

-- Function to update updated_at column
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply triggers to all tables with updated_at column
DO $$
DECLARE
    table_name TEXT;
BEGIN
    FOR table_name IN 
        SELECT tablename FROM pg_tables 
        WHERE schemaname = 'bri' 
        AND tablename IN (
            'market_data', 'gdelt_events', 'reddit_posts', 'sentiment_data',
            'behavioral_features', 'bri_timeseries', 'vix_data', 
            'validation_results', 'backtest_results', 'model_performance'
        )
    LOOP
        EXECUTE format('
            CREATE TRIGGER update_%s_updated_at
            BEFORE UPDATE ON %I
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column()',
            table_name, table_name);
    END LOOP;
END $$;

-- Insert initial data

-- Insert sample BRI data
INSERT INTO bri_timeseries (date, bri, bri_normalized, risk_level, trend) VALUES
('2024-01-01', 45.2, 45.2, 'Medium', 'Stable'),
('2024-01-02', 47.1, 47.1, 'Medium', 'Rising'),
('2024-01-03', 52.3, 52.3, 'Medium', 'Rising')
ON CONFLICT (date) DO NOTHING;

-- Create monitoring schema tables
SET search_path TO monitoring, public;

-- System metrics table
CREATE TABLE IF NOT EXISTS system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15, 6) NOT NULL,
    tags JSONB
);

-- Application logs table
CREATE TABLE IF NOT EXISTS application_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    level VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    module VARCHAR(100),
    function_name VARCHAR(100),
    line_number INTEGER,
    extra_data JSONB
);

-- Create indexes for monitoring tables
CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_system_metrics_name ON system_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_application_logs_timestamp ON application_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_application_logs_level ON application_logs(level);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA bri TO bri_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA monitoring TO bri_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA bri TO bri_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA monitoring TO bri_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA bri TO bri_user;
GRANT USAGE ON SCHEMA bri TO bri_user;
GRANT USAGE ON SCHEMA monitoring TO bri_user;

-- Set default privileges
ALTER DEFAULT PRIVILEGES IN SCHEMA bri GRANT ALL ON TABLES TO bri_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA monitoring GRANT ALL ON TABLES TO bri_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA bri GRANT ALL ON SEQUENCES TO bri_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA monitoring GRANT ALL ON SEQUENCES TO bri_user;
