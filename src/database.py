"""
Database Connection and Operations Module

This module provides database connectivity and operations for the BRI application.
It includes connection management, data persistence, and query utilities.
"""

import os
import logging
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Union
import json
from datetime import datetime, date
import numpy as np

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Database manager for BRI application
    Handles connections, data persistence, and queries
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize database manager with configuration"""
        self.config = config or self._load_config()
        self.engine = None
        self.Session = None
        self._initialize_connection()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load database configuration from environment variables"""
        return {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'bri_db'),
            'username': os.getenv('POSTGRES_USER', 'bri_user'),
            'password': os.getenv('POSTGRES_PASSWORD', 'password'),
            'pool_size': int(os.getenv('DB_POOL_SIZE', '10')),
            'max_overflow': int(os.getenv('DB_MAX_OVERFLOW', '20')),
            'pool_timeout': int(os.getenv('DB_POOL_TIMEOUT', '30')),
            'pool_recycle': int(os.getenv('DB_POOL_RECYCLE', '3600'))
        }
    
    def _initialize_connection(self):
        """Initialize database connection and session factory"""
        try:
            # Create connection string
            connection_string = (
                f"postgresql://{self.config['username']}:{self.config['password']}"
                f"@{self.config['host']}:{self.config['port']}/{self.config['database']}"
            )
            
            # Create engine with connection pooling
            self.engine = create_engine(
                connection_string,
                poolclass=QueuePool,
                pool_size=self.config['pool_size'],
                max_overflow=self.config['max_overflow'],
                pool_timeout=self.config['pool_timeout'],
                pool_recycle=self.config['pool_recycle'],
                echo=False
            )
            
            # Create session factory
            self.Session = sessionmaker(bind=self.engine)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info("Database connection initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {e}")
            raise
    
    @contextmanager
    def get_session(self):
        """Get database session with automatic cleanup"""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                return df
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def insert_dataframe(self, df: pd.DataFrame, table_name: str, 
                        schema: str = 'bri', if_exists: str = 'append') -> bool:
        """Insert DataFrame into database table"""
        try:
            df.to_sql(
                table_name,
                self.engine,
                schema=schema,
                if_exists=if_exists,
                index=False,
                method='multi'
            )
            logger.info(f"Successfully inserted {len(df)} rows into {schema}.{table_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to insert data into {schema}.{table_name}: {e}")
            return False
    
    def upsert_dataframe(self, df: pd.DataFrame, table_name: str, 
                        conflict_columns: List[str], schema: str = 'bri') -> bool:
        """Upsert DataFrame into database table"""
        try:
            # Convert DataFrame to list of dictionaries
            records = df.to_dict('records')
            
            with self.get_session() as session:
                for record in records:
                    # Build upsert query
                    columns = list(record.keys())
                    values = list(record.values())
                    
                    # Create INSERT ... ON CONFLICT query
                    conflict_clause = ', '.join(conflict_columns)
                    update_clause = ', '.join([f"{col} = EXCLUDED.{col}" 
                                            for col in columns if col not in conflict_columns])
                    
                    query = f"""
                    INSERT INTO {schema}.{table_name} ({', '.join(columns)})
                    VALUES ({', '.join(['%s'] * len(values))})
                    ON CONFLICT ({conflict_clause})
                    DO UPDATE SET {update_clause}
                    """
                    
                    session.execute(text(query), values)
                
                session.commit()
            
            logger.info(f"Successfully upserted {len(df)} rows into {schema}.{table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert data into {schema}.{table_name}: {e}")
            return False

class BRIDataManager:
    """
    Data manager specifically for BRI application data
    Provides high-level methods for common operations
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def save_market_data(self, market_data: pd.DataFrame) -> bool:
        """Save market data to database"""
        try:
            # Ensure required columns exist
            required_columns = ['date', 'symbol', 'close']
            if not all(col in market_data.columns for col in required_columns):
                raise ValueError(f"Missing required columns: {required_columns}")
            
            # Convert date column to datetime
            market_data['date'] = pd.to_datetime(market_data['date']).dt.date
            
            return self.db.insert_dataframe(market_data, 'market_data')
        except Exception as e:
            logger.error(f"Failed to save market data: {e}")
            return False
    
    def save_reddit_data(self, reddit_data: pd.DataFrame) -> bool:
        """Save Reddit data to database"""
        try:
            # Ensure required columns exist
            required_columns = ['date', 'title', 'subreddit', 'score']
            if not all(col in reddit_data.columns for col in required_columns):
                raise ValueError(f"Missing required columns: {required_columns}")
            
            # Convert date column to datetime
            reddit_data['date'] = pd.to_datetime(reddit_data['date']).dt.date
            
            # Add reddit_id if not present
            if 'reddit_id' not in reddit_data.columns:
                reddit_data['reddit_id'] = reddit_data.index.astype(str)
            
            return self.db.upsert_dataframe(reddit_data, 'reddit_posts', ['reddit_id'])
        except Exception as e:
            logger.error(f"Failed to save Reddit data: {e}")
            return False
    
    def save_gdelt_data(self, gdelt_data: pd.DataFrame) -> bool:
        """Save GDELT data to database"""
        try:
            # Ensure required columns exist
            required_columns = ['date', 'GLOBALEVENTID', 'GoldsteinScale', 'AvgTone']
            if not all(col in gdelt_data.columns for col in required_columns):
                raise ValueError(f"Missing required columns: {required_columns}")
            
            # Convert date column to datetime
            gdelt_data['date'] = pd.to_datetime(gdelt_data['date']).dt.date
            
            # Rename columns to match database schema
            column_mapping = {
                'GLOBALEVENTID': 'global_event_id',
                'GoldsteinScale': 'goldstein_scale',
                'AvgTone': 'avg_tone',
                'NumMentions': 'num_mentions',
                'NumSources': 'num_sources',
                'NumArticles': 'num_articles',
                'SOURCEURL': 'source_url',
                'Actor1Name': 'actor1_name',
                'Actor2Name': 'actor2_name',
                'EventCode': 'event_code'
            }
            
            gdelt_data = gdelt_data.rename(columns=column_mapping)
            
            return self.db.upsert_dataframe(gdelt_data, 'gdelt_events', ['global_event_id'])
        except Exception as e:
            logger.error(f"Failed to save GDELT data: {e}")
            return False
    
    def save_sentiment_data(self, sentiment_data: pd.DataFrame) -> bool:
        """Save sentiment data to database"""
        try:
            # Ensure required columns exist
            required_columns = ['date', 'source', 'sentiment', 'confidence']
            if not all(col in sentiment_data.columns for col in required_columns):
                raise ValueError(f"Missing required columns: {required_columns}")
            
            # Convert date column to datetime
            sentiment_data['date'] = pd.to_datetime(sentiment_data['date']).dt.date
            
            return self.db.insert_dataframe(sentiment_data, 'sentiment_data')
        except Exception as e:
            logger.error(f"Failed to save sentiment data: {e}")
            return False
    
    def save_behavioral_features(self, features_data: pd.DataFrame) -> bool:
        """Save behavioral features to database"""
        try:
            # Ensure required columns exist
            required_columns = ['date']
            if not all(col in features_data.columns for col in required_columns):
                raise ValueError(f"Missing required columns: {required_columns}")
            
            # Convert date column to datetime
            features_data['date'] = pd.to_datetime(features_data['date']).dt.date
            
            return self.db.upsert_dataframe(features_data, 'behavioral_features', ['date'])
        except Exception as e:
            logger.error(f"Failed to save behavioral features: {e}")
            return False
    
    def save_bri_timeseries(self, bri_data: pd.DataFrame) -> bool:
        """Save BRI time series to database"""
        try:
            # Ensure required columns exist
            required_columns = ['date', 'bri', 'bri_normalized']
            if not all(col in bri_data.columns for col in required_columns):
                raise ValueError(f"Missing required columns: {required_columns}")
            
            # Convert date column to datetime
            bri_data['date'] = pd.to_datetime(bri_data['date']).dt.date
            
            # Add risk level and trend if not present
            if 'risk_level' not in bri_data.columns:
                bri_data['risk_level'] = bri_data['bri_normalized'].apply(
                    lambda x: 'Low' if x < 30 else 'Medium' if x < 60 else 'High'
                )
            
            if 'trend' not in bri_data.columns:
                bri_data['trend'] = 'Stable'  # Default trend
            
            return self.db.upsert_dataframe(bri_data, 'bri_timeseries', ['date'])
        except Exception as e:
            logger.error(f"Failed to save BRI time series: {e}")
            return False
    
    def save_vix_data(self, vix_data: pd.DataFrame) -> bool:
        """Save VIX data to database"""
        try:
            # Ensure required columns exist
            required_columns = ['date', 'vix']
            if not all(col in vix_data.columns for col in required_columns):
                raise ValueError(f"Missing required columns: {required_columns}")
            
            # Convert date column to datetime
            vix_data['date'] = pd.to_datetime(vix_data['date']).dt.date
            
            return self.db.upsert_dataframe(vix_data, 'vix_data', ['date'])
        except Exception as e:
            logger.error(f"Failed to save VIX data: {e}")
            return False
    
    def get_bri_summary(self, start_date: Optional[date] = None, 
                       end_date: Optional[date] = None) -> Dict[str, Any]:
        """Get BRI summary statistics"""
        try:
            query = """
            SELECT 
                AVG(bri) as avg_bri,
                STDDEV(bri) as std_bri,
                MIN(bri) as min_bri,
                MAX(bri) as max_bri,
                COUNT(*) as data_points,
                AVG(bri_normalized) as avg_bri_normalized
            FROM bri_timeseries
            """
            
            params = {}
            if start_date:
                query += " WHERE date >= :start_date"
                params['start_date'] = start_date
            
            if end_date:
                query += " AND date <= :end_date" if start_date else " WHERE date <= :end_date"
                params['end_date'] = end_date
            
            result = self.db.execute_query(query, params)
            
            if result.empty:
                return {}
            
            return result.iloc[0].to_dict()
            
        except Exception as e:
            logger.error(f"Failed to get BRI summary: {e}")
            return {}
    
    def get_correlation_analysis(self, start_date: Optional[date] = None,
                                end_date: Optional[date] = None) -> Dict[str, Any]:
        """Get correlation analysis between BRI and VIX"""
        try:
            query = """
            SELECT 
                CORR(b.bri, v.vix) as bri_vix_correlation,
                CORR(b.bri_normalized, v.vix) as bri_normalized_vix_correlation,
                COUNT(*) as data_points
            FROM bri_timeseries b
            JOIN vix_data v ON b.date = v.date
            """
            
            params = {}
            if start_date:
                query += " WHERE b.date >= :start_date"
                params['start_date'] = start_date
            
            if end_date:
                query += " AND b.date <= :end_date" if start_date else " WHERE b.date <= :end_date"
                params['end_date'] = end_date
            
            result = self.db.execute_query(query, params)
            
            if result.empty:
                return {}
            
            return result.iloc[0].to_dict()
            
        except Exception as e:
            logger.error(f"Failed to get correlation analysis: {e}")
            return {}
    
    def get_risk_level_distribution(self, start_date: Optional[date] = None,
                                   end_date: Optional[date] = None) -> pd.DataFrame:
        """Get risk level distribution"""
        try:
            query = """
            SELECT 
                risk_level,
                COUNT(*) as count_days,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
            FROM bri_timeseries
            """
            
            params = {}
            if start_date:
                query += " WHERE date >= :start_date"
                params['start_date'] = start_date
            
            if end_date:
                query += " AND date <= :end_date" if start_date else " WHERE date <= :end_date"
                params['end_date'] = end_date
            
            query += " GROUP BY risk_level ORDER BY count_days DESC"
            
            return self.db.execute_query(query, params)
            
        except Exception as e:
            logger.error(f"Failed to get risk level distribution: {e}")
            return pd.DataFrame()
    
    def get_latest_bri(self) -> Optional[Dict[str, Any]]:
        """Get latest BRI data"""
        try:
            query = """
            SELECT 
                date,
                bri,
                bri_normalized,
                risk_level,
                trend
            FROM bri_timeseries
            ORDER BY date DESC
            LIMIT 1
            """
            
            result = self.db.execute_query(query)
            
            if result.empty:
                return None
            
            return result.iloc[0].to_dict()
            
        except Exception as e:
            logger.error(f"Failed to get latest BRI: {e}")
            return None
    
    def get_bri_timeseries(self, start_date: Optional[date] = None,
                          end_date: Optional[date] = None) -> pd.DataFrame:
        """Get BRI time series data"""
        try:
            query = """
            SELECT 
                date,
                bri,
                bri_normalized,
                risk_level,
                trend
            FROM bri_timeseries
            """
            
            params = {}
            if start_date:
                query += " WHERE date >= :start_date"
                params['start_date'] = start_date
            
            if end_date:
                query += " AND date <= :end_date" if start_date else " WHERE date <= :end_date"
                params['end_date'] = end_date
            
            query += " ORDER BY date ASC"
            
            return self.db.execute_query(query, params)
            
        except Exception as e:
            logger.error(f"Failed to get BRI time series: {e}")
            return pd.DataFrame()

# Global database manager instance
db_manager = None

def get_database_manager() -> DatabaseManager:
    """Get global database manager instance"""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
    return db_manager

def get_bri_data_manager() -> BRIDataManager:
    """Get BRI data manager instance"""
    return BRIDataManager(get_database_manager())

# Example usage
if __name__ == "__main__":
    # Initialize database manager
    db = get_database_manager()
    bri_data = get_bri_data_manager()
    
    # Test connection
    try:
        result = db.execute_query("SELECT 1 as test")
        print("Database connection successful!")
        print(result)
    except Exception as e:
        print(f"Database connection failed: {e}")
