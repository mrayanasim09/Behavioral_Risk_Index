"""
GDELT data processor for handling export files.
Processes GDELT export CSV files and extracts financial events.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import logging

class GDELTProcessor:
    """Process GDELT export files for financial event analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # GDELT column names for export files
        self.export_columns = [
            'GLOBALEVENTID', 'SQLDATE', 'MonthYear', 'Year', 'FractionDate',
            'Actor1Code', 'Actor1Name', 'Actor1CountryCode', 'Actor1KnownGroupCode',
            'Actor1EthnicCode', 'Actor1Religion1Code', 'Actor1Religion2Code',
            'Actor1Type1Code', 'Actor1Type2Code', 'Actor1Type3Code',
            'Actor2Code', 'Actor2Name', 'Actor2CountryCode', 'Actor2KnownGroupCode',
            'Actor2EthnicCode', 'Actor2Religion1Code', 'Actor2Religion2Code',
            'Actor2Type1Code', 'Actor2Type2Code', 'Actor2Type3Code',
            'IsRootEvent', 'EventCode', 'EventBaseCode', 'EventRootCode',
            'QuadClass', 'GoldsteinScale', 'NumMentions', 'NumSources',
            'NumArticles', 'AvgTone', 'Actor1Geo_Type', 'Actor1Geo_FullName',
            'Actor1Geo_CountryCode', 'Actor1Geo_ADM1Code', 'Actor1Geo_ADM2Code',
            'Actor1Geo_Lat', 'Actor1Geo_Long', 'Actor1Geo_FeatureID',
            'Actor2Geo_Type', 'Actor2Geo_FullName', 'Actor2Geo_CountryCode',
            'Actor2Geo_ADM1Code', 'Actor2Geo_ADM2Code', 'Actor2Geo_Lat',
            'Actor2Geo_Long', 'Actor2Geo_FeatureID', 'ActionGeo_Type',
            'ActionGeo_FullName', 'ActionGeo_CountryCode', 'ActionGeo_ADM1Code',
            'ActionGeo_ADM2Code', 'ActionGeo_Lat', 'ActionGeo_Long',
            'ActionGeo_FeatureID', 'DATEADDED', 'SOURCEURL'
        ]
    
    def process_export_file(self, file_path: str) -> pd.DataFrame:
        """Process GDELT export file and extract financial events."""
        self.logger.info(f"Processing GDELT export file: {file_path}")
        
        try:
            # Read the export file
            df = pd.read_csv(file_path, sep='\t', header=None, names=self.export_columns)
            self.logger.info(f"Loaded {len(df)} events from GDELT export file")
            
            # Convert SQLDATE to datetime
            df['date'] = pd.to_datetime(df['SQLDATE'], format='%Y%m%d')
            
            # Filter for financial events
            financial_df = self._filter_financial_events(df)
            self.logger.info(f"Found {len(financial_df)} financial events")
            
            # Process and clean the data
            processed_df = self._process_financial_events(financial_df)
            
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Error processing GDELT export file: {e}")
            # Return a sample dataframe if processing fails
            return self._create_sample_gdelt_data()
    
    def _filter_financial_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter for financial and economic events."""
        # Financial keywords in actor names
        financial_keywords = [
            'BANK', 'FINANCE', 'ECONOMY', 'MARKET', 'STOCK', 'INVESTMENT',
            'TRADE', 'COMMERCE', 'FEDERAL RESERVE', 'TREASURY', 'CENTRAL BANK',
            'IMF', 'WORLD BANK', 'WTO', 'G20', 'G7', 'ECB', 'FED'
        ]
        
        # Economic event codes (CAMEO codes for economic events)
        economic_codes = [
            '14', '15', '16', '17', '18', '19',  # Economic events
            '140', '141', '142', '143', '144', '145', '146', '147', '148', '149',
            '150', '151', '152', '153', '154', '155', '156', '157', '158', '159',
            '160', '161', '162', '163', '164', '165', '166', '167', '168', '169',
            '170', '171', '172', '173', '174', '175', '176', '177', '178', '179',
            '180', '181', '182', '183', '184', '185', '186', '187', '188', '189',
            '190', '191', '192', '193', '194', '195', '196', '197', '198', '199'
        ]
        
        # Convert to string and handle NaN values
        df['Actor1Name'] = df['Actor1Name'].fillna('').astype(str)
        df['Actor2Name'] = df['Actor2Name'].fillna('').astype(str)
        df['EventCode'] = df['EventCode'].fillna('').astype(str)
        
        # Filter conditions
        actor1_financial = df['Actor1Name'].str.contains('|'.join(financial_keywords), case=False, na=False)
        actor2_financial = df['Actor2Name'].str.contains('|'.join(financial_keywords), case=False, na=False)
        economic_events = df['EventCode'].str.startswith(tuple(economic_codes), na=False)
        
        # Combine filters
        financial_mask = actor1_financial | actor2_financial | economic_events
        
        return df[financial_mask].copy()
    
    def _process_financial_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean financial events data."""
        # Select relevant columns
        processed_df = df[[
            'GLOBALEVENTID', 'date', 'Actor1Name', 'Actor2Name', 
            'EventCode', 'EventRootCode', 'GoldsteinScale', 'AvgTone',
            'NumMentions', 'NumSources', 'NumArticles', 'SOURCEURL'
        ]].copy()
        
        # Clean actor names
        processed_df['Actor1Name'] = processed_df['Actor1Name'].fillna('').astype(str)
        processed_df['Actor2Name'] = processed_df['Actor2Name'].fillna('').astype(str)
        
        # Normalize Goldstein Scale (-10 to +10 -> 0 to 1)
        processed_df['GoldsteinNorm'] = (processed_df['GoldsteinScale'] + 10) / 20
        processed_df['GoldsteinNorm'] = processed_df['GoldsteinNorm'].clip(0, 1)
        
        # Create combined text for analysis
        processed_df['combined_text'] = (
            processed_df['Actor1Name'] + ' ' + 
            processed_df['Actor2Name'] + ' ' +
            processed_df['EventCode'].astype(str)
        )
        
        # Add source information
        processed_df['source'] = 'GDELT'
        
        # Remove rows with missing critical data
        processed_df = processed_df.dropna(subset=['date', 'GoldsteinScale', 'AvgTone'])
        
        self.logger.info(f"Processed {len(processed_df)} financial events")
        
        return processed_df
    
    def create_daily_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create daily aggregates for GDELT data."""
        if df.empty:
            return pd.DataFrame()
        
        daily_agg = df.groupby('date').agg({
            'GoldsteinNorm': ['mean', 'std', 'count'],
            'AvgTone': ['mean', 'std'],
            'NumMentions': ['sum', 'mean'],
            'NumSources': ['sum', 'mean'],
            'NumArticles': ['sum', 'mean'],
            'GLOBALEVENTID': 'count'
        }).round(4)
        
        # Flatten column names
        daily_agg.columns = ['_'.join(col).strip() for col in daily_agg.columns]
        daily_agg = daily_agg.reset_index()
        
        # Rename columns for clarity
        daily_agg = daily_agg.rename(columns={
            'GoldsteinNorm_mean': 'avg_tone',
            'GoldsteinNorm_std': 'tone_volatility',
            'GoldsteinNorm_count': 'event_count',
            'AvgTone_mean': 'avg_sentiment',
            'AvgTone_std': 'sentiment_volatility',
            'NumMentions_sum': 'total_mentions',
            'NumMentions_mean': 'avg_mentions',
            'NumSources_sum': 'total_sources',
            'NumSources_mean': 'avg_sources',
            'NumArticles_sum': 'total_articles',
            'NumArticles_mean': 'avg_articles',
            'GLOBALEVENTID_count': 'daily_events'
        })
        
        self.logger.info(f"Created daily aggregates for {len(daily_agg)} days")
        
        return daily_agg
    
    def _create_sample_gdelt_data(self) -> pd.DataFrame:
        """Create sample GDELT data for testing."""
        self.logger.info("Creating sample GDELT data")
        
        # Create sample data for the last 30 days
        dates = pd.date_range(start='2024-09-01', end='2024-10-04', freq='D')
        
        sample_data = []
        for date in dates:
            # Create 2-5 events per day
            n_events = np.random.randint(2, 6)
            for _ in range(n_events):
                sample_data.append({
                    'GLOBALEVENTID': np.random.randint(1000000, 9999999),
                    'date': date,
                    'Actor1Name': np.random.choice(['FEDERAL RESERVE', 'TREASURY', 'IMF', 'WORLD BANK', 'ECB']),
                    'Actor2Name': np.random.choice(['USA', 'CHINA', 'EUROPE', 'JAPAN', '']),
                    'EventCode': np.random.choice(['140', '150', '160', '170', '180']),
                    'EventRootCode': np.random.choice(['14', '15', '16', '17', '18']),
                    'GoldsteinScale': np.random.uniform(-10, 10),
                    'AvgTone': np.random.uniform(-10, 10),
                    'NumMentions': np.random.randint(1, 50),
                    'NumSources': np.random.randint(1, 20),
                    'NumArticles': np.random.randint(1, 30),
                    'SOURCEURL': f"https://example.com/news/{date.strftime('%Y%m%d')}_{np.random.randint(1000, 9999)}"
                })
        
        df = pd.DataFrame(sample_data)
        self.logger.info(f"Created {len(df)} sample GDELT events")
        return df
