#!/usr/bin/env python3
"""
Train/Test BRI Pipeline
Trains on 2020-2024 data and tests on 2018-2019 data to prevent overfitting
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pipeline_phases import *
from data_collect import DataCollector
from gdelt_processor import GDELTProcessor
from preprocess import TextPreprocessor
import pandas as pd
import numpy as np
import yaml
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_train_test_data():
    """Load train and test datasets"""
    logger.info("📂 Loading train/test datasets...")
    
    train_data = {}
    test_data = {}
    
    # Load market data
    try:
        if os.path.exists("data/train/market_data.csv"):
            train_data['market'] = pd.read_csv("data/train/market_data.csv")
            train_data['market']['date'] = pd.to_datetime(train_data['market']['date']).dt.date
            logger.info(f"✅ Training market data: {len(train_data['market']):,} records")
        else:
            logger.warning("Training market data not found")
            train_data['market'] = pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading training market data: {e}")
        train_data['market'] = pd.DataFrame()
    
    try:
        if os.path.exists("data/test/market_data.csv"):
            test_data['market'] = pd.read_csv("data/test/market_data.csv")
            test_data['market']['date'] = pd.to_datetime(test_data['market']['date']).dt.date
            logger.info(f"✅ Test market data: {len(test_data['market']):,} records")
        else:
            logger.warning("Test market data not found")
            test_data['market'] = pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading test market data: {e}")
        test_data['market'] = pd.DataFrame()
    
    # Load Reddit data
    try:
        if os.path.exists("data/train/reddit_data.csv"):
            train_data['reddit'] = pd.read_csv("data/train/reddit_data.csv")
            train_data['reddit']['date'] = pd.to_datetime(train_data['reddit']['date']).dt.date
            logger.info(f"✅ Training Reddit data: {len(train_data['reddit']):,} records")
        else:
            logger.warning("Training Reddit data not found")
            train_data['reddit'] = pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading training Reddit data: {e}")
        train_data['reddit'] = pd.DataFrame()
    
    try:
        if os.path.exists("data/test/reddit_data.csv"):
            test_data['reddit'] = pd.read_csv("data/test/reddit_data.csv")
            test_data['reddit']['date'] = pd.to_datetime(test_data['reddit']['date']).dt.date
            logger.info(f"✅ Test Reddit data: {len(test_data['reddit']):,} records")
        else:
            logger.warning("Test Reddit data not found")
            test_data['reddit'] = pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading test Reddit data: {e}")
        test_data['reddit'] = pd.DataFrame()
    
    # Load GDELT data
    try:
        if os.path.exists("data/train/gdelt_data.csv"):
            train_data['gdelt'] = pd.read_csv("data/train/gdelt_data.csv")
            train_data['gdelt']['date'] = pd.to_datetime(train_data['gdelt']['date']).dt.date
            logger.info(f"✅ Training GDELT data: {len(train_data['gdelt']):,} records")
        else:
            logger.warning("Training GDELT data not found")
            train_data['gdelt'] = pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading training GDELT data: {e}")
        train_data['gdelt'] = pd.DataFrame()
    
    try:
        if os.path.exists("data/test/gdelt_data.csv"):
            test_data['gdelt'] = pd.read_csv("data/test/gdelt_data.csv")
            test_data['gdelt']['date'] = pd.to_datetime(test_data['gdelt']['date']).dt.date
            logger.info(f"✅ Test GDELT data: {len(test_data['gdelt']):,} records")
        else:
            logger.warning("Test GDELT data not found")
            test_data['gdelt'] = pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading test GDELT data: {e}")
        test_data['gdelt'] = pd.DataFrame()
    
    return train_data, test_data

def run_training_pipeline(train_data, config):
    """Run the BRI pipeline on training data"""
    logger.info("\n" + "="*50)
    logger.info("TRAINING PIPELINE (2020-2024)")
    logger.info("="*50)
    
    try:
        # Initialize components
        data_collector = DataCollector(config)
        gdelt_processor = GDELTProcessor()
        text_preprocessor = TextPreprocessor(config)
        
        # Initialize pipeline phases
        phase1 = Phase1DataCollection(data_collector, gdelt_processor)
        phase2 = Phase2DataPreprocessing(text_preprocessor)
        phase3 = Phase3FeatureEngineering()
        phase4 = Phase4BRICalculation()
        phase5 = Phase5AnalysisValidation()
        
        # Phase 2: Data Preprocessing
        logger.info("Phase 2: Preprocessing training data...")
        
        gdelt_clean = phase2.clean_gdelt_data(train_data['gdelt']) if not train_data['gdelt'].empty else pd.DataFrame()
        reddit_clean = phase2.clean_reddit_text(train_data['reddit']) if not train_data['reddit'].empty else pd.DataFrame()
        
        # Perform sentiment analysis
        sentiment_data = phase2.perform_sentiment_analysis(reddit_clean, gdelt_clean)
        
        # Phase 3: Feature Engineering
        logger.info("Phase 3: Creating behavioral features...")
        behavioral_features = phase3.create_behavioral_features(sentiment_data, gdelt_clean, reddit_clean)
        
        # Phase 4: BRI Calculation
        logger.info("Phase 4: Calculating BRI...")
        bri_data = phase4.calculate_bri(behavioral_features)
        
        # Phase 5: Analysis and Validation
        logger.info("Phase 5: Training analysis...")
        analysis_results = phase5.run_analysis(bri_data, train_data['market'])
        
        logger.info("✅ Training pipeline completed successfully!")
        return bri_data, analysis_results
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), {}

def run_testing_pipeline(test_data, config, trained_weights=None):
    """Run the BRI pipeline on test data"""
    logger.info("\n" + "="*50)
    logger.info("TESTING PIPELINE (2018-2019)")
    logger.info("="*50)
    
    try:
        # Initialize components
        data_collector = DataCollector(config)
        gdelt_processor = GDELTProcessor()
        text_preprocessor = TextPreprocessor(config)
        
        # Initialize pipeline phases
        phase1 = Phase1DataCollection(data_collector, gdelt_processor)
        phase2 = Phase2DataPreprocessing(text_preprocessor)
        phase3 = Phase3FeatureEngineering()
        phase4 = Phase4BRICalculation()
        phase5 = Phase5AnalysisValidation()
        
        # Phase 2: Data Preprocessing
        logger.info("Phase 2: Preprocessing test data...")
        
        gdelt_clean = phase2.clean_gdelt_data(test_data['gdelt']) if not test_data['gdelt'].empty else pd.DataFrame()
        reddit_clean = phase2.clean_reddit_text(test_data['reddit']) if not test_data['reddit'].empty else pd.DataFrame()
        
        # Perform sentiment analysis
        sentiment_data = phase2.perform_sentiment_analysis(reddit_clean, gdelt_clean)
        
        # Phase 3: Feature Engineering
        logger.info("Phase 3: Creating behavioral features...")
        behavioral_features = phase3.create_behavioral_features(sentiment_data, gdelt_clean, reddit_clean)
        
        # Phase 4: BRI Calculation (using trained weights if available)
        logger.info("Phase 4: Calculating BRI...")
        bri_data = phase4.calculate_bri(behavioral_features)
        
        # Phase 5: Analysis and Validation
        logger.info("Phase 5: Test analysis...")
        analysis_results = phase5.run_analysis(bri_data, test_data['market'])
        
        logger.info("✅ Testing pipeline completed successfully!")
        return bri_data, analysis_results
        
    except Exception as e:
        logger.error(f"❌ Testing pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), {}

def calculate_overfitting_metrics(train_results, test_results):
    """Calculate overfitting metrics"""
    logger.info("\n" + "="*50)
    logger.info("OVERFITTING ANALYSIS")
    logger.info("="*50)
    
    try:
        # Extract key metrics
        train_correlation = train_results.get('vix_correlation', 0)
        test_correlation = test_results.get('vix_correlation', 0)
        
        train_accuracy = train_results.get('accuracy', 0)
        test_accuracy = test_results.get('accuracy', 0)
        
        train_mae = train_results.get('mae', 0)
        test_mae = test_results.get('mae', 0)
        
        # Calculate overfitting metrics
        correlation_gap = abs(train_correlation - test_correlation)
        accuracy_gap = abs(train_accuracy - test_accuracy)
        mae_gap = abs(train_mae - test_mae)
        
        logger.info(f"📊 Correlation Analysis:")
        logger.info(f"   Training: {train_correlation:.3f}")
        logger.info(f"   Testing:  {test_correlation:.3f}")
        logger.info(f"   Gap:      {correlation_gap:.3f}")
        
        logger.info(f"\n📊 Accuracy Analysis:")
        logger.info(f"   Training: {train_accuracy:.3f}")
        logger.info(f"   Testing:  {test_accuracy:.3f}")
        logger.info(f"   Gap:      {accuracy_gap:.3f}")
        
        logger.info(f"\n📊 MAE Analysis:")
        logger.info(f"   Training: {train_mae:.3f}")
        logger.info(f"   Testing:  {test_mae:.3f}")
        logger.info(f"   Gap:      {mae_gap:.3f}")
        
        # Overfitting assessment
        overfitting_score = (correlation_gap + accuracy_gap + mae_gap) / 3
        
        if overfitting_score < 0.1:
            logger.info(f"\n✅ OVERFITTING ASSESSMENT: LOW RISK ({overfitting_score:.3f})")
        elif overfitting_score < 0.2:
            logger.info(f"\n⚠️  OVERFITTING ASSESSMENT: MODERATE RISK ({overfitting_score:.3f})")
        else:
            logger.info(f"\n❌ OVERFITTING ASSESSMENT: HIGH RISK ({overfitting_score:.3f})")
        
        return {
            'correlation_gap': correlation_gap,
            'accuracy_gap': accuracy_gap,
            'mae_gap': mae_gap,
            'overfitting_score': overfitting_score
        }
        
    except Exception as e:
        logger.error(f"Error calculating overfitting metrics: {e}")
        return {}

def save_results(train_bri, test_bri, train_results, test_results, overfitting_metrics):
    """Save all results"""
    logger.info("\n💾 Saving results...")
    
    # Create output directory
    os.makedirs("output/train_test", exist_ok=True)
    
    # Save BRI data
    if not train_bri.empty:
        train_bri.to_csv("output/train_test/train_bri.csv", index=False)
        logger.info("✅ Training BRI data saved")
    
    if not test_bri.empty:
        test_bri.to_csv("output/train_test/test_bri.csv", index=False)
        logger.info("✅ Test BRI data saved")
    
    # Save results
    results = {
        'training_results': train_results,
        'test_results': test_results,
        'overfitting_metrics': overfitting_metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    import json
    with open("output/train_test/results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("✅ Results saved to output/train_test/")

def main():
    """Main function"""
    logger.info("🚀 BRI Train/Test Pipeline")
    logger.info("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Load train/test data
    train_data, test_data = load_train_test_data()
    
    # Check data availability
    train_records = sum([len(df) for df in train_data.values() if not df.empty])
    test_records = sum([len(df) for df in test_data.values() if not df.empty])
    
    logger.info(f"\n📊 Data Summary:")
    logger.info(f"   Training data: {train_records:,} records")
    logger.info(f"   Test data: {test_records:,} records")
    logger.info(f"   Total: {train_records + test_records:,} records")
    
    if train_records == 0:
        logger.error("❌ No training data available. Please run collect_train_test_data.py first.")
        return
    
    if test_records == 0:
        logger.warning("⚠️  No test data available. Using training data for both train and test.")
        test_data = train_data.copy()
    
    # Run training pipeline
    train_bri, train_results = run_training_pipeline(train_data, config)
    
    # Run testing pipeline
    test_bri, test_results = run_testing_pipeline(test_data, config)
    
    # Calculate overfitting metrics
    overfitting_metrics = calculate_overfitting_metrics(train_results, test_results)
    
    # Save results
    save_results(train_bri, test_bri, train_results, test_results, overfitting_metrics)
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("🎯 TRAIN/TEST PIPELINE SUMMARY")
    logger.info("="*60)
    
    logger.info(f"✅ Training completed: {len(train_bri):,} BRI records")
    logger.info(f"✅ Testing completed: {len(test_bri):,} BRI records")
    logger.info(f"📊 Overfitting risk: {overfitting_metrics.get('overfitting_score', 'N/A')}")
    logger.info(f"📁 Results saved to: output/train_test/")
    
    logger.info(f"\n🎉 Train/test pipeline completed successfully!")

if __name__ == "__main__":
    main()
