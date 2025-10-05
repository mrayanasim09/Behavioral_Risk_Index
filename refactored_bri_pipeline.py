#!/usr/bin/env python3
"""
Refactored BRI Pipeline - Modular and Maintainable

This refactored version breaks down the large bri_pipeline.py into smaller,
focused modules organized by pipeline phases. Each phase is responsible for
a specific part of the pipeline, making the code more maintainable and testable.
"""

import argparse
import os
import sys
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import pipeline phases
from pipeline_phases import (
    Phase1DataCollection,
    Phase2DataPreprocessing,
    Phase3FeatureEngineering,
    Phase4BRICalculation,
    Phase5AnalysisValidation,
    Phase6Visualization,
    Phase7FinalDeliverables
)

# Import required modules
from data_collect import DataCollector
from gdelt_processor import GDELTProcessor
from preprocess import TextPreprocessor
from utils import setup_logging, ensure_directory

def main():
    """Refactored BRI Pipeline - All 7 Phases."""
    parser = argparse.ArgumentParser(description='Refactored BRI Pipeline - All 7 Phases')
    parser.add_argument('--start-date', type=str, default='2022-01-01', 
                       help='Start date for data collection (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-12-31', 
                       help='End date for data collection (YYYY-MM-DD)')
    parser.add_argument('--output-dir', type=str, default='output/refactored',
                       help='Output directory for results')
    parser.add_argument('--gdelt-file', type=str, default='20251004214500.export.CSV',
                       help='Path to GDELT export file')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging('INFO')
    logger.info("=" * 80)
    logger.info("REFACTORED BEHAVIORAL RISK INDEX PIPELINE - ALL 7 PHASES")
    logger.info("=" * 80)
    logger.info(f"Analysis Period: {args.start_date} to {args.end_date}")
    logger.info(f"Output Directory: {args.output_dir}")
    
    # Create output directory
    ensure_directory(args.output_dir)
    
    try:
        # Initialize pipeline phases
        data_collector = DataCollector()
        gdelt_processor = GDELTProcessor()
        text_preprocessor = TextPreprocessor()
        
        # PHASE 1: Data Collection
        logger.info("\n" + "="*60)
        logger.info("PHASE 1: DATA COLLECTION")
        logger.info("="*60)
        
        phase1 = Phase1DataCollection(data_collector, gdelt_processor)
        market_data = phase1.collect_market_data(args.start_date, args.end_date)
        gdelt_events = phase1.process_gdelt_data(args.gdelt_file)
        reddit_data = phase1.collect_reddit_data(args.start_date, args.end_date)
        
        # PHASE 2: Data Preprocessing
        logger.info("\n" + "="*60)
        logger.info("PHASE 2: DATA PREPROCESSING")
        logger.info("="*60)
        
        phase2 = Phase2DataPreprocessing(text_preprocessor)
        gdelt_clean = phase2.clean_gdelt_data(gdelt_events)
        reddit_clean = phase2.clean_reddit_text(reddit_data)
        sentiment_data = phase2.perform_sentiment_analysis(reddit_clean, gdelt_clean)
        
        # PHASE 3: Feature Engineering
        logger.info("\n" + "="*60)
        logger.info("PHASE 3: FEATURE ENGINEERING")
        logger.info("="*60)
        
        phase3 = Phase3FeatureEngineering()
        behavioral_features = phase3.create_behavioral_features(sentiment_data, gdelt_clean, reddit_clean)
        
        # PHASE 4: BRI Calculation
        logger.info("\n" + "="*60)
        logger.info("PHASE 4: BUILDING THE BEHAVIORAL RISK INDEX")
        logger.info("="*60)
        
        phase4 = Phase4BRICalculation()
        bri_data = phase4.calculate_bri(behavioral_features)
        
        # PHASE 5: Analysis & Validation
        logger.info("\n" + "="*60)
        logger.info("PHASE 5: ANALYSIS & VALIDATION")
        logger.info("="*60)
        
        phase5 = Phase5AnalysisValidation(data_collector)
        vix_data = phase5.collect_vix_data(args.start_date, args.end_date)
        validation_results = phase5.run_validation_analysis(bri_data, vix_data, market_data)
        backtest_results = phase5.run_economic_backtesting(bri_data, market_data)
        
        # PHASE 6: Visualization Dashboard
        logger.info("\n" + "="*60)
        logger.info("PHASE 6: VISUALIZATION DASHBOARD")
        logger.info("="*60)
        
        phase6 = Phase6Visualization()
        phase6.create_visualizations(bri_data, vix_data, market_data, args.output_dir)
        
        # PHASE 7: Final Deliverables
        logger.info("\n" + "="*60)
        logger.info("PHASE 7: FINAL DELIVERABLES")
        logger.info("="*60)
        
        phase7 = Phase7FinalDeliverables()
        deliverables = {
            'bri_pipeline': 'refactored_bri_pipeline.py',
            'bri_features': behavioral_features,
            'bri_timeseries': bri_data,
            'validation_results': validation_results,
            'backtest_results': backtest_results,
            'market_data': market_data,
            'gdelt_data': gdelt_clean,
            'reddit_data': reddit_clean,
            'sentiment_data': sentiment_data
        }
        
        phase7.save_all_deliverables(args.output_dir, deliverables)
        phase7.generate_final_report(args.output_dir, validation_results, backtest_results)
        
        logger.info("\n" + "="*80)
        logger.info("REFACTORED BRI PIPELINE - ALL 7 PHASES COMPLETED!")
        logger.info("="*80)
        logger.info(f"Results saved to: {args.output_dir}")
        
        # Print summary
        print_summary(bri_data, validation_results, backtest_results)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def print_summary(bri_data, validation_results, backtest_results):
    """Print a summary of the pipeline results"""
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    
    if not bri_data.empty:
        print(f"BRI Data Points: {len(bri_data)}")
        print(f"BRI Range: {bri_data['bri'].min():.2f} - {bri_data['bri'].max():.2f}")
        print(f"BRI Mean: {bri_data['bri'].mean():.2f}")
        print(f"BRI Std: {bri_data['bri'].std():.2f}")
    
    if validation_results:
        print(f"\nValidation Data Points: {validation_results.get('data_points', 'N/A')}")
        correlations = validation_results.get('correlations', {})
        for key, value in correlations.items():
            print(f"{key}: {value:.3f}")
    
    if backtest_results:
        print(f"\nEvents Analyzed: {backtest_results.get('events_analyzed', 'N/A')}")
        event_results = backtest_results.get('event_results', [])
        for event in event_results[:3]:  # Show first 3 events
            print(f"- {event['event']}: {event.get('bri_change', 'N/A')}")
    
    print("="*60)

if __name__ == "__main__":
    main()
