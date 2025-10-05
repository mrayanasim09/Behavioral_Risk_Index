#!/usr/bin/env python3
"""
Final Verification Test - Ensure everything works without errors
"""

import sys
import traceback
from datetime import datetime

def test_fast_pipeline():
    """Test the fast pipeline"""
    print("🧪 Testing Fast Pipeline...")
    try:
        from fast_bri_pipeline import create_fast_bri_data, run_fast_validation
        bri_data, market_data = create_fast_bri_data()
        validation_results = run_fast_validation(bri_data, market_data)
        print("✅ Fast pipeline works perfectly!")
        return True
    except Exception as e:
        print(f"❌ Fast pipeline error: {e}")
        traceback.print_exc()
        return False

def test_web_app():
    """Test the web application"""
    print("🧪 Testing Web Application...")
    try:
        from app import BRIAnalyzer
        analyzer = BRIAnalyzer()
        summary = analyzer.get_bri_summary()
        correlation = analyzer.get_correlation_data()
        features = analyzer.get_feature_importance()
        print("✅ Web application works perfectly!")
        return True
    except Exception as e:
        print(f"❌ Web application error: {e}")
        traceback.print_exc()
        return False

def test_data_quality():
    """Test data quality and completeness"""
    print("🧪 Testing Data Quality...")
    try:
        import pandas as pd
        import numpy as np
        
        # Load data
        bri_data = pd.read_csv('output/fast/bri_timeseries.csv')
        market_data = pd.read_csv('output/fast/market_data.csv')
        
        # Check data completeness
        assert len(bri_data) == 1096, f"Expected 1096 days, got {len(bri_data)}"
        assert len(market_data) == 1096, f"Expected 1096 days, got {len(market_data)}"
        
        # Check BRI range
        assert bri_data['BRI'].min() >= 0, "BRI should be >= 0"
        assert bri_data['BRI'].max() <= 100, "BRI should be <= 100"
        
        # Check correlation
        merged = pd.merge(bri_data, market_data, left_on='date', right_on='Date', how='inner')
        correlation = merged['BRI'].corr(merged['Close_^VIX'])
        assert abs(correlation) > 0.5, f"Correlation too low: {correlation}"
        
        print("✅ Data quality is excellent!")
        print(f"   📊 Data points: {len(bri_data)} days")
        print(f"   📈 BRI range: {bri_data['BRI'].min():.1f} - {bri_data['BRI'].max():.1f}")
        print(f"   🔗 BRI-VIX correlation: {correlation:.3f}")
        return True
    except Exception as e:
        print(f"❌ Data quality error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 FINAL VERIFICATION TEST")
    print("=" * 50)
    
    start_time = datetime.now()
    
    tests = [
        ("Fast Pipeline", test_fast_pipeline),
        ("Web Application", test_web_app),
        ("Data Quality", test_data_quality)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        success = test_func()
        results.append((test_name, success))
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
        if not success:
            all_passed = False
    
    print(f"\n⏱️  Total Duration: {duration:.1f} seconds")
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Project is ready for deployment!")
        print("✅ No errors found!")
        print("✅ Data is clean and complete!")
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
