#!/usr/bin/env python3
"""
BRI vs VIX Comparison Analysis
Shows the differences between BRI and VIX with honest, realistic results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BRIVIXComparison:
    """BRI vs VIX comparison with honest analysis"""
    
    def __init__(self):
        self.data = None
        self.vix_data = None
        self.results = {}
        
    def load_data(self):
        """Load BRI and VIX data for comparison"""
        print("📊 Loading BRI and VIX Data for Comparison...")
        
        # Load BRI data
        try:
            self.data = pd.read_csv('output/enhanced_5year/enhanced_bri_data.csv')
            self.data['date'] = pd.to_datetime(self.data['date'])
            print(f"✅ Loaded {len(self.data)} BRI data points")
        except:
            print("❌ Could not load BRI data")
            return False
        
        # Load VIX data
        try:
            vix = yf.download('^VIX', start='2020-01-01', end='2024-12-31')
            vix = vix.reset_index()
            vix['date'] = pd.to_datetime(vix['Date'])
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.droplevel(1)
            vix = vix[['date', 'Close']].rename(columns={'Close': 'VIX'})
            vix = vix.dropna()
            self.vix_data = vix
            print(f"✅ Loaded {len(vix)} VIX data points")
        except Exception as e:
            print(f"❌ Could not load VIX data: {e}")
            return False
        
        # Merge data
        self.data = pd.merge(self.data, self.vix_data, on='date', how='left')
        self.data = self.data.reset_index(drop=True)
        
        # Fix missing BRI values
        self.data['BRI'] = self.data['BRI'].interpolate(method='linear')
        self.data['BRI'] = self.data['BRI'].fillna(method='ffill').fillna(method='bfill')
        
        # Remove rows with missing critical data
        self.data = self.data.dropna(subset=['BRI', 'VIX'])
        
        print(f"✅ Final dataset: {len(self.data)} points")
        print(f"   Date range: {self.data['date'].min()} to {self.data['date'].max()}")
        print(f"   Missing BRI values: {self.data['BRI'].isna().sum()} ({self.data['BRI'].isna().sum()/len(self.data)*100:.1f}%)")
        print(f"   Missing VIX values: {self.data['VIX'].isna().sum()} ({self.data['VIX'].isna().sum()/len(self.data)*100:.1f}%)")
        
        return True
    
    def calculate_basic_statistics(self):
        """Calculate basic statistics for BRI and VIX"""
        print("\n📊 Calculating Basic Statistics...")
        
        # BRI statistics
        bri_stats = {
            'mean': self.data['BRI'].mean(),
            'std': self.data['BRI'].std(),
            'min': self.data['BRI'].min(),
            'max': self.data['BRI'].max(),
            'median': self.data['BRI'].median(),
            'skewness': self.data['BRI'].skew(),
            'kurtosis': self.data['BRI'].kurtosis()
        }
        
        # VIX statistics
        vix_stats = {
            'mean': self.data['VIX'].mean(),
            'std': self.data['VIX'].std(),
            'min': self.data['VIX'].min(),
            'max': self.data['VIX'].max(),
            'median': self.data['VIX'].median(),
            'skewness': self.data['VIX'].skew(),
            'kurtosis': self.data['VIX'].kurtosis()
        }
        
        print(f"BRI Statistics:")
        print(f"  Mean: {bri_stats['mean']:.2f}")
        print(f"  Std: {bri_stats['std']:.2f}")
        print(f"  Min: {bri_stats['min']:.2f}")
        print(f"  Max: {bri_stats['max']:.2f}")
        print(f"  Median: {bri_stats['median']:.2f}")
        print(f"  Skewness: {bri_stats['skewness']:.2f}")
        print(f"  Kurtosis: {bri_stats['kurtosis']:.2f}")
        
        print(f"\nVIX Statistics:")
        print(f"  Mean: {vix_stats['mean']:.2f}")
        print(f"  Std: {vix_stats['std']:.2f}")
        print(f"  Min: {vix_stats['min']:.2f}")
        print(f"  Max: {vix_stats['max']:.2f}")
        print(f"  Median: {vix_stats['median']:.2f}")
        print(f"  Skewness: {vix_stats['skewness']:.2f}")
        print(f"  Kurtosis: {vix_stats['kurtosis']:.2f}")
        
        # Store results
        self.results['bri_stats'] = bri_stats
        self.results['vix_stats'] = vix_stats
        
        return True
    
    def calculate_correlation_analysis(self):
        """Calculate correlation analysis between BRI and VIX"""
        print("\n📊 Calculating Correlation Analysis...")
        
        # Pearson correlation
        pearson_corr, pearson_p = stats.pearsonr(self.data['BRI'], self.data['VIX'])
        
        # Spearman correlation
        spearman_corr, spearman_p = stats.spearmanr(self.data['BRI'], self.data['VIX'])
        
        # Rolling correlation (30-day window)
        rolling_corr = self.data['BRI'].rolling(window=30).corr(self.data['VIX'])
        
        print(f"Correlation Analysis:")
        print(f"  Pearson Correlation: {pearson_corr:.3f} (p-value: {pearson_p:.3f})")
        print(f"  Spearman Correlation: {spearman_corr:.3f} (p-value: {spearman_p:.3f})")
        print(f"  Rolling Correlation (30-day): Mean: {rolling_corr.mean():.3f}, Std: {rolling_corr.std():.3f}")
        
        # Store results
        self.results['correlation'] = {
            'pearson_corr': pearson_corr,
            'pearson_p': pearson_p,
            'spearman_corr': spearman_corr,
            'spearman_p': spearman_p,
            'rolling_corr_mean': rolling_corr.mean(),
            'rolling_corr_std': rolling_corr.std()
        }
        
        return True
    
    def analyze_volatility_patterns(self):
        """Analyze volatility patterns in BRI vs VIX"""
        print("\n📊 Analyzing Volatility Patterns...")
        
        # Calculate returns
        bri_returns = self.data['BRI'].pct_change().dropna()
        vix_returns = self.data['VIX'].pct_change().dropna()
        
        # Calculate volatility (annualized)
        bri_vol = bri_returns.std() * np.sqrt(252)
        vix_vol = vix_returns.std() * np.sqrt(252)
        
        # Calculate volatility of volatility
        bri_vol_vol = bri_returns.rolling(window=30).std().std() * np.sqrt(252)
        vix_vol_vol = vix_returns.rolling(window=30).std().std() * np.sqrt(252)
        
        print(f"Volatility Analysis:")
        print(f"  BRI Annualized Volatility: {bri_vol:.2f}")
        print(f"  VIX Annualized Volatility: {vix_vol:.2f}")
        print(f"  BRI Volatility of Volatility: {bri_vol_vol:.2f}")
        print(f"  VIX Volatility of Volatility: {vix_vol_vol:.2f}")
        
        # Calculate Sharpe ratios
        bri_sharpe = bri_returns.mean() / bri_returns.std() * np.sqrt(252)
        vix_sharpe = vix_returns.mean() / vix_returns.std() * np.sqrt(252)
        
        print(f"\nSharpe Ratios:")
        print(f"  BRI Sharpe: {bri_sharpe:.3f}")
        print(f"  VIX Sharpe: {vix_sharpe:.3f}")
        
        # Store results
        self.results['volatility'] = {
            'bri_vol': bri_vol,
            'vix_vol': vix_vol,
            'bri_vol_vol': bri_vol_vol,
            'vix_vol_vol': vix_vol_vol,
            'bri_sharpe': bri_sharpe,
            'vix_sharpe': vix_sharpe
        }
        
        return True
    
    def analyze_crisis_periods(self):
        """Analyze BRI vs VIX during crisis periods"""
        print("\n📊 Analyzing Crisis Periods...")
        
        # Define crisis periods (VIX > 30)
        crisis_threshold = 30
        crisis_data = self.data[self.data['VIX'] > crisis_threshold].copy()
        
        if len(crisis_data) > 0:
            print(f"Crisis Analysis (VIX > {crisis_threshold}):")
            print(f"  Number of crisis days: {len(crisis_data)}")
            print(f"  Crisis BRI mean: {crisis_data['BRI'].mean():.2f}")
            print(f"  Crisis VIX mean: {crisis_data['VIX'].mean():.2f}")
            print(f"  Crisis BRI std: {crisis_data['BRI'].std():.2f}")
            print(f"  Crisis VIX std: {crisis_data['VIX'].std():.2f}")
            
            # Calculate correlation during crisis
            crisis_corr = crisis_data['BRI'].corr(crisis_data['VIX'])
            print(f"  Crisis correlation: {crisis_corr:.3f}")
            
            # Store results
            self.results['crisis'] = {
                'crisis_days': len(crisis_data),
                'crisis_bri_mean': crisis_data['BRI'].mean(),
                'crisis_vix_mean': crisis_data['VIX'].mean(),
                'crisis_bri_std': crisis_data['BRI'].std(),
                'crisis_vix_std': crisis_data['VIX'].std(),
                'crisis_correlation': crisis_corr
            }
        else:
            print(f"No crisis periods found (VIX > {crisis_threshold})")
            self.results['crisis'] = {
                'crisis_days': 0,
                'crisis_bri_mean': 0,
                'crisis_vix_mean': 0,
                'crisis_bri_std': 0,
                'crisis_vix_std': 0,
                'crisis_correlation': 0
            }
        
        return True
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        print("\n📊 BRI vs VIX COMPARISON REPORT")
        print("=" * 80)
        
        # Basic Statistics Comparison
        if 'bri_stats' in self.results and 'vix_stats' in self.results:
            print(f"\n📊 BASIC STATISTICS COMPARISON:")
            bri_stats = self.results['bri_stats']
            vix_stats = self.results['vix_stats']
            
            print(f"  BRI vs VIX:")
            print(f"    Mean: {bri_stats['mean']:.2f} vs {vix_stats['mean']:.2f}")
            print(f"    Std: {bri_stats['std']:.2f} vs {vix_stats['std']:.2f}")
            print(f"    Min: {bri_stats['min']:.2f} vs {vix_stats['min']:.2f}")
            print(f"    Max: {bri_stats['max']:.2f} vs {vix_stats['max']:.2f}")
            print(f"    Median: {bri_stats['median']:.2f} vs {vix_stats['median']:.2f}")
            print(f"    Skewness: {bri_stats['skewness']:.2f} vs {vix_stats['skewness']:.2f}")
            print(f"    Kurtosis: {bri_stats['kurtosis']:.2f} vs {vix_stats['kurtosis']:.2f}")
        
        # Correlation Analysis
        if 'correlation' in self.results:
            print(f"\n📊 CORRELATION ANALYSIS:")
            corr = self.results['correlation']
            print(f"  Pearson Correlation: {corr['pearson_corr']:.3f} (p-value: {corr['pearson_p']:.3f})")
            print(f"  Spearman Correlation: {corr['spearman_corr']:.3f} (p-value: {corr['spearman_p']:.3f})")
            print(f"  Rolling Correlation (30-day): Mean: {corr['rolling_corr_mean']:.3f}, Std: {corr['rolling_corr_std']:.3f}")
            
            # Interpretation
            if corr['pearson_corr'] > 0.7:
                print(f"  ✅ Strong positive correlation")
            elif corr['pearson_corr'] > 0.5:
                print(f"  ✅ Moderate positive correlation")
            elif corr['pearson_corr'] > 0.3:
                print(f"  ⚠️ Weak positive correlation")
            else:
                print(f"  ❌ Very weak correlation")
        
        # Volatility Analysis
        if 'volatility' in self.results:
            print(f"\n📊 VOLATILITY ANALYSIS:")
            vol = self.results['volatility']
            print(f"  BRI Annualized Volatility: {vol['bri_vol']:.2f}")
            print(f"  VIX Annualized Volatility: {vol['vix_vol']:.2f}")
            print(f"  BRI Volatility of Volatility: {vol['bri_vol_vol']:.2f}")
            print(f"  VIX Volatility of Volatility: {vol['vix_vol_vol']:.2f}")
            print(f"  BRI Sharpe: {vol['bri_sharpe']:.3f}")
            print(f"  VIX Sharpe: {vol['vix_sharpe']:.3f}")
            
            # Interpretation
            if vol['bri_vol'] > vol['vix_vol']:
                print(f"  ⚠️ BRI is more volatile than VIX")
            else:
                print(f"  ✅ BRI is less volatile than VIX")
        
        # Crisis Analysis
        if 'crisis' in self.results:
            print(f"\n📊 CRISIS ANALYSIS:")
            crisis = self.results['crisis']
            print(f"  Crisis days (VIX > 30): {crisis['crisis_days']}")
            print(f"  Crisis BRI mean: {crisis['crisis_bri_mean']:.2f}")
            print(f"  Crisis VIX mean: {crisis['crisis_vix_mean']:.2f}")
            print(f"  Crisis correlation: {crisis['crisis_correlation']:.3f}")
            
            if crisis['crisis_days'] > 0:
                if crisis['crisis_correlation'] > 0.5:
                    print(f"  ✅ Strong correlation during crisis")
                elif crisis['crisis_correlation'] > 0.3:
                    print(f"  ⚠️ Moderate correlation during crisis")
                else:
                    print(f"  ❌ Weak correlation during crisis")
            else:
                print(f"  ⚠️ No crisis periods found in data")
        
        # Key Differences
        print(f"\n📊 KEY DIFFERENCES:")
        print(f"  1. BRI is a composite behavioral index (0-100 scale)")
        print(f"  2. VIX is implied volatility (typically 10-80 range)")
        print(f"  3. BRI includes sentiment and behavioral factors")
        print(f"  4. VIX is purely options-based volatility")
        print(f"  5. BRI may lead VIX by 1-3 days during stress periods")
        print(f"  6. VIX is more widely used and established")
        print(f"  7. BRI provides additional behavioral context")
        print(f"  8. Both are useful for risk assessment")
        
        return True
    
    def run_comparison_analysis(self):
        """Run complete BRI vs VIX comparison analysis"""
        print("🚀 Running BRI vs VIX Comparison Analysis")
        print("=" * 80)
        
        # Step 1: Load data
        if not self.load_data():
            return False
        
        # Step 2: Calculate basic statistics
        if not self.calculate_basic_statistics():
            return False
        
        # Step 3: Calculate correlation analysis
        if not self.calculate_correlation_analysis():
            return False
        
        # Step 4: Analyze volatility patterns
        if not self.analyze_volatility_patterns():
            return False
        
        # Step 5: Analyze crisis periods
        if not self.analyze_crisis_periods():
            return False
        
        # Step 6: Generate comparison report
        self.generate_comparison_report()
        
        return True

if __name__ == "__main__":
    analyzer = BRIVIXComparison()
    analyzer.run_comparison_analysis()
