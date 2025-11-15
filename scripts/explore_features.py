"""
Interactive Market Profile Features Explorer

This script provides interactive visualizations and explanations of all Market Profile features.
Run with: python scripts/explore_features.py
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from datetime import datetime

from src.data.loader import DataLoader
from src.features.market_profile import MarketProfileEngine, TechnicalFeatures

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def print_feature_explanation(feature_name, description, value=None, interpretation=None):
    """Print feature explanation in a formatted way."""
    print("\n" + "="*80)
    print(f"📊 {feature_name.upper()}")
    print("="*80)
    print(f"Description: {description}")
    if value is not None:
        print(f"Current Value: {value:,.2f}")
    if interpretation:
        print(f"💡 Interpretation: {interpretation}")
    print("="*80)

def visualize_market_profile_session(data, date, mp_engine):
    """Visualize a single Market Profile session with all features."""
    daily_data = data[data.index.date == date]
    if daily_data.empty:
        print(f"No data for {date}")
        return None
    
    profile = mp_engine.compute_daily_profile(daily_data)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Market Profile Analysis - {date}\n', fontsize=16, fontweight='bold')
    
    # 1. Price Chart with POC, VAH, VAL
    ax1 = axes[0, 0]
    ax1.plot(daily_data.index, daily_data['close'], 'b-', linewidth=2, label='Close Price')
    ax1.fill_between(daily_data.index, daily_data['low'], daily_data['high'], 
                     alpha=0.3, color='gray', label='Price Range')
    
    # Mark POC, VAH, VAL
    ax1.axhline(y=profile['poc'], color='red', linestyle='--', linewidth=2, label=f"POC: ${profile['poc']:,.2f}")
    ax1.axhline(y=profile['vah'], color='green', linestyle='--', linewidth=2, label=f"VAH: ${profile['vah']:,.2f}")
    ax1.axhline(y=profile['val'], color='orange', linestyle='--', linewidth=2, label=f"VAL: ${profile['val']:,.2f}")
    
    # Highlight Value Area
    ax1.axhspan(profile['val'], profile['vah'], alpha=0.2, color='yellow', label='Value Area (70% volume)')
    
    ax1.set_title('Price Chart with Market Profile Levels', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price ($)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. Volume Distribution (Market Profile Histogram)
    ax2 = axes[0, 1]
    price_bins = mp_engine._create_price_bins(daily_data)
    volume_dist = mp_engine._distribute_volume(daily_data, price_bins)
    
    # Create horizontal bar chart
    bin_centers = (price_bins[:-1] + price_bins[1:]) / 2
    colors = ['red' if abs(bc - profile['poc']) < 10 else 'blue' for bc in bin_centers]
    ax2.barh(bin_centers, volume_dist, color=colors, alpha=0.6)
    ax2.axvline(x=profile['poc_volume'], color='red', linestyle='--', linewidth=2, label='POC Volume')
    ax2.set_title('Volume Distribution by Price (Market Profile)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Volume')
    ax2.set_ylabel('Price ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Balance Flag Visualization
    ax3 = axes[1, 0]
    day_range = profile['day_high'] - profile['day_low']
    poc_position = (profile['poc'] - profile['day_low']) / day_range
    
    # Draw price range
    ax3.barh([0], [day_range], left=profile['day_low'], height=0.3, color='lightblue', alpha=0.5, label='Day Range')
    
    # Mark quarters
    q1 = profile['day_low'] + day_range * 0.25
    q3 = profile['day_low'] + day_range * 0.75
    ax3.axvline(x=q1, color='gray', linestyle=':', linewidth=1)
    ax3.axvline(x=q3, color='gray', linestyle=':', linewidth=1)
    
    # Mark POC
    ax3.scatter([profile['poc']], [0], s=200, color='red', marker='o', zorder=5, label=f"POC: ${profile['poc']:,.2f}")
    
    # Highlight balanced zone
    ax3.axvspan(q1, q3, alpha=0.2, color='green', label='Balanced Zone (25%-75%)')
    
    balance_text = "BALANCED" if profile['balance_flag'] == 1 else "UNBALANCED"
    ax3.set_title(f'Balance Flag: {balance_text} (POC at {poc_position*100:.1f}% of range)', 
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel('Price ($)')
    ax3.set_xticks([profile['day_low'], q1, profile['poc'], q3, profile['day_high']])
    ax3.set_xticklabels([f'${profile["day_low"]:.0f}', '25%', f'POC\n${profile["poc"]:.0f}', 
                         '75%', f'${profile["day_high"]:.0f}'], rotation=45, ha='right')
    ax3.set_yticks([])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Volume Imbalance
    ax4 = axes[1, 1]
    imbalance = profile['volume_imbalance']
    colors_pie = ['green' if imbalance > 0.5 else 'red', 'gray']
    ax4.pie([imbalance, 1-imbalance], labels=[f'Upside\n{imbalance*100:.1f}%', 
                                              f'Downside\n{(1-imbalance)*100:.1f}%'],
            colors=colors_pie, autopct='%1.1f%%', startangle=90)
    ax4.set_title(f'Volume Imbalance: {imbalance:.3f}\n(Upside/Downside Ratio)', 
                  fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig, profile

def explain_all_features(profile, tech_features=None):
    """Print detailed explanations of all features."""
    
    print("\n" + "🔍"*40)
    print("MARKET PROFILE FEATURES EXPLANATION")
    print("🔍"*40)
    
    # 1. Session POC
    print_feature_explanation(
        "session_poc (Point of Control)",
        "The price level where the most trading volume occurred during the session. This is the 'fair value' price where buyers and sellers agreed most.",
        profile['poc'],
        f"Most trading happened at ${profile['poc']:,.2f}. This is the price level with highest consensus."
    )
    
    # 2. Session VAH
    print_feature_explanation(
        "session_vah (Value Area High)",
        "The upper boundary of the Value Area, containing 70% of session volume. Prices above VAH are considered 'expensive'.",
        profile['vah'],
        f"${profile['vah']:,.2f} is the upper limit of fair value. Breakouts above this level indicate strong buying pressure."
    )
    
    # 3. Session VAL
    print_feature_explanation(
        "session_val (Value Area Low)",
        "The lower boundary of the Value Area, containing 70% of session volume. Prices below VAL are considered 'cheap'.",
        profile['val'],
        f"${profile['val']:,.2f} is the lower limit of fair value. Prices below this indicate selling pressure."
    )
    
    # 4. VA Range Width
    va_width = profile['va_range_width']
    print_feature_explanation(
        "va_range_width (Value Area Range Width)",
        "The difference between VAH and VAL. Measures the width of the price range where 70% of volume traded.",
        va_width,
        f"A width of ${va_width:,.2f} indicates {'high volatility' if va_width > 500 else 'low volatility'}. Wider ranges suggest more uncertainty."
    )
    
    # 5. Balance Flag
    balance_text = "BALANCED" if profile['balance_flag'] == 1 else "UNBALANCED"
    print_feature_explanation(
        "balance_flag",
        "Indicates if the session was balanced (POC in middle 50% of price range) or unbalanced (POC near extremes).",
        profile['balance_flag'],
        f"Session is {balance_text}. Balanced sessions suggest consolidation, unbalanced suggest directional moves."
    )
    
    # 6. Volume Imbalance
    imbalance = profile['volume_imbalance']
    print_feature_explanation(
        "volume_imbalance",
        "Ratio of upside volume (volume when price moved up) to total volume. Measures buying vs selling pressure.",
        imbalance,
        f"Value of {imbalance:.3f} means {imbalance*100:.1f}% of volume was on the upside. {'Bullish' if imbalance > 0.5 else 'Bearish'} pressure."
    )
    
    # 7. Session Volume
    print_feature_explanation(
        "session_volume",
        "Total trading volume for the session. Higher volume indicates more participation and conviction.",
        profile['session_volume'],
        f"Total volume: {profile['session_volume']:,.0f}. {'High' if profile['session_volume'] > 1e6 else 'Low'} participation level."
    )
    
    # Technical Indicators
    if tech_features:
        # 8. ATR
        atr = tech_features.get('atr_14', None)
        if atr is not None:
            print_feature_explanation(
                "atr_14 (Average True Range)",
                "14-period Average True Range. Measures volatility by averaging the true range (high-low, accounting for gaps).",
                atr,
                f"ATR of ${atr:,.2f} indicates {'high' if atr > 500 else 'moderate' if atr > 200 else 'low'} volatility. Higher ATR = more price movement expected."
            )
        
        # 9. RSI
        rsi = tech_features.get('rsi_14', None)
        if rsi is not None:
            rsi_status = "OVERSOLD (<30)" if rsi < 30 else "OVERBOUGHT (>70)" if rsi > 70 else "NEUTRAL"
            print_feature_explanation(
                "rsi_14 (Relative Strength Index)",
                "14-period RSI. Momentum oscillator measuring speed and magnitude of price changes. Range: 0-100.",
                rsi,
                f"RSI of {rsi:.1f} indicates {rsi_status}. {'Buying opportunity' if rsi < 30 else 'Potential reversal' if rsi > 70 else 'Normal momentum'}."
            )
        
        # 10. Returns
        one_day = tech_features.get('one_day_return', None)
        three_day = tech_features.get('three_day_return', None)
        if one_day is not None:
            print_feature_explanation(
                "one_day_return",
                "1-day return: (Today's close - Yesterday's close) / Yesterday's close. Measures short-term trend.",
                one_day,
                f"Return of {one_day*100:.2f}% indicates {'positive' if one_day > 0 else 'negative'} short-term momentum."
            )
        if three_day is not None:
            print_feature_explanation(
                "three_day_return",
                "3-day return: (Today's close - 3 days ago close) / 3 days ago close. Measures medium-term trend.",
                three_day,
                f"Return of {three_day*100:.2f}% indicates {'positive' if three_day > 0 else 'negative'} medium-term momentum."
            )

def main():
    """Main function to run interactive feature exploration."""
    print("\n" + "="*80)
    print("🎯 MARKET PROFILE FEATURES INTERACTIVE EXPLORER")
    print("="*80)
    print("\nThis tool will help you understand all Market Profile features with real data.")
    print("Loading Bitcoin data...\n")
    
    # Load data
    loader = DataLoader(ticker="BTC-USD", period="2y", interval="1h")
    raw_data = loader.fetch_data(use_cache=True, cache_dir="data/raw")
    raw_data = loader.clean_data(raw_data)
    
    print(f"✅ Loaded {len(raw_data)} hourly candles")
    print(f"📅 Date range: {raw_data.index.min().date()} to {raw_data.index.max().date()}\n")
    
    # Initialize engines
    mp_engine = MarketProfileEngine(tpo_size=30, vol_percentile=70)
    tech_engine = TechnicalFeatures()
    
    # Get a recent date for visualization
    available_dates = sorted(set(raw_data.index.date))
    recent_date = available_dates[-10]  # Use a date from 10 days ago
    
    print(f"📊 Analyzing session from: {recent_date}\n")
    
    # Visualize Market Profile
    fig, profile = visualize_market_profile_session(raw_data, recent_date, mp_engine)
    
    # Calculate technical indicators for this date
    daily_data = raw_data[raw_data.index.date == recent_date]
    if not daily_data.empty:
        # Get ATR and RSI (need full data for calculation)
        atr = tech_engine.calculate_atr(raw_data['high'], raw_data['low'], raw_data['close'], period=14)
        rsi = tech_engine.calculate_rsi(raw_data['close'], period=14)
        returns = tech_engine.calculate_returns(raw_data['close'], periods=[1, 3])
        
        # Get values for the specific date
        date_timestamp = daily_data.index[-1]
        tech_features = {
            'atr_14': atr.loc[date_timestamp] if date_timestamp in atr.index else None,
            'rsi_14': rsi.loc[date_timestamp] if date_timestamp in rsi.index else None,
            'one_day_return': returns['return_1d'].loc[date_timestamp] if date_timestamp in returns['return_1d'].index else None,
            'three_day_return': returns['return_3d'].loc[date_timestamp] if date_timestamp in returns['return_3d'].index else None,
        }
    else:
        tech_features = {}
    
    # Print all explanations
    explain_all_features(profile, tech_features)
    
    # Show plot
    print("\n" + "="*80)
    print("📈 Displaying interactive visualization...")
    print("="*80)
    print("\nClose the plot window to continue.\n")
    plt.show()
    
    # Summary
    print("\n" + "="*80)
    print("📋 FEATURE SUMMARY")
    print("="*80)
    print(f"""
    Session Date: {recent_date}
    POC: ${profile['poc']:,.2f} (Point of Control)
    VAH: ${profile['vah']:,.2f} (Value Area High)
    VAL: ${profile['val']:,.2f} (Value Area Low)
    VA Width: ${profile['va_range_width']:,.2f}
    Balance: {'✅ Balanced' if profile['balance_flag'] == 1 else '❌ Unbalanced'}
    Volume Imbalance: {profile['volume_imbalance']:.3f} ({'Bullish' if profile['volume_imbalance'] > 0.5 else 'Bearish'})
    Session Volume: {profile['session_volume']:,.0f}
    """)
    
    if tech_features.get('atr_14'):
        print(f"    ATR (14): ${tech_features['atr_14']:,.2f}")
    if tech_features.get('rsi_14'):
        print(f"    RSI (14): {tech_features['rsi_14']:.1f}")
    if tech_features.get('one_day_return'):
        print(f"    1-Day Return: {tech_features['one_day_return']*100:.2f}%")
    if tech_features.get('three_day_return'):
        print(f"    3-Day Return: {tech_features['three_day_return']*100:.2f}%")
    
    print("\n" + "="*80)
    print("✅ Feature exploration complete!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

