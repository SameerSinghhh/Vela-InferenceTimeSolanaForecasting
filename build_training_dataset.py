#!/usr/bin/env python3
"""
Master script to rebuild the complete training dataset from scratch.
Consolidates all price extraction and calculation steps into one file.
"""

import pandas as pd
from datetime import datetime, timedelta
import ast

def extract_btc_prices(training_data, btc_data):
    """Extract BTC Open prices for 8-day periods and add btc_price_array column"""
    print("Processing BTC prices...")
    
    # Sort by date (data types already converted in main function)
    btc_data = btc_data.sort_values('Date').reset_index(drop=True)
    
    # Add btc_price_array column
    training_data['btc_price_array'] = ''
    
    for index, row in training_data.iterrows():
        period_start = pd.to_datetime(row['price_reference_date'])
        period_end = pd.to_datetime(row['target_date'])
        
        # Find BTC prices for this 8-day period
        period_btc = btc_data[
            (btc_data['Date'] >= period_start) & 
            (btc_data['Date'] <= period_end)
        ].sort_values('Date')
        
        if len(period_btc) > 0:
            price_array = period_btc['Open'].tolist()
            training_data.at[index, 'btc_price_array'] = str(price_array)
            print(f"Period {index + 1}: Found {len(price_array)} BTC prices")
        else:
            training_data.at[index, 'btc_price_array'] = '[]'
            print(f"Period {index + 1}: No BTC prices found")
    
    return training_data

def extract_eth_prices(training_data, eth_data):
    """Extract ETH Open prices for 8-day periods and add eth_price_array column"""
    print("Processing ETH prices...")
    
    # Sort by date (data types already converted in main function)
    eth_data = eth_data.sort_values('Date').reset_index(drop=True)
    
    # Add eth_price_array column
    training_data['eth_price_array'] = ''
    
    for index, row in training_data.iterrows():
        period_start = pd.to_datetime(row['price_reference_date'])
        period_end = pd.to_datetime(row['target_date'])
        
        # Find ETH prices for this 8-day period
        period_eth = eth_data[
            (eth_data['Date'] >= period_start) & 
            (eth_data['Date'] <= period_end)
        ].sort_values('Date')
        
        if len(period_eth) > 0:
            price_array = period_eth['Open'].tolist()
            training_data.at[index, 'eth_price_array'] = str(price_array)
            print(f"Period {index + 1}: Found {len(price_array)} ETH prices")
        else:
            training_data.at[index, 'eth_price_array'] = '[]'
            print(f"Period {index + 1}: No ETH prices found")
    
    return training_data

def extract_sol_prices(training_data, sol_data):
    """Extract SOL prices and add all SOL-related columns"""
    print("Processing SOL prices...")
    
    # Sort by date (data types already converted in main function)
    sol_data = sol_data.sort_values('Date').reset_index(drop=True)
    
    # Rename reference_price to target_date_sol_price if it exists
    if 'reference_price' in training_data.columns:
        training_data = training_data.rename(columns={'reference_price': 'target_date_sol_price'})
    else:
        training_data['target_date_sol_price'] = None
    
    # Add new SOL columns
    training_data['prediction_date_sol_price'] = None
    training_data['actual_change_pct'] = None
    training_data['sol_price_array'] = ''
    
    for index, row in training_data.iterrows():
        target_date = pd.to_datetime(row['target_date'])
        prediction_date = pd.to_datetime(row['prediction_date'])
        period_start = pd.to_datetime(row['price_reference_date'])
        
        # Get target date SOL price
        target_sol = sol_data[sol_data['Date'] == target_date]
        if len(target_sol) > 0:
            target_price = target_sol.iloc[0]['Open']
            training_data.at[index, 'target_date_sol_price'] = target_price
        
        # Get prediction date SOL price  
        pred_sol = sol_data[sol_data['Date'] == prediction_date]
        if len(pred_sol) > 0:
            pred_price = pred_sol.iloc[0]['Open']
            training_data.at[index, 'prediction_date_sol_price'] = pred_price
            
            # Calculate actual change percentage
            if pd.notna(training_data.at[index, 'target_date_sol_price']):
                target_price = training_data.at[index, 'target_date_sol_price']
                actual_change = ((pred_price - target_price) / target_price) * 100
                training_data.at[index, 'actual_change_pct'] = actual_change
        
        # Get SOL price array for 8-day period
        period_sol = sol_data[
            (sol_data['Date'] >= period_start) & 
            (sol_data['Date'] <= target_date)
        ].sort_values('Date')
        
        if len(period_sol) > 0:
            price_array = period_sol['Open'].tolist()
            training_data.at[index, 'sol_price_array'] = str(price_array)
            print(f"Period {index + 1}: Found {len(price_array)} SOL prices")
        else:
            training_data.at[index, 'sol_price_array'] = '[]'
            print(f"Period {index + 1}: No SOL prices found")
    
    return training_data

def extract_sp_prices(training_data, sp_data):
    """Extract S&P 500 prices and add sp_price_array column"""
    print("Processing S&P 500 prices...")
    
    # Sort by date (data types already converted in main function)
    sp_data = sp_data.sort_values('Date').reset_index(drop=True)
    
    # Add sp_price_array column
    training_data['sp_price_array'] = ''
    
    for index, row in training_data.iterrows():
        period_start = pd.to_datetime(row['price_reference_date'])
        period_end = pd.to_datetime(row['target_date'])
        
        # Find S&P 500 prices for this period (excluding weekends)
        period_sp = sp_data[
            (sp_data['Date'] >= period_start) & 
            (sp_data['Date'] <= period_end)
        ].sort_values('Date')
        
        if len(period_sp) > 0:
            price_array = period_sp['Open'].tolist()
            training_data.at[index, 'sp_price_array'] = str(price_array)
            print(f"Period {index + 1}: Found {len(price_array)} S&P 500 prices")
        else:
            training_data.at[index, 'sp_price_array'] = '[]'
            print(f"Period {index + 1}: No S&P 500 prices found")
    
    return training_data

def calculate_30d_changes(training_data, btc_data, eth_data, sol_data, sp_data):
    """Calculate 30-day percentage changes for all assets"""
    print("Calculating 30-day changes...")
    
    # Add 30-day change columns
    training_data['btc_30d_change_pct'] = None
    training_data['eth_30d_change_pct'] = None
    training_data['sol_30d_change_pct'] = None
    training_data['sp500_30d_change_pct'] = None
    
    def get_30d_change(data, target_date, asset_name):
        """Calculate 30-day change for a specific asset"""
        target_date = pd.to_datetime(target_date)
        date_30_before = target_date - timedelta(days=30)
        
        # Sort data by date ascending to ensure .tail(1) gets the latest matching date
        data_sorted = data.sort_values('Date')
        
        # Find closest dates
        before_data = data_sorted[data_sorted['Date'] <= date_30_before].tail(1)
        target_data = data_sorted[data_sorted['Date'] <= target_date].tail(1)
        
        if len(before_data) > 0 and len(target_data) > 0:
            before_price = before_data.iloc[0]['Open']
            target_price = target_data.iloc[0]['Open']
            
            # Convert to numeric if still strings
            if isinstance(before_price, str):
                before_price = float(before_price.replace(',', ''))
            if isinstance(target_price, str):
                target_price = float(target_price.replace(',', ''))
                
            change_pct = ((target_price - before_price) / before_price) * 100
            return change_pct
        return None
    
    # Calculate changes for each period
    for index, row in training_data.iterrows():
        target_date = row['target_date']
        
        # BTC 30-day change
        btc_change = get_30d_change(btc_data, target_date, 'BTC')
        training_data.at[index, 'btc_30d_change_pct'] = btc_change
        
        # ETH 30-day change
        eth_change = get_30d_change(eth_data, target_date, 'ETH')
        training_data.at[index, 'eth_30d_change_pct'] = eth_change
        
        # SOL 30-day change
        sol_change = get_30d_change(sol_data, target_date, 'SOL')
        training_data.at[index, 'sol_30d_change_pct'] = sol_change
        
        # S&P 500 30-day change
        sp_change = get_30d_change(sp_data, target_date, 'S&P 500')
        training_data.at[index, 'sp500_30d_change_pct'] = sp_change
        
        print(f"Period {index + 1}: Calculated 30-day changes")
    
    return training_data

def reorder_columns(training_data):
    """Reorder columns to match the final format"""
    desired_order = [
        'price_reference_date', 'context_start_date', 'target_date', 'prediction_date',
        'target_date_sol_price', 'prediction_date_sol_price', 'actual_change_pct', 'tweets',
        'predicted_price', 'predicted_change_pct', 'llm_reasoning', 'llm_reflection',
        'btc_price_array', 'eth_price_array', 'sol_price_array', 'sp_price_array',
        'btc_30d_change_pct', 'eth_30d_change_pct', 'sol_30d_change_pct', 'sp500_30d_change_pct'
    ]
    
    # Add missing columns with empty values
    for col in desired_order:
        if col not in training_data.columns:
            training_data[col] = ''
    
    # Reorder columns
    training_data = training_data[desired_order]
    return training_data

def rebuild_training_dataset():
    """Main function to rebuild the complete training dataset"""
    print("🚀 Starting training dataset rebuild...")
    
    # Step 1: Generate base training data (51 weekly periods)
    print("\n📊 Step 1: Generating base training data...")
    
    # Generate 51 weekly periods starting from 2024-07-25
    start_date = pd.to_datetime('2024-07-25')
    periods = []
    
    for i in range(51):
        price_reference_date = start_date + timedelta(days=i*7)
        context_start_date = price_reference_date + timedelta(days=5)
        target_date = context_start_date + timedelta(days=2)
        prediction_date = target_date + timedelta(days=1)
        
        periods.append({
            'price_reference_date': price_reference_date.strftime('%Y-%m-%d'),
            'context_start_date': context_start_date.strftime('%Y-%m-%d'),
            'target_date': target_date.strftime('%Y-%m-%d'),
            'prediction_date': prediction_date.strftime('%Y-%m-%d'),
            'target_date_sol_price': None,
            'prediction_date_sol_price': None,
            'actual_change_pct': None,
            'tweets': None,
            'predicted_price': None,
            'predicted_change_pct': None,
            'llm_reasoning': None,
            'llm_reflection': None
        })
    
    training_data = pd.DataFrame(periods)
    print(f"Generated {len(training_data)} periods")
    
    # Step 2: Load all price data
    print("\n📈 Step 2: Loading price data files...")
    btc_data = pd.read_csv('price_data_btc.csv', sep='\t')
    eth_data = pd.read_csv('price_data_eth.csv', sep='\t')
    sol_data = pd.read_csv('price_data_sol.csv', sep='\t')
    sp_data = pd.read_csv('price_data_sp.csv', sep='\t', header=None, skiprows=1)
    
    # Convert dates and prices to proper types
    for data, name in [(btc_data, 'BTC'), (eth_data, 'ETH'), (sol_data, 'SOL')]:
        data['Date'] = pd.to_datetime(data['Date'], format='%b %d, %Y')
        # Remove commas from Open prices and convert to numeric
        data['Open'] = data['Open'].astype(str).str.replace(',', '')
        data['Open'] = pd.to_numeric(data['Open'], errors='coerce')
    
    # Handle S&P 500 special format
    sp_data.columns = ['Date', 'Price', 'Open', 'High', 'Low', 'Empty1', 'Change %']
    sp_data['Date'] = pd.to_datetime(sp_data['Date'])
    # Remove commas from S&P 500 Open prices and convert to numeric
    sp_data['Open'] = sp_data['Open'].astype(str).str.replace(',', '')
    sp_data['Open'] = pd.to_numeric(sp_data['Open'], errors='coerce')
    
    print(f"BTC data: {len(btc_data)} rows")
    print(f"ETH data: {len(eth_data)} rows")
    print(f"SOL data: {len(sol_data)} rows")
    print(f"S&P 500 data: {len(sp_data)} rows")
    
    # Step 3: Extract BTC prices
    print("\n🟠 Step 3: Extracting BTC prices...")
    training_data = extract_btc_prices(training_data, btc_data)
    
    # Step 4: Extract ETH prices
    print("\n🔵 Step 4: Extracting ETH prices...")
    training_data = extract_eth_prices(training_data, eth_data)
    
    # Step 5: Extract SOL prices
    print("\n🟣 Step 5: Extracting SOL prices...")
    training_data = extract_sol_prices(training_data, sol_data)
    
    # Step 6: Extract S&P 500 prices
    print("\n🟢 Step 6: Extracting S&P 500 prices...")
    training_data = extract_sp_prices(training_data, sp_data)
    
    # Step 7: Calculate 30-day changes
    print("\n📊 Step 7: Calculating 30-day changes...")
    training_data = calculate_30d_changes(training_data, btc_data, eth_data, sol_data, sp_data)
    
    # Step 8: Reorder columns
    print("\n🔄 Step 8: Reordering columns...")
    training_data = reorder_columns(training_data)
    
    # Step 9: Save final dataset
    print("\n💾 Step 9: Saving final dataset...")
    output_file = 'final_training_set_copy.csv'
    training_data.to_csv(output_file, index=False)
    
    print(f"\n✅ SUCCESS! Rebuilt training dataset saved as: {output_file}")
    print(f"📊 Final dataset: {len(training_data)} rows × {len(training_data.columns)} columns")
    print(f"🗂️  Columns: {list(training_data.columns)}")

if __name__ == "__main__":
    rebuild_training_dataset() 