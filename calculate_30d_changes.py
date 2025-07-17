#!/usr/bin/env python3
"""
Script to calculate 30-day percentage changes for BTC, ETH, SOL, and S&P 500
"""

import pandas as pd
from datetime import datetime, timedelta
import ast

def calculate_30d_changes():
    """Calculate 30-day percentage changes for all assets"""
    
    # Read the training dataset
    print("Reading training dataset...")
    training_data = pd.read_csv('training_set2_with_btc_eth_sol_sp.csv')
    
    # Read all price data files
    print("Reading price data files...")
    btc_data = pd.read_csv('price_data_btc.csv', sep='\t')
    eth_data = pd.read_csv('price_data_eth.csv', sep='\t')
    sol_data = pd.read_csv('price_data_sol.csv', sep='\t')
    sp_data = pd.read_csv('price_data_sp.csv', sep='\t', header=None, skiprows=1)
    
    # Process S&P 500 data (different format)
    sp_data.columns = ['Date', 'Price', 'Open', 'High', 'Low', 'Empty1', 'Change %']
    
    # Convert dates to datetime
    btc_data['Date'] = pd.to_datetime(btc_data['Date'])
    eth_data['Date'] = pd.to_datetime(eth_data['Date'])
    sol_data['Date'] = pd.to_datetime(sol_data['Date'])
    sp_data['Date'] = pd.to_datetime(sp_data['Date'])
    
    # Sort by date
    btc_data = btc_data.sort_values('Date').reset_index(drop=True)
    eth_data = eth_data.sort_values('Date').reset_index(drop=True)
    sol_data = sol_data.sort_values('Date').reset_index(drop=True)
    sp_data = sp_data.sort_values('Date').reset_index(drop=True)
    
    # Convert target_date to datetime
    training_data['target_date'] = pd.to_datetime(training_data['target_date'])
    
    # Lists to store calculated changes
    btc_30d_changes = []
    eth_30d_changes = []
    sol_30d_changes = []
    sp500_30d_changes = []
    
    print("Calculating 30-day changes for each period...")
    
    for idx, row in training_data.iterrows():
        target_date = row['target_date']
        thirty_days_ago = target_date - timedelta(days=30)
        
        print(f"Processing period {idx+1}: {target_date.strftime('%Y-%m-%d')} (30 days ago: {thirty_days_ago.strftime('%Y-%m-%d')})")
        
        # Get target date prices from the price arrays
        try:
            btc_array = ast.literal_eval(row['btc_price_array'])
            eth_array = ast.literal_eval(row['eth_price_array'])
            sol_array = ast.literal_eval(row['sol_price_array'])
            sp_array = ast.literal_eval(row['sp_price_array'])
            
            # Target date prices (last item in each array)
            btc_target_price = btc_array[-1]
            eth_target_price = eth_array[-1]
            sol_target_price = sol_array[-1]
            sp_target_price = sp_array[-1]
            
        except (ValueError, IndexError, SyntaxError):
            print(f"  WARNING: Could not parse price arrays for period {idx+1}")
            btc_30d_changes.append(None)
            eth_30d_changes.append(None)
            sol_30d_changes.append(None)
            sp500_30d_changes.append(None)
            continue
        
        # Calculate BTC 30-day change
        btc_change = calculate_change(btc_data, thirty_days_ago, btc_target_price, 'BTC')
        btc_30d_changes.append(btc_change)
        
        # Calculate ETH 30-day change
        eth_change = calculate_change(eth_data, thirty_days_ago, eth_target_price, 'ETH')
        eth_30d_changes.append(eth_change)
        
        # Calculate SOL 30-day change
        sol_change = calculate_change(sol_data, thirty_days_ago, sol_target_price, 'SOL')
        sol_30d_changes.append(sol_change)
        
        # Calculate S&P 500 30-day change
        sp_change = calculate_change(sp_data, thirty_days_ago, sp_target_price, 'S&P 500')
        sp500_30d_changes.append(sp_change)
        
        # Print results
        print(f"  BTC: {btc_change:.2f}%" if btc_change is not None else "  BTC: N/A")
        print(f"  ETH: {eth_change:.2f}%" if eth_change is not None else "  ETH: N/A")
        print(f"  SOL: {sol_change:.2f}%" if sol_change is not None else "  SOL: N/A")
        print(f"  S&P 500: {sp_change:.2f}%" if sp_change is not None else "  S&P 500: N/A")
        print()
    
    # Update the training data with calculated changes
    training_data['btc_30d_change_pct'] = btc_30d_changes
    training_data['eth_30d_change_pct'] = eth_30d_changes
    training_data['sol_30d_change_pct'] = sol_30d_changes
    training_data['sp500_30d_change_pct'] = sp500_30d_changes
    
    # Save the updated file
    training_data.to_csv('training_set2_with_btc_eth_sol_sp.csv', index=False)
    
    print(f"Completed! Updated training_set2_with_btc_eth_sol_sp.csv with 30-day changes")
    print(f"Processed {len(training_data)} periods")
    
    # Show summary statistics
    valid_btc = [x for x in btc_30d_changes if x is not None]
    valid_eth = [x for x in eth_30d_changes if x is not None]
    valid_sol = [x for x in sol_30d_changes if x is not None]
    valid_sp = [x for x in sp500_30d_changes if x is not None]
    
    print(f"\nSummary:")
    print(f"  Valid BTC calculations: {len(valid_btc)}/{len(btc_30d_changes)}")
    print(f"  Valid ETH calculations: {len(valid_eth)}/{len(eth_30d_changes)}")
    print(f"  Valid SOL calculations: {len(valid_sol)}/{len(sol_30d_changes)}")
    print(f"  Valid S&P 500 calculations: {len(valid_sp)}/{len(sp500_30d_changes)}")


def calculate_change(price_data, thirty_days_ago, target_price, asset_name):
    """Calculate percentage change from 30 days ago to target price"""
    
    # Find the closest date to 30 days ago
    date_diffs = (price_data['Date'] - thirty_days_ago).abs()
    closest_idx = date_diffs.idxmin()
    closest_date = price_data.loc[closest_idx, 'Date']
    
    # Check if we found a date within reasonable range (within 5 days)
    if abs((closest_date - thirty_days_ago).days) > 5:
        print(f"    WARNING: No {asset_name} price data within 5 days of {thirty_days_ago.strftime('%Y-%m-%d')}")
        return None
    
    # Get the price 30 days ago
    thirty_day_price = price_data.loc[closest_idx, 'Open']
    
    # Clean the price (remove commas if string)
    if isinstance(thirty_day_price, str):
        thirty_day_price = float(thirty_day_price.replace(',', ''))
    else:
        thirty_day_price = float(thirty_day_price)
    
    # Calculate percentage change
    if thirty_day_price > 0:
        change_pct = ((target_price - thirty_day_price) / thirty_day_price) * 100
        return change_pct
    else:
        print(f"    WARNING: Invalid {asset_name} price 30 days ago: {thirty_day_price}")
        return None


if __name__ == "__main__":
    calculate_30d_changes() 