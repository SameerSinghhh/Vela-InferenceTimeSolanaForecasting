#!/usr/bin/env python3
"""
Script to extract S&P 500 Open prices and add sp_price_array column
"""

import pandas as pd
from datetime import datetime, timedelta

def extract_sp_prices():
    """Extract S&P 500 Open prices and add to existing file"""
    
    # Read the S&P 500 price data (mixed delimiter format)
    print("Reading S&P 500 price data...")
    # The header is space-separated, but data rows are tab-separated
    sp_data = pd.read_csv('price_data_sp.csv', sep='\t', header=None, skiprows=1)
    
    # Manually assign column names
    sp_data.columns = ['Date', 'Price', 'Open', 'High', 'Low', 'Empty1', 'Change %']
    
    # Convert Date column to datetime (handling the "Jul 15, 2025" format)
    sp_data['Date'] = pd.to_datetime(sp_data['Date'])
    
    # Sort by date (ascending) for easier processing
    sp_data = sp_data.sort_values('Date').reset_index(drop=True)
    
    # Read the most recent training set file
    print("Reading existing training set...")
    try:
        training_data = pd.read_csv('training_set2_with_btc_eth_sol.csv')
        print("Found BTC+ETH+SOL version")
    except FileNotFoundError:
        try:
            training_data = pd.read_csv('training_set2_with_btc_eth.csv')
            print("Found BTC+ETH version")
        except FileNotFoundError:
            try:
                training_data = pd.read_csv('training_set2_with_btc.csv')
                print("Found BTC-only version")
            except FileNotFoundError:
                training_data = pd.read_csv('training_set2.csv')
                print("Found base version")
    
    # Convert date columns to datetime
    training_data['price_reference_date'] = pd.to_datetime(training_data['price_reference_date'])
    training_data['target_date'] = pd.to_datetime(training_data['target_date'])
    
    # Create list to store the S&P 500 data
    sp_price_arrays = []
    
    print("Extracting S&P 500 prices for each period...")
    
    for idx, row in training_data.iterrows():
        start_date = row['price_reference_date']
        end_date = row['target_date']
        
        print(f"Processing {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Get prices for the date range (inclusive)
        mask = (sp_data['Date'] >= start_date) & (sp_data['Date'] <= end_date)
        period_data = sp_data.loc[mask, ['Date', 'Open']].copy()
        
        if len(period_data) == 0:
            print(f"  WARNING: No S&P 500 data found for period {start_date} to {end_date}")
            sp_price_arrays.append("[]")
            continue
            
        # Sort by date to ensure correct order
        period_data = period_data.sort_values('Date')
        
        # Extract the Open prices and format as clean array
        prices = period_data['Open'].tolist()
        
        # Clean the prices (remove commas, convert to float)
        clean_prices = []
        for price in prices:
            if isinstance(price, str):
                # Remove commas and convert to float
                clean_price = float(price.replace(',', ''))
            else:
                clean_price = float(price)
            clean_prices.append(clean_price)
        
        # Format as string array
        price_array_str = str(clean_prices)
        sp_price_arrays.append(price_array_str)
        
        print(f"  Found {len(clean_prices)} S&P 500 prices")
    
    # Add the S&P 500 column to the training data
    training_data['sp_price_array'] = sp_price_arrays
    
    # Determine output filename based on existing columns
    if 'sol_price_array' in training_data.columns:
        output_file = 'training_set2_with_btc_eth_sol_sp.csv'
    elif 'eth_price_array' in training_data.columns:
        output_file = 'training_set2_with_btc_eth_sp.csv'
    elif 'btc_price_array' in training_data.columns:
        output_file = 'training_set2_with_btc_sp.csv'
    else:
        output_file = 'training_set2_with_sp.csv'
    
    training_data.to_csv(output_file, index=False)
    
    print(f"\nCompleted! Updated file saved as: {output_file}")
    print(f"Processed {len(sp_price_arrays)} periods")
    
    # Show sample of results
    print("\nSample results:")
    for i in range(min(3, len(training_data))):
        row = training_data.iloc[i]
        print(f"Period {i+1}: {row['price_reference_date'].strftime('%Y-%m-%d')} to {row['target_date'].strftime('%Y-%m-%d')}")
        print(f"  S&P 500 array: {row['sp_price_array'][:50]}...")
        if 'sol_price_array' in training_data.columns:
            print(f"  SOL array: {row['sol_price_array'][:50]}...")
        if 'btc_price_array' in training_data.columns:
            print(f"  BTC array: {row['btc_price_array'][:50]}...")
        if 'eth_price_array' in training_data.columns:
            print(f"  ETH array: {row['eth_price_array'][:50]}...")
        print()

if __name__ == "__main__":
    extract_sp_prices() 