#!/usr/bin/env python3
"""
Script to extract ETH Open prices for each date range and add to training_set2_with_btc.csv
"""

import pandas as pd
from datetime import datetime, timedelta

def extract_eth_prices():
    """Extract ETH Open prices for each training period and add to existing file"""
    
    # Read the ETH price data (tab-separated)
    print("Reading ETH price data...")
    eth_data = pd.read_csv('price_data_eth.csv', sep='\t')
    
    # Convert Date column to datetime
    eth_data['Date'] = pd.to_datetime(eth_data['Date'])
    
    # Sort by date (ascending) for easier processing
    eth_data = eth_data.sort_values('Date').reset_index(drop=True)
    
    # Read the existing training set with BTC data
    print("Reading existing training set with BTC data...")
    training_data = pd.read_csv('training_set2_with_btc.csv')
    
    # Convert date columns to datetime
    training_data['price_reference_date'] = pd.to_datetime(training_data['price_reference_date'])
    training_data['target_date'] = pd.to_datetime(training_data['target_date'])
    
    # Create a list to store the ETH price arrays
    eth_price_arrays = []
    
    print("Extracting ETH prices for each period...")
    
    for idx, row in training_data.iterrows():
        start_date = row['price_reference_date']
        end_date = row['target_date']
        
        print(f"Processing {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Get prices for the date range (inclusive)
        mask = (eth_data['Date'] >= start_date) & (eth_data['Date'] <= end_date)
        period_data = eth_data.loc[mask, ['Date', 'Open']].copy()
        
        if len(period_data) == 0:
            print(f"  WARNING: No ETH data found for period {start_date} to {end_date}")
            eth_price_arrays.append("[]")
            continue
            
        # Sort by date to ensure correct order
        period_data = period_data.sort_values('Date')
        
        # Extract the Open prices and format as clean array
        prices = period_data['Open'].tolist()
        
        # Clean the prices (remove commas, convert to float)
        clean_prices = []
        for price in prices:
            if isinstance(price, str):
                clean_price = float(price.replace(',', ''))
            else:
                clean_price = float(price)
            clean_prices.append(clean_price)
        
        # Format as string array
        price_array_str = str(clean_prices)
        eth_price_arrays.append(price_array_str)
        
        print(f"  Found {len(clean_prices)} ETH prices: {price_array_str[:50]}...")
    
    # Add the ETH price arrays to the training data
    training_data['eth_price_array'] = eth_price_arrays
    
    # Save the updated file
    output_file = 'training_set2_with_btc_eth.csv'
    training_data.to_csv(output_file, index=False)
    
    print(f"\nCompleted! Updated file saved as: {output_file}")
    print(f"Processed {len(eth_price_arrays)} periods")
    
    # Show sample of results
    print("\nSample results:")
    for i in range(min(3, len(training_data))):
        row = training_data.iloc[i]
        print(f"Period {i+1}: {row['price_reference_date'].strftime('%Y-%m-%d')} to {row['target_date'].strftime('%Y-%m-%d')}")
        print(f"  BTC: {row['btc_price_array'][:50]}...")
        print(f"  ETH: {row['eth_price_array'][:50]}...")
        print()

if __name__ == "__main__":
    extract_eth_prices() 