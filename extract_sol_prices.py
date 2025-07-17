#!/usr/bin/env python3
"""
Script to extract SOL Open prices and add sol_price_array + target_date_sol_price + prediction_date_sol_price + actual_change_pct columns
"""

import pandas as pd
from datetime import datetime, timedelta

def extract_sol_prices():
    """Extract SOL Open prices and add to existing file"""
    
    # Read the SOL price data (tab-separated)
    print("Reading SOL price data...")
    sol_data = pd.read_csv('price_data_sol.csv', sep='\t')
    
    # Convert Date column to datetime
    sol_data['Date'] = pd.to_datetime(sol_data['Date'])
    
    # Sort by date (ascending) for easier processing
    sol_data = sol_data.sort_values('Date').reset_index(drop=True)
    
    # Read the existing training set (check if ETH version exists, otherwise use BTC version)
    print("Reading existing training set...")
    try:
        training_data = pd.read_csv('training_set2_with_btc_eth.csv')
        print("Found BTC+ETH version")
    except FileNotFoundError:
        training_data = pd.read_csv('training_set2_with_btc.csv')
        print("Found BTC-only version")
    
    # Convert date columns to datetime
    training_data['price_reference_date'] = pd.to_datetime(training_data['price_reference_date'])
    training_data['target_date'] = pd.to_datetime(training_data['target_date'])
    training_data['prediction_date'] = pd.to_datetime(training_data['prediction_date'])
    
    # Create lists to store the SOL data
    sol_price_arrays = []
    target_date_sol_prices = []
    prediction_date_sol_prices = []
    actual_change_pcts = []
    
    print("Extracting SOL prices for each period...")
    
    for idx, row in training_data.iterrows():
        start_date = row['price_reference_date']
        end_date = row['target_date']
        prediction_date = row['prediction_date']
        
        print(f"Processing {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}, prediction: {prediction_date.strftime('%Y-%m-%d')}")
        
        # Get prices for the date range (inclusive)
        mask = (sol_data['Date'] >= start_date) & (sol_data['Date'] <= end_date)
        period_data = sol_data.loc[mask, ['Date', 'Open']].copy()
        
        if len(period_data) == 0:
            print(f"  WARNING: No SOL data found for period {start_date} to {end_date}")
            sol_price_arrays.append("[]")
            target_date_sol_prices.append(None)
            prediction_date_sol_prices.append(None)
            actual_change_pcts.append(None)
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
        sol_price_arrays.append(price_array_str)
        
        # Get the SOL price specifically for the target date
        target_date_mask = sol_data['Date'] == end_date
        target_date_data = sol_data.loc[target_date_mask, 'Open']
        
        target_price = None
        if len(target_date_data) > 0:
            target_price = target_date_data.iloc[0]
            if isinstance(target_price, str):
                target_price = float(target_price.replace(',', ''))
            else:
                target_price = float(target_price)
            target_date_sol_prices.append(target_price)
            print(f"  Found {len(clean_prices)} SOL prices, target date price: {target_price}")
        else:
            print(f"  WARNING: No SOL price found for target date {end_date}")
            target_date_sol_prices.append(None)
        
        # Get the SOL price for the prediction date (actual_price)
        prediction_date_mask = sol_data['Date'] == prediction_date
        prediction_date_data = sol_data.loc[prediction_date_mask, 'Open']
        
        actual_price = None
        change_percent = None
        
        if len(prediction_date_data) > 0:
            prediction_price = prediction_date_data.iloc[0]
            if isinstance(prediction_price, str):
                prediction_price = float(prediction_price.replace(',', ''))
            else:
                prediction_price = float(prediction_price)
            prediction_date_sol_prices.append(prediction_price)
            
            # Calculate percentage change if we have both prices
            if target_price is not None and prediction_price is not None:
                change_percent = ((prediction_price - target_price) / target_price) * 100
                actual_change_pcts.append(change_percent)
                print(f"  Prediction date SOL price: {prediction_price}, change: {change_percent:.2f}%")
            else:
                actual_change_pcts.append(None)
                print(f"  Prediction date SOL price: {prediction_price}, change: N/A (missing target price)")
        else:
            print(f"  WARNING: No SOL price found for prediction date {prediction_date}")
            prediction_date_sol_prices.append(None)
            actual_change_pcts.append(None)
    
    # Add the SOL columns to the training data
    training_data['sol_price_array'] = sol_price_arrays
    
    # Rename reference_price to target_date_sol_price and populate it
    if 'reference_price' in training_data.columns:
        training_data = training_data.rename(columns={'reference_price': 'target_date_sol_price'})
    else:
        training_data['target_date_sol_price'] = None
    
    training_data['target_date_sol_price'] = target_date_sol_prices
    
    # Add prediction_date_sol_price and update actual_change_pct columns
    training_data['prediction_date_sol_price'] = prediction_date_sol_prices
    training_data['actual_change_pct'] = actual_change_pcts
    
    # Remove old actual_price column if it exists
    if 'actual_price' in training_data.columns:
        training_data = training_data.drop(columns=['actual_price'])
    
    # Reorder columns to put prediction_date_sol_price next to target_date_sol_price, 
    # and actual_change_pct right after that
    cols = list(training_data.columns)
    
    # Find the position of target_date_sol_price
    if 'target_date_sol_price' in cols:
        target_idx = cols.index('target_date_sol_price')
        
        # Remove prediction_date_sol_price and actual_change_pct from their current positions
        if 'prediction_date_sol_price' in cols:
            cols.remove('prediction_date_sol_price')
        if 'actual_change_pct' in cols:
            cols.remove('actual_change_pct')
        
        # Insert them right after target_date_sol_price
        cols.insert(target_idx + 1, 'prediction_date_sol_price')
        cols.insert(target_idx + 2, 'actual_change_pct')
        
        # Reorder the dataframe
        training_data = training_data[cols]
    
    # Save the updated file
    if 'eth_price_array' in training_data.columns:
        output_file = 'training_set2_with_btc_eth_sol.csv'
    else:
        output_file = 'training_set2_with_btc_sol.csv'
    
    training_data.to_csv(output_file, index=False)
    
    print(f"\nCompleted! Updated file saved as: {output_file}")
    print(f"Processed {len(sol_price_arrays)} periods")
    
    # Show sample of results
    print("\nSample results:")
    for i in range(min(3, len(training_data))):
        row = training_data.iloc[i]
        print(f"Period {i+1}: {row['price_reference_date'].strftime('%Y-%m-%d')} to {row['target_date'].strftime('%Y-%m-%d')}")
        print(f"  Target SOL price: {row['target_date_sol_price']}")
        print(f"  Prediction date SOL price: {row['prediction_date_sol_price']}")
        print(f"  Actual change: {row['actual_change_pct']:.2f}%" if row['actual_change_pct'] is not None else "  Actual change: N/A")
        print(f"  SOL array: {row['sol_price_array'][:50]}...")
        if 'btc_price_array' in training_data.columns:
            print(f"  BTC array: {row['btc_price_array'][:50]}...")
        if 'eth_price_array' in training_data.columns:
            print(f"  ETH array: {row['eth_price_array'][:50]}...")
        print()

if __name__ == "__main__":
    extract_sol_prices() 