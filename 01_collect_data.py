import requests
import csv
import logging
from datetime import datetime, timedelta
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SolanaPriceCollector:
    def __init__(self):
        """Initialize the Solana price data collector"""
        # Price data API
        self.price_api_url = "https://onchainbe.vela.partners/market/history/So11111111111111111111111111111111111111111?days=365"
        
        # Load price data once at initialization
        self.price_data = self._load_price_data()
        
    def _load_price_data(self) -> dict:
        """Load SOL price data from the API"""
        try:
            logger.info("Loading SOL price data from API...")
            response = requests.get(self.price_api_url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data.get("success"):
                raise ValueError("Price API returned unsuccessful response")
            
            # Convert to date string -> price mapping
            price_dict = {}
            sol_data = data["data"]["So11111111111111111111111111111111111111111"]
            
            for entry in sol_data:
                date_str = str(entry["date"])  # YYYYMMDD format
                # Convert to YYYY-MM-DD format
                formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                price_dict[formatted_date] = entry["price"]
            
            logger.info(f"✅ Loaded {len(price_dict)} price data points")
            
            # Show date range for verification
            dates = sorted(price_dict.keys())
            logger.info(f"📅 Price data range: {dates[0]} to {dates[-1]}")
            
            return price_dict
            
        except Exception as e:
            logger.error(f"❌ Failed to load price data: {e}")
            return {}
    
    def generate_biweekly_dates(self, start_date: str = "2024-07-17", training_end_date: str = "2025-04-15") -> List[str]:
        """Generate biweekly target dates for training set: July 17, 2024 to mid-April 2025"""
        dates = []
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        
        # Training set goes until mid-April 2025, test set comes after
        training_cutoff = datetime.strptime(training_end_date, "%Y-%m-%d")
        
        # Also ensure we have enough price data (need 7 days ahead for prediction)
        latest_price_date = max(self.price_data.keys()) if self.price_data else start_date
        latest_available = datetime.strptime(latest_price_date, "%Y-%m-%d") - timedelta(days=7)
        
        # Use the earlier of training cutoff or available data
        end_date = min(training_cutoff, latest_available)
        
        while current_date <= end_date:
            dates.append(current_date.strftime("%Y-%m-%d"))
            current_date += timedelta(days=14)  # Add 2 weeks
        
        logger.info(f"📋 Generated {len(dates)} biweekly target dates for TRAINING SET")
        logger.info(f"📋 Training range: {dates[0]} to {dates[-1]}")
        logger.info(f"📋 Training cutoff: {training_end_date}")
        logger.info(f"📋 Test set will start after: {training_end_date}")
        return dates
    
    def get_price_on_date(self, date: str) -> Optional[float]:
        """Get SOL price on a specific date"""
        return self.price_data.get(date)
    
    def calculate_price_change(self, start_date: str, end_date: str) -> Optional[float]:
        """Calculate percentage price change between two dates"""
        start_price = self.get_price_on_date(start_date)
        end_price = self.get_price_on_date(end_date)
        
        if start_price and end_price:
            change_pct = ((end_price - start_price) / start_price) * 100
            return round(change_pct, 2)
        
        return None
    
    def create_training_csv(self, output_file: str = "training_set.csv"):
        """Create the complete CSV structure with all dates and price data"""
        logger.info(f"🏗️  Creating training CSV: {output_file}")
        
        # Generate all target dates
        target_dates = self.generate_biweekly_dates()
        
        # Define CSV columns
        csv_columns = [
            'target_date',           # The prediction point date
            'context_start',         # Start of news context window (target_date - 7 days)
            'target_price',          # SOL price on target_date  
            'prediction_end',        # End of prediction window (target_date + 7 days)
            'actual_price',          # SOL price on prediction_end
            'actual_change_pct',     # Actual % change from target_date to prediction_end
            'summarized_context',    # News summary (to be filled later)
            'predicted_price',       # LLM predicted price (to be filled later)
            'predicted_change_pct',  # LLM predicted % change (to be filled later)
            'llm_reasoning',         # LLM reasoning (to be filled later)
            'llm_reflection'         # LLM reflection on errors (to be filled later)
        ]
        
        # Create CSV with all rows
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            
            logger.info(f"📊 Processing {len(target_dates)} target dates...")
            
            for i, target_date in enumerate(target_dates, 1):
                # Calculate related dates
                target_dt = datetime.strptime(target_date, "%Y-%m-%d")
                context_start = (target_dt - timedelta(days=7)).strftime("%Y-%m-%d")
                prediction_end = (target_dt + timedelta(days=7)).strftime("%Y-%m-%d")
                
                # Get price data
                target_price = self.get_price_on_date(target_date)
                actual_price = self.get_price_on_date(prediction_end)
                actual_change_pct = self.calculate_price_change(target_date, prediction_end)
                
                # Create row
                row = {
                    'target_date': target_date,
                    'context_start': context_start,
                    'target_price': target_price if target_price else '',
                    'prediction_end': prediction_end,
                    'actual_price': actual_price if actual_price else '',
                    'actual_change_pct': actual_change_pct if actual_change_pct is not None else '',
                    'summarized_context': '',      # To be filled by news collection
                    'predicted_price': '',         # To be filled by prediction model
                    'predicted_change_pct': '',    # To be filled by prediction model
                    'llm_reasoning': '',           # To be filled by prediction model
                    'llm_reflection': ''           # To be filled by reflection model
                }
                
                writer.writerow(row)
                
                # Log progress
                status = "✅" if actual_change_pct is not None else "⚠️ "
                logger.info(f"{status} {i:2d}/{len(target_dates)}: {target_date} | "
                          f"Price: ${target_price} → ${actual_price} | "
                          f"Change: {actual_change_pct}%")
        
        logger.info(f"🎉 CSV created successfully: {output_file}")
        self.show_summary_stats(target_dates)
    
    def show_summary_stats(self, target_dates: List[str]):
        """Show summary statistics of the collected data"""
        logger.info("\n📈 SUMMARY STATISTICS:")
        
        # Count data availability
        price_data_count = 0
        change_data_count = 0
        price_changes = []
        
        for target_date in target_dates:
            target_dt = datetime.strptime(target_date, "%Y-%m-%d")
            prediction_end = (target_dt + timedelta(days=7)).strftime("%Y-%m-%d")
            
            target_price = self.get_price_on_date(target_date)
            actual_price = self.get_price_on_date(prediction_end)
            
            if target_price:
                price_data_count += 1
            
            if target_price and actual_price:
                change_data_count += 1
                change_pct = self.calculate_price_change(target_date, prediction_end)
                if change_pct is not None:
                    price_changes.append(change_pct)
        
        logger.info(f"   📊 Total target dates: {len(target_dates)}")
        logger.info(f"   💰 Dates with price data: {price_data_count}/{len(target_dates)}")
        logger.info(f"   📈 Dates with change data: {change_data_count}/{len(target_dates)}")
        
        if price_changes:
            avg_change = sum(price_changes) / len(price_changes)
            max_change = max(price_changes)
            min_change = min(price_changes)
            logger.info(f"   📊 Average change: {avg_change:.2f}%")
            logger.info(f"   📊 Max change: {max_change:.2f}%")
            logger.info(f"   📊 Min change: {min_change:.2f}%")
            
            positive_changes = [c for c in price_changes if c > 0]
            logger.info(f"   📊 Positive weeks: {len(positive_changes)}/{len(price_changes)} ({len(positive_changes)/len(price_changes)*100:.1f}%)")
    
    def update_existing_csv(self, csv_file: str = "training_set.csv"):
        """Update an existing CSV with any missing price data"""
        logger.info(f"🔄 Updating existing CSV: {csv_file}")
        
        try:
            # Read existing data
            rows = []
            with open(csv_file, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
            
            # Update rows with missing price data
            updated_count = 0
            for row in rows:
                target_date = row['target_date']
                target_dt = datetime.strptime(target_date, "%Y-%m-%d")
                prediction_end = (target_dt + timedelta(days=7)).strftime("%Y-%m-%d")
                
                # Update missing price data
                if not row.get('target_price'):
                    price = self.get_price_on_date(target_date)
                    if price:
                        row['target_price'] = price
                        updated_count += 1
                
                if not row.get('actual_price'):
                    price = self.get_price_on_date(prediction_end)
                    if price:
                        row['actual_price'] = price
                        updated_count += 1
                
                if not row.get('actual_change_pct'):
                    change = self.calculate_price_change(target_date, prediction_end)
                    if change is not None:
                        row['actual_change_pct'] = change
                        updated_count += 1
            
            # Write back to file
            with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = rows[0].keys() if rows else []
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            
            logger.info(f"✅ Updated {updated_count} missing price data points")
            
        except FileNotFoundError:
            logger.info("📄 No existing CSV found, creating new one...")
            self.create_training_csv(csv_file)
        except Exception as e:
            logger.error(f"❌ Failed to update CSV: {e}")

# Example usage and testing
if __name__ == "__main__":
    collector = SolanaPriceCollector()
    
    # Check if we have price data
    if not collector.price_data:
        logger.error("❌ No price data loaded. Cannot proceed.")
        exit(1)
    
    # Create the training CSV with all dates and price data
    collector.create_training_csv("training_set.csv")
    
    logger.info("\n🎯 Next steps:")
    logger.info("   1. ✅ Price data collection - COMPLETE")
    logger.info("   2. 📰 News collection - TODO (separate script)")
    logger.info("   3. 🤖 LLM predictions - TODO")
    logger.info("   4. 🧠 LLM reflections - TODO") 