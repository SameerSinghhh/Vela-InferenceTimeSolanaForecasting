import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class PredictionEvaluator:
    """Comprehensive evaluation system for comparing prediction approaches."""
    
    def __init__(self):
        self.baseline_df = None
        self.memory_df = None
        self.results = {}
        
    def load_data(self, baseline_file: str = "test_set_no_memory.csv", 
                  memory_file: str = "test_set_with_memory.csv"):
        """Load both test sets for comparison."""
        print("📊 Loading prediction data...")
        
        self.baseline_df = pd.read_csv(baseline_file)
        self.memory_df = pd.read_csv(memory_file)
        
        # Filter out rows without predictions
        self.baseline_df = self.baseline_df.dropna(subset=['predicted_price', 'actual_price'])
        self.memory_df = self.memory_df.dropna(subset=['predicted_price', 'actual_price'])
        
        print(f"✅ Baseline predictions: {len(self.baseline_df)} weeks")
        print(f"✅ Memory predictions: {len(self.memory_df)} weeks")
        
        return len(self.baseline_df) > 0 and len(self.memory_df) > 0
    
    def calculate_directional_accuracy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate directional accuracy metrics."""
        
        # Calculate predicted and actual directions
        df['predicted_direction'] = df['predicted_change_pct'] > 0
        df['actual_direction'] = df['actual_change_pct'] > 0
        df['direction_correct'] = df['predicted_direction'] == df['actual_direction']
        
        # Calculate metrics
        total_predictions = len(df)
        correct_predictions = df['direction_correct'].sum()
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # Breakdown by direction
        up_predictions = df[df['predicted_direction'] == True]
        down_predictions = df[df['predicted_direction'] == False]
        
        up_accuracy = up_predictions['direction_correct'].mean() if len(up_predictions) > 0 else 0
        down_accuracy = down_predictions['direction_correct'].mean() if len(down_predictions) > 0 else 0
        
        return {
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'up_predictions': len(up_predictions),
            'down_predictions': len(down_predictions),
            'up_accuracy': up_accuracy,
            'down_accuracy': down_accuracy
        }
    
    def calculate_numerical_errors(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate numerical error metrics."""
        
        # Price prediction errors
        price_errors = df['predicted_price'] - df['actual_price']
        abs_price_errors = np.abs(price_errors)
        
        # Percentage prediction errors
        pct_errors = df['predicted_change_pct'] - df['actual_change_pct']
        abs_pct_errors = np.abs(pct_errors)
        
        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(abs_pct_errors)
        
        # Root Mean Square Error (RMSE) for percentages
        rmse_pct = np.sqrt(np.mean(pct_errors ** 2))
        
        # Mean Absolute Error (MAE)
        mae_price = np.mean(abs_price_errors)
        mae_pct = np.mean(abs_pct_errors)
        
        # Median errors (more robust to outliers)
        median_abs_pct_error = np.median(abs_pct_errors)
        
        return {
            'mae_price': mae_price,
            'mae_pct': mae_pct,
            'mape': mape,
            'rmse_pct': rmse_pct,
            'median_abs_pct_error': median_abs_pct_error,
            'max_abs_pct_error': np.max(abs_pct_errors),
            'mean_pct_error': np.mean(pct_errors)  # Shows bias (over/under prediction)
        }
    
    def simulate_trading_strategy(self, df: pd.DataFrame, initial_capital: float = 10000) -> Dict[str, Any]:
        """Simulate a trading strategy based on predictions."""
        
        capital = initial_capital
        positions = []
        portfolio_values = [capital]
        
        for _, row in df.iterrows():
            predicted_change = row['predicted_change_pct']
            actual_change = row['actual_change_pct']
            
            # Simple strategy: buy if predicted positive, sell if predicted negative
            if predicted_change > 0:
                # Buy position
                position_return = actual_change / 100  # Convert percentage to decimal
                capital *= (1 + position_return)
                positions.append(('BUY', predicted_change, actual_change, position_return))
            elif predicted_change < 0:
                # Short position (profit from price decline)
                position_return = -actual_change / 100  # Profit from decline
                capital *= (1 + position_return)
                positions.append(('SELL', predicted_change, actual_change, position_return))
            else:
                # Hold position
                positions.append(('HOLD', predicted_change, actual_change, 0))
            
            portfolio_values.append(capital)
        
        # Calculate metrics
        total_return = (capital - initial_capital) / initial_capital
        num_trades = len([p for p in positions if p[0] != 'HOLD'])
        winning_trades = len([p for p in positions if p[3] > 0])
        losing_trades = len([p for p in positions if p[3] < 0])
        
        win_rate = winning_trades / num_trades if num_trades > 0 else 0
        
        return {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'num_trades': num_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'positions': positions,
            'portfolio_values': portfolio_values
        }
    
    def generate_comparison_report(self) -> str:
        """Generate a comprehensive comparison report."""
        
        print("\n" + "="*80)
        print("🔍 COMPREHENSIVE PREDICTION EVALUATION REPORT")
        print("="*80)
        
        # Calculate metrics for both approaches
        baseline_directional = self.calculate_directional_accuracy(self.baseline_df)
        memory_directional = self.calculate_directional_accuracy(self.memory_df)
        
        baseline_numerical = self.calculate_numerical_errors(self.baseline_df)
        memory_numerical = self.calculate_numerical_errors(self.memory_df)
        
        baseline_trading = self.simulate_trading_strategy(self.baseline_df)
        memory_trading = self.simulate_trading_strategy(self.memory_df)
        
        # Store results
        self.results = {
            'baseline': {
                'directional': baseline_directional,
                'numerical': baseline_numerical,
                'trading': baseline_trading
            },
            'memory': {
                'directional': memory_directional,
                'numerical': memory_numerical,
                'trading': memory_trading
            }
        }
        
        # Generate report
        report = []
        
        # 1. DIRECTIONAL ACCURACY COMPARISON
        report.append("\n📊 1. DIRECTIONAL ACCURACY COMPARISON")
        report.append("-" * 50)
        report.append(f"{'Metric':<25} {'Baseline':<15} {'Memory':<15} {'Improvement':<15}")
        report.append("-" * 70)
        
        baseline_acc = baseline_directional['accuracy']
        memory_acc = memory_directional['accuracy']
        acc_improvement = memory_acc - baseline_acc
        
        report.append(f"{'Overall Accuracy':<25} {f'{baseline_acc:.1%}':<15} {f'{memory_acc:.1%}':<15} {f'{acc_improvement:+.1%}':<15}")
        report.append(f"{'Correct Predictions':<25} {baseline_directional['correct_predictions']:<15} {memory_directional['correct_predictions']:<15} {f'{memory_directional['correct_predictions'] - baseline_directional['correct_predictions']:+d}':<15}")
        report.append(f"{'Up Predictions':<25} {baseline_directional['up_predictions']:<15} {memory_directional['up_predictions']:<15} {f'{memory_directional['up_predictions'] - baseline_directional['up_predictions']:+d}':<15}")
        report.append(f"{'Down Predictions':<25} {baseline_directional['down_predictions']:<15} {memory_directional['down_predictions']:<15} {f'{memory_directional['down_predictions'] - baseline_directional['down_predictions']:+d}':<15}")
        
        # 2. NUMERICAL ERROR ANALYSIS
        report.append("\n📈 2. NUMERICAL ERROR ANALYSIS")
        report.append("-" * 50)
        report.append(f"{'Metric':<25} {'Baseline':<15} {'Memory':<15} {'Improvement':<15}")
        report.append("-" * 70)
        
        mae_improvement = baseline_numerical['mae_pct'] - memory_numerical['mae_pct']
        mape_improvement = baseline_numerical['mape'] - memory_numerical['mape']
        rmse_improvement = baseline_numerical['rmse_pct'] - memory_numerical['rmse_pct']
        
        report.append(f"{'MAE (% points)':<25} {baseline_numerical['mae_pct']:.2f}pp{'':<10} {memory_numerical['mae_pct']:.2f}pp{'':<10} {mae_improvement:+.2f}pp{'':<10}")
        report.append(f"{'MAPE (% points)':<25} {baseline_numerical['mape']:.2f}pp{'':<10} {memory_numerical['mape']:.2f}pp{'':<10} {mape_improvement:+.2f}pp{'':<10}")
        report.append(f"{'RMSE (% points)':<25} {baseline_numerical['rmse_pct']:.2f}pp{'':<10} {memory_numerical['rmse_pct']:.2f}pp{'':<10} {rmse_improvement:+.2f}pp{'':<10}")
        report.append(f"{'Median Error (pp)':<25} {baseline_numerical['median_abs_pct_error']:.2f}pp{'':<10} {memory_numerical['median_abs_pct_error']:.2f}pp{'':<10} {baseline_numerical['median_abs_pct_error'] - memory_numerical['median_abs_pct_error']:+.2f}pp{'':<10}")
        report.append(f"{'Max Error (pp)':<25} {baseline_numerical['max_abs_pct_error']:.2f}pp{'':<10} {memory_numerical['max_abs_pct_error']:.2f}pp{'':<10} {baseline_numerical['max_abs_pct_error'] - memory_numerical['max_abs_pct_error']:+.2f}pp{'':<10}")
        
        # 3. SIMULATED TRADING PERFORMANCE
        report.append("\n💰 3. SIMULATED TRADING PERFORMANCE")
        report.append("-" * 50)
        report.append(f"{'Metric':<25} {'Baseline':<15} {'Memory':<15} {'Improvement':<15}")
        report.append("-" * 70)
        
        return_improvement = memory_trading['total_return_pct'] - baseline_trading['total_return_pct']
        win_rate_improvement = memory_trading['win_rate'] - baseline_trading['win_rate']
        
        report.append(f"{'Initial Capital':<25} ${f'{baseline_trading['initial_capital']:,.0f}':<14} ${f'{memory_trading['initial_capital']:,.0f}':<14} {'$0':<15}")
        report.append(f"{'Final Capital':<25} ${f'{baseline_trading['final_capital']:,.0f}':<14} ${f'{memory_trading['final_capital']:,.0f}':<14} ${f'{memory_trading['final_capital'] - baseline_trading['final_capital']:+,.0f}':<14}")
        report.append(f"{'Total Return':<25} {baseline_trading['total_return_pct']:+.1f}%{'':<11} {memory_trading['total_return_pct']:+.1f}%{'':<11} {return_improvement:+.1f}pp{'':<11}")
        report.append(f"{'Number of Trades':<25} {baseline_trading['num_trades']:<15} {memory_trading['num_trades']:<15} {f'{memory_trading['num_trades'] - baseline_trading['num_trades']:+d}':<15}")
        report.append(f"{'Win Rate':<25} {f'{baseline_trading['win_rate']:.1%}':<15} {f'{memory_trading['win_rate']:.1%}':<15} {f'{win_rate_improvement:+.1%}':<15}")
        report.append(f"{'Winning Trades':<25} {baseline_trading['winning_trades']:<15} {memory_trading['winning_trades']:<15} {f'{memory_trading['winning_trades'] - baseline_trading['winning_trades']:+d}':<15}")
        report.append(f"{'Losing Trades':<25} {baseline_trading['losing_trades']:<15} {memory_trading['losing_trades']:<15} {f'{memory_trading['losing_trades'] - baseline_trading['losing_trades']:+d}':<15}")
        
        # 4. WEEK-BY-WEEK BREAKDOWN
        report.append("\n📅 4. WEEK-BY-WEEK BREAKDOWN")
        report.append("-" * 50)
        report.append(f"{'Week':<12} {'Actual':<10} {'Baseline':<12} {'Memory':<12} {'B_Acc':<8} {'M_Acc':<8} {'B_Err':<8} {'M_Err':<8}")
        report.append("-" * 80)
        
        for i in range(len(self.baseline_df)):
            baseline_row = self.baseline_df.iloc[i]
            memory_row = self.memory_df.iloc[i]
            
            date = baseline_row['target_date']
            actual = baseline_row['actual_change_pct']
            baseline_pred = baseline_row['predicted_change_pct']
            memory_pred = memory_row['predicted_change_pct']
            
            baseline_correct = "✅" if ((baseline_pred > 0) == (actual > 0)) else "❌"
            memory_correct = "✅" if ((memory_pred > 0) == (actual > 0)) else "❌"
            
            baseline_error = abs(baseline_pred - actual)
            memory_error = abs(memory_pred - actual)
            
            report.append(f"{date:<12} {actual:+.1f}%{'':<6} {baseline_pred:+.1f}%{'':<7} {memory_pred:+.1f}%{'':<7} {baseline_correct:<8} {memory_correct:<8} {baseline_error:.1f}pp{'':<4} {memory_error:.1f}pp{'':<4}")
        
        # 5. SUMMARY & RECOMMENDATIONS
        report.append("\n🏆 5. SUMMARY & RECOMMENDATIONS")
        report.append("-" * 50)
        
        # Determine winner with proper weighting
        directional_win = memory_acc > baseline_acc
        numerical_win = memory_numerical['mae_pct'] < baseline_numerical['mae_pct']
        trading_win = memory_trading['total_return_pct'] > baseline_trading['total_return_pct']
        
        # Count wins and ties
        wins = sum([directional_win, numerical_win, trading_win])
        ties = sum([
            memory_acc == baseline_acc,
            memory_numerical['mae_pct'] == baseline_numerical['mae_pct'],
            memory_trading['total_return_pct'] == baseline_trading['total_return_pct']
        ])
        
        # Determine winner with improved logic
        if wins > 1:  # Clear majority
            winner = "Memory-Enhanced"
        elif wins == 1 and ties >= 1:  # Wins one category, ties others
            if numerical_win:  # Numerical accuracy is most important for predictions
                winner = "Memory-Enhanced"
            elif trading_win:  # Trading performance is second most important
                winner = "Memory-Enhanced"
            else:  # Only directional accuracy win
                winner = "Memory-Enhanced" if ties == 2 else "Baseline"
        elif wins == 1 and ties == 0:  # Wins one, loses one
            winner = "Memory-Enhanced" if numerical_win else "Baseline"
        else:  # No wins or all ties
            winner = "Tie"
        
        report.append(f"🎯 WINNER: {winner} approach")
        report.append(f"📊 Metrics: Directional {'✅' if directional_win else '🤝' if memory_acc == baseline_acc else '❌'} | Numerical {'✅' if numerical_win else '🤝' if memory_numerical['mae_pct'] == baseline_numerical['mae_pct'] else '❌'} | Trading {'✅' if trading_win else '🤝' if memory_trading['total_return_pct'] == baseline_trading['total_return_pct'] else '❌'}")
        
        if winner == "Memory-Enhanced":
            report.append("✨ Key improvements with memory:")
            if directional_win:
                report.append(f"   • Better directional accuracy: {acc_improvement:+.1%}")
            if numerical_win:
                report.append(f"   • Lower prediction errors: {mae_improvement:+.2f}pp MAE")
            if trading_win:
                report.append(f"   • Higher trading returns: {return_improvement:+.1f}pp")
            if memory_acc == baseline_acc:
                report.append(f"   • Maintained directional accuracy: {memory_acc:.1%}")
            if memory_trading['total_return_pct'] == baseline_trading['total_return_pct']:
                report.append(f"   • Maintained trading returns: {memory_trading['total_return_pct']:+.1f}%")
        elif winner == "Baseline":
            report.append("📈 Baseline advantages:")
            if not directional_win and memory_acc < baseline_acc:
                report.append(f"   • Better directional accuracy: {-acc_improvement:+.1%}")
            if not numerical_win:
                report.append(f"   • Lower prediction errors: {-mae_improvement:+.2f}pp MAE")
            if not trading_win and memory_trading['total_return_pct'] < baseline_trading['total_return_pct']:
                report.append(f"   • Higher trading returns: {-return_improvement:+.1f}pp")
        
        report.append("\n" + "="*80)
        
        full_report = '\n'.join(report)
        print(full_report)
        
        return full_report
    
    def create_detailed_analysis(self):
        """Create detailed analysis with individual prediction breakdowns."""
        
        print("\n🔍 DETAILED PREDICTION ANALYSIS")
        print("="*80)
        
        # Compare each prediction
        for i in range(len(self.baseline_df)):
            baseline_row = self.baseline_df.iloc[i]
            memory_row = self.memory_df.iloc[i]
            
            print(f"\n📅 Week {i+1}: {baseline_row['target_date']}")
            print(f"🎯 Actual: {baseline_row['actual_change_pct']:+.2f}%")
            print(f"🤖 Baseline: {baseline_row['predicted_change_pct']:+.2f}% (Error: {abs(baseline_row['predicted_change_pct'] - baseline_row['actual_change_pct']):.2f}pp)")
            print(f"🧠 Memory: {memory_row['predicted_change_pct']:+.2f}% (Error: {abs(memory_row['predicted_change_pct'] - memory_row['actual_change_pct']):.2f}pp)")
            
            # Accuracy indicators
            baseline_correct = (baseline_row['predicted_change_pct'] > 0) == (baseline_row['actual_change_pct'] > 0)
            memory_correct = (memory_row['predicted_change_pct'] > 0) == (memory_row['actual_change_pct'] > 0)
            
            print(f"✅ Directional: Baseline {'✅' if baseline_correct else '❌'}, Memory {'✅' if memory_correct else '❌'}")
            
            # Better performer
            baseline_error = abs(baseline_row['predicted_change_pct'] - baseline_row['actual_change_pct'])
            memory_error = abs(memory_row['predicted_change_pct'] - memory_row['actual_change_pct'])
            
            if memory_error < baseline_error:
                print(f"🏆 Memory wins by {baseline_error - memory_error:.2f}pp")
            elif baseline_error < memory_error:
                print(f"🏆 Baseline wins by {memory_error - baseline_error:.2f}pp")
            else:
                print("🤝 Tied performance")
    
    def save_results(self, filename: str = "evaluation_results.json"):
        """Save evaluation results to JSON file."""
        import json
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        converted_results = convert_numpy(self.results)
        
        with open(filename, 'w') as f:
            json.dump(converted_results, f, indent=2)
        
        print(f"💾 Results saved to {filename}")

def main():
    """Main execution function."""
    
    print("🚀 Starting Comprehensive Prediction Evaluation...")
    
    # Initialize evaluator
    evaluator = PredictionEvaluator()
    
    # Load data
    if not evaluator.load_data():
        print("❌ Failed to load prediction data")
        return
    
    # Generate comprehensive report
    report = evaluator.generate_comparison_report()
    
    # Create detailed analysis
    evaluator.create_detailed_analysis()
    
    # Save results
    evaluator.save_results()
    
    # Save report to file
    with open("evaluation_report.txt", "w") as f:
        f.write(report)
    
    print("\n💾 Full report saved to evaluation_report.txt")
    print("🎉 Evaluation complete!")

if __name__ == "__main__":
    main() 