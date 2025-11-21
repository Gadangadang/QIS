"""
Find best optimization trial and save parameters.
Reads all trial results, ranks by Sharpe ratio, and saves best params.
"""
import pandas as pd
import json
from pathlib import Path


def find_best_trial(
    trials_dir: str = 'logs/optimization/trials',
    metric: str = 'sharpe',
    output_file: str = 'logs/optimization/best_params.json'
):
    """
    Find the best trial based on performance metric.
    
    Args:
        trials_dir: Directory containing trial folders
        metric: Metric to optimize ('sharpe', 'total_return', 'calmar', etc.)
        output_file: Where to save best parameters JSON
        
    Returns:
        dict with best_params, best_score, trial_name, and full results DataFrame
    """
    trials_path = Path(trials_dir)
    
    if not trials_path.exists():
        print(f"Error: {trials_dir} does not exist")
        return None
    
    # Collect results from all trials
    results = []
    
    for trial_dir in sorted(trials_path.glob('trial_*')):
        # Extract params from folder name (e.g., trial_001_lb200_sl8)
        parts = trial_dir.name.split('_')
        trial_num = int(parts[1])
        
        # Parse lookback and stop_loss from folder name
        params = {}
        for part in parts[2:]:
            if part.startswith('lb'):
                params['lookback'] = int(part[2:])
            elif part.startswith('sl'):
                params['stop_loss_pct'] = int(part[2:]) / 100.0
        
        # Load equity and returns to calculate metrics
        try:
            equity_file = trial_dir / 'stitched_equity.csv'
            returns_file = trial_dir / 'combined_returns.csv'
            
            if not equity_file.exists() or not returns_file.exists():
                print(f"⚠️  Skipping {trial_dir.name} - missing data files")
                continue
            
            equity = pd.read_csv(equity_file, index_col=0, parse_dates=True)
            returns = pd.read_csv(returns_file, index_col=0, parse_dates=True)
            
            # Calculate metrics
            strat_returns = returns['Strategy']
            total_return = (equity['PortfolioValue'].iloc[-1] / equity['PortfolioValue'].iloc[0] - 1) * 100
            sharpe = strat_returns.mean() / strat_returns.std() * (252 ** 0.5) if strat_returns.std() > 0 else 0
            
            # Max drawdown
            rolling_max = equity['PortfolioValue'].expanding().max()
            drawdown = (equity['PortfolioValue'] / rolling_max - 1) * 100
            max_dd = drawdown.min()
            
            # Calmar ratio
            years = len(equity) / 252
            cagr = ((equity['PortfolioValue'].iloc[-1] / equity['PortfolioValue'].iloc[0]) ** (1/years) - 1) * 100
            calmar = cagr / abs(max_dd) if max_dd != 0 else 0
            
            # Load trades for win rate
            trades_files = list(trial_dir.glob('trades_fold_*.csv'))
            if trades_files:
                trades = pd.concat([pd.read_csv(f) for f in trades_files], ignore_index=True)
                win_rate = (trades['pnl_pct'] > 0).sum() / len(trades) * 100 if len(trades) > 0 else 0
                num_trades = len(trades)
            else:
                win_rate = 0
                num_trades = 0
            
            results.append({
                'trial_num': trial_num,
                'trial_name': trial_dir.name,
                'lookback': params.get('lookback'),
                'stop_loss_pct': params.get('stop_loss_pct'),
                'sharpe': sharpe,
                'total_return': total_return,
                'max_drawdown': max_dd,
                'calmar': calmar,
                'win_rate': win_rate,
                'num_trades': num_trades,
                'cagr': cagr,
                'trial_path': str(trial_dir)
            })
            
        except Exception as e:
            print(f"⚠️  Error processing {trial_dir.name}: {e}")
            continue
    
    if not results:
        print("No valid trials found!")
        return None
    
    # Create DataFrame and sort by metric
    df = pd.DataFrame(results)
    df_sorted = df.sort_values(metric, ascending=False)
    
    # Get best trial
    best = df_sorted.iloc[0]
    
    # Prepare best params dict
    best_params = {
        'lookback': int(best['lookback']) if pd.notna(best['lookback']) else None,
        'stop_loss_pct': float(best['stop_loss_pct']) if pd.notna(best['stop_loss_pct']) else None,
        'entry_threshold': 0.02,  # Default - update if you track this
        'exit_threshold': -0.01   # Default - update if you track this
    }
    
    # Save to JSON
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'best_params': best_params,
            'best_score': float(best[metric]),
            'metric': metric,
            'trial_name': best['trial_name'],
            'trial_path': best['trial_path'],
            'all_metrics': {
                'sharpe': float(best['sharpe']),
                'total_return': float(best['total_return']),
                'max_drawdown': float(best['max_drawdown']),
                'calmar': float(best['calmar']),
                'win_rate': float(best['win_rate']),
                'num_trades': int(best['num_trades']),
                'cagr': float(best['cagr'])
            }
        }, f, indent=2)
    
    print(f"=" * 70)
    print(f"BEST TRIAL: {best['trial_name']}")
    print(f"=" * 70)
    print(f"Optimized for: {metric}")
    print(f"Score: {best[metric]:.4f}")
    print()
    print("Parameters:")
    print(f"  lookback:        {best['lookback']}")
    print(f"  stop_loss_pct:   {best['stop_loss_pct']:.2%}")
    print()
    print("Performance:")
    print(f"  Sharpe Ratio:    {best['sharpe']:.3f}")
    print(f"  Total Return:    {best['total_return']:.2f}%")
    print(f"  CAGR:            {best['cagr']:.2f}%")
    print(f"  Max Drawdown:    {best['max_drawdown']:.2f}%")
    print(f"  Calmar Ratio:    {best['calmar']:.3f}")
    print(f"  Win Rate:        {best['win_rate']:.1f}%")
    print(f"  Num Trades:      {best['num_trades']}")
    print()
    print(f"✅ Best parameters saved to: {output_path}")
    print(f"📂 Full results: {best['trial_path']}")
    print()
    print("=" * 70)
    print("\nTOP 5 TRIALS:")
    print("=" * 70)
    print(df_sorted[['trial_name', 'sharpe', 'total_return', 'max_drawdown', 'win_rate']].head().to_string(index=False))
    
    return {
        'best_params': best_params,
        'best_score': float(best[metric]),
        'trial_name': best['trial_name'],
        'results_df': df_sorted
    }


if __name__ == '__main__':
    # Run with default settings (optimize for Sharpe)
    result = find_best_trial(
        trials_dir='logs/optimization/trials',
        metric='sharpe',
        output_file='logs/optimization/best_params.json'
    )
    
    # Optionally save full results CSV
    if result is not None:
        results_csv = 'logs/optimization/all_trials_ranked.csv'
        result['results_df'].to_csv(results_csv, index=False)
        print(f"\n📊 Full rankings saved to: {results_csv}")
