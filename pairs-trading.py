import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StatisticalArbitrageSystem:
    """
    A pairs trading system that identifies cointegrated pairs and trades mean reversion.
    This demonstrates key quant skills: statistical testing, risk management, and backtesting.
    """
    
    def __init__(self, lookback_window=60, zscore_entry=2.0, zscore_exit=0.5, 
                 max_holding_period=20, position_size=0.1):
        self.lookback_window = lookback_window
        self.zscore_entry = zscore_entry
        self.zscore_exit = zscore_exit
        self.max_holding_period = max_holding_period
        self.position_size = position_size
        self.pairs = []
        self.signals = {}
        
    def generate_synthetic_data(self, n_days=500):
        """Generate synthetic price data for demonstration"""
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        
        # Create correlated assets
        np.random.seed(42)
        
        # Base trend
        trend = np.cumsum(np.random.randn(n_days) * 0.01)
        
        # Asset 1: Base trend + noise
        asset1 = 100 * np.exp(trend + np.cumsum(np.random.randn(n_days) * 0.02))
        
        # Asset 2: Similar to Asset 1 but with mean-reverting spread
        spread = np.zeros(n_days)
        spread[0] = 0
        for i in range(1, n_days):
            spread[i] = 0.9 * spread[i-1] + np.random.randn() * 0.01
        
        asset2 = asset1 * np.exp(spread)
        
        # Asset 3: Less correlated
        asset3 = 100 * np.exp(np.cumsum(np.random.randn(n_days) * 0.03))
        
        data = pd.DataFrame({
            'AAPL': asset1,
            'MSFT': asset2,
            'GOOGL': asset3
        }, index=dates)
        
        return data
    
    def test_cointegration(self, series1, series2):
        """
        Test for cointegration using Engle-Granger method
        Returns: (is_cointegrated, hedge_ratio, spread_stats)
        """
        # Run linear regression
        X = series1.values.reshape(-1, 1)
        y = series2.values
        
        model = LinearRegression()
        model.fit(X, y)
        hedge_ratio = model.coef_[0]
        
        # Calculate spread
        spread = y - hedge_ratio * series1.values
        
        # Test stationarity of spread using Augmented Dickey-Fuller
        from statsmodels.tsa.stattools import adfuller
        adf_result = adfuller(spread)
        
        # Calculate spread statistics
        spread_mean = np.mean(spread)
        spread_std = np.std(spread)
        
        # Cointegrated if p-value < 0.05
        is_cointegrated = adf_result[1] < 0.05
        
        return is_cointegrated, hedge_ratio, {
            'mean': spread_mean,
            'std': spread_std,
            'adf_pvalue': adf_result[1]
        }
    
    def find_pairs(self, data):
        """Identify cointegrated pairs from the universe"""
        symbols = data.columns.tolist()
        self.pairs = []
        
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                series1 = data[symbols[i]]
                series2 = data[symbols[j]]
                
                is_coint, hedge_ratio, stats = self.test_cointegration(series1, series2)
                
                if is_coint:
                    self.pairs.append({
                        'asset1': symbols[i],
                        'asset2': symbols[j],
                        'hedge_ratio': hedge_ratio,
                        'spread_mean': stats['mean'],
                        'spread_std': stats['std'],
                        'adf_pvalue': stats['adf_pvalue']
                    })
                    
        return self.pairs
    
    def calculate_signals(self, data, pair):
        """Generate trading signals for a pair"""
        asset1 = data[pair['asset1']]
        asset2 = data[pair['asset2']]
        
        # Calculate spread
        spread = asset2 - pair['hedge_ratio'] * asset1
        
        # Calculate rolling statistics
        spread_mean = spread.rolling(window=self.lookback_window).mean()
        spread_std = spread.rolling(window=self.lookback_window).std()
        
        # Calculate z-score
        zscore = (spread - spread_mean) / spread_std
        
        # Generate signals
        signals = pd.DataFrame(index=data.index)
        signals['spread'] = spread
        signals['zscore'] = zscore
        signals['position'] = 0
        
        # Entry signals
        signals.loc[zscore > self.zscore_entry, 'position'] = -1  # Short spread
        signals.loc[zscore < -self.zscore_entry, 'position'] = 1  # Long spread
        
        # Exit signals
        in_position = False
        position_days = 0
        
        for i in range(len(signals)):
            if i == 0:
                continue
                
            # Carry forward position
            if signals['position'].iloc[i] == 0 and in_position:
                signals['position'].iloc[i] = signals['position'].iloc[i-1]
                position_days += 1
                
                # Exit conditions
                if (abs(signals['zscore'].iloc[i]) < self.zscore_exit or 
                    position_days > self.max_holding_period):
                    signals['position'].iloc[i] = 0
                    in_position = False
                    position_days = 0
            
            # New position
            elif signals['position'].iloc[i] != 0:
                in_position = True
                position_days = 1
        
        return signals
    
    def backtest_pair(self, data, pair, signals):
        """Backtest a single pair with proper transaction costs"""
        asset1_returns = data[pair['asset1']].pct_change()
        asset2_returns = data[pair['asset2']].pct_change()
        
        # Calculate position sizes (dollar neutral)
        position1 = -signals['position'] * pair['hedge_ratio'] * self.position_size
        position2 = signals['position'] * self.position_size
        
        # Calculate returns
        strategy_returns = position1.shift(1) * asset1_returns + position2.shift(1) * asset2_returns
        
        # Transaction costs (5 bps per trade)
        trades = signals['position'].diff().abs()
        transaction_costs = trades * 0.0005
        
        # Net returns
        net_returns = strategy_returns - transaction_costs
        
        # Calculate performance metrics
        cumulative_returns = (1 + net_returns).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1
        
        # Sharpe ratio (assuming 252 trading days)
        sharpe_ratio = np.sqrt(252) * net_returns.mean() / net_returns.std()
        
        # Maximum drawdown
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_days = (net_returns > 0).sum()
        total_days = (net_returns != 0).sum()
        win_rate = winning_days / total_days if total_days > 0 else 0
        
        return {
            'cumulative_returns': cumulative_returns,
            'net_returns': net_returns,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': trades.sum() / 2
        }
    
    def run_full_backtest(self, data):
        """Run complete backtest on all pairs"""
        results = {}
        
        print("Finding cointegrated pairs...")
        self.find_pairs(data)
        
        if not self.pairs:
            print("No cointegrated pairs found!")
            return results
        
        print(f"Found {len(self.pairs)} cointegrated pairs")
        
        for pair in self.pairs:
            print(f"\nAnalyzing {pair['asset1']}-{pair['asset2']} pair:")
            print(f"  Hedge ratio: {pair['hedge_ratio']:.4f}")
            print(f"  ADF p-value: {pair['adf_pvalue']:.4f}")
            
            # Generate signals
            signals = self.calculate_signals(data, pair)
            
            # Backtest
            backtest_results = self.backtest_pair(data, pair, signals)
            
            print(f"  Total Return: {backtest_results['total_return']:.2%}")
            print(f"  Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {backtest_results['max_drawdown']:.2%}")
            print(f"  Win Rate: {backtest_results['win_rate']:.2%}")
            print(f"  Number of Trades: {int(backtest_results['num_trades'])}")
            
            # Store results
            pair_name = f"{pair['asset1']}-{pair['asset2']}"
            results[pair_name] = {
                'pair': pair,
                'signals': signals,
                'backtest': backtest_results
            }
        
        return results
    
    def plot_results(self, results):
        """Visualize backtest results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Statistical Arbitrage System Results', fontsize=16)
        
        # Plot 1: Cumulative returns for all pairs
        ax1 = axes[0, 0]
        for pair_name, result in results.items():
            ax1.plot(result['backtest']['cumulative_returns'], label=pair_name)
        ax1.set_title('Cumulative Returns by Pair')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative Returns')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Z-scores for best pair
        best_pair = max(results.items(), key=lambda x: x[1]['backtest']['sharpe_ratio'])
        ax2 = axes[0, 1]
        ax2.plot(best_pair[1]['signals']['zscore'])
        ax2.axhline(y=self.zscore_entry, color='r', linestyle='--', label='Entry threshold')
        ax2.axhline(y=-self.zscore_entry, color='r', linestyle='--')
        ax2.axhline(y=self.zscore_exit, color='g', linestyle='--', label='Exit threshold')
        ax2.axhline(y=-self.zscore_exit, color='g', linestyle='--')
        ax2.set_title(f'Z-Score Evolution: {best_pair[0]}')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Z-Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Performance metrics comparison
        ax3 = axes[1, 0]
        metrics = ['Sharpe Ratio', 'Total Return', 'Win Rate']
        pairs = list(results.keys())
        x = np.arange(len(pairs))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            if metric == 'Sharpe Ratio':
                values = [results[p]['backtest']['sharpe_ratio'] for p in pairs]
            elif metric == 'Total Return':
                values = [results[p]['backtest']['total_return'] * 100 for p in pairs]
            else:  # Win Rate
                values = [results[p]['backtest']['win_rate'] * 100 for p in pairs]
            
            ax3.bar(x + i*width, values, width, label=metric)
        
        ax3.set_xlabel('Pairs')
        ax3.set_ylabel('Value')
        ax3.set_title('Performance Metrics by Pair')
        ax3.set_xticks(x + width)
        ax3.set_xticklabels(pairs)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Drawdown analysis
        ax4 = axes[1, 1]
        for pair_name, result in results.items():
            returns = result['backtest']['cumulative_returns']
            running_max = returns.expanding().max()
            drawdown = (returns - running_max) / running_max
            ax4.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, label=pair_name)
        ax4.set_title('Drawdown Analysis')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Drawdown')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig

# Example usage
if __name__ == "__main__":
    # Initialize the system
    system = StatisticalArbitrageSystem(
        lookback_window=60,
        zscore_entry=2.0,
        zscore_exit=0.5,
        max_holding_period=20,
        position_size=0.1
    )
    
    # Generate synthetic data (in practice, you'd use real market data)
    print("Generating synthetic market data...")
    data = system.generate_synthetic_data(n_days=500)
    
    # Run the backtest
    print("\nRunning backtest...")
    results = system.run_full_backtest(data)
    
    # Plot results
    if results:
        print("\nGenerating performance plots...")
        system.plot_results(results)
    
    # Risk analysis
    print("\n" + "="*50)
    print("RISK ANALYSIS AND CONSIDERATIONS")
    print("="*50)
    print("""
    1. Model Risk:
       - Cointegration relationships can break down
       - Parameters may be overfit to historical data
       - Need to monitor relationship stability
    
    2. Execution Risk:
       - Slippage in fast markets
       - Difficulty getting fills on both legs
       - Short sale availability
    
    3. Market Risk:
       - Systematic shocks affecting all pairs
       - Sector rotation breaking relationships
       - Liquidity crises
    
    4. Improvements for Production:
       - Dynamic parameter adjustment
       - Multiple timeframe analysis
       - Correlation with market regimes
       - Better position sizing (Kelly criterion)
       - Stop-loss mechanisms
    """)