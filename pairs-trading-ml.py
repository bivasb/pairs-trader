import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MLEnhancedPairsTradingSystem:
    """
    An ML-enhanced pairs trading system that uses Random Forest to rank and select pairs.
    This demonstrates the integration of machine learning with traditional quant strategies.
    """
    
    def __init__(self, lookback_window=60, zscore_entry=2.0, zscore_exit=0.5, 
                 max_holding_period=20, position_size=0.1, ml_features_window=20):
        self.lookback_window = lookback_window
        self.zscore_entry = zscore_entry
        self.zscore_exit = zscore_exit
        self.max_holding_period = max_holding_period
        self.position_size = position_size
        self.ml_features_window = ml_features_window
        self.pairs = []
        self.signals = {}
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_importance = {}
        
    def generate_synthetic_data(self, n_days=500):
        """Generate synthetic price data for demonstration"""
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        
        # Create correlated assets with varying relationships
        np.random.seed(42)
        
        # Base trends
        trend1 = np.cumsum(np.random.randn(n_days) * 0.01)
        trend2 = np.cumsum(np.random.randn(n_days) * 0.01)
        
        # Asset pairs with different characteristics
        # Strong cointegration pair
        asset1 = 100 * np.exp(trend1 + np.cumsum(np.random.randn(n_days) * 0.02))
        spread1 = np.zeros(n_days)
        spread1[0] = 0
        for i in range(1, n_days):
            spread1[i] = 0.9 * spread1[i-1] + np.random.randn() * 0.01
        asset2 = asset1 * np.exp(spread1)
        
        # Moderate cointegration pair
        asset3 = 100 * np.exp(trend1 * 0.8 + trend2 * 0.2 + np.cumsum(np.random.randn(n_days) * 0.025))
        spread2 = np.zeros(n_days)
        for i in range(1, n_days):
            spread2[i] = 0.85 * spread2[i-1] + np.random.randn() * 0.015
        asset4 = asset3 * np.exp(spread2)
        
        # Weak cointegration pair
        asset5 = 100 * np.exp(trend2 + np.cumsum(np.random.randn(n_days) * 0.03))
        asset6 = 100 * np.exp(trend2 * 0.5 + np.cumsum(np.random.randn(n_days) * 0.04))
        
        data = pd.DataFrame({
            'AAPL': asset1,
            'MSFT': asset2,
            'GOOGL': asset3,
            'AMZN': asset4,
            'META': asset5,
            'NFLX': asset6
        }, index=dates)
        
        return data
    
    def calculate_half_life(self, spread):
        """Calculate half-life of mean reversion"""
        try:
            spread_lag = np.roll(spread, 1)
            spread_lag[0] = spread_lag[1]
            spread_ret = spread - spread_lag
            
            # Remove any NaN or infinite values
            mask = ~(np.isnan(spread_ret) | np.isnan(spread_lag) | 
                    np.isinf(spread_ret) | np.isinf(spread_lag))
            spread_ret_clean = spread_ret[mask]
            spread_lag_clean = spread_lag[mask]
            
            if len(spread_ret_clean) < 10:  # Need minimum data points
                return 60.0  # Default to 60 days
            
            # Simple OLS without statsmodels to avoid import issues
            X = spread_lag_clean
            y = spread_ret_clean
            
            # Calculate regression coefficients
            X_mean = np.mean(X)
            y_mean = np.mean(y)
            
            numerator = np.sum((X - X_mean) * (y - y_mean))
            denominator = np.sum((X - X_mean) ** 2)
            
            if denominator == 0:
                return 60.0
            
            beta = numerator / denominator
            
            if beta >= 0 or beta <= -1:
                return 60.0  # Default for non-mean-reverting
            
            halflife = -np.log(2) / beta
            
            # Cap half-life to reasonable range
            return np.clip(halflife, 1.0, 100.0)
            
        except Exception as e:
            return 60.0  # Default value on any error
    
    def calculate_hurst_exponent(self, series, max_lag=20):
        """Calculate Hurst exponent to measure mean reversion tendency"""
        lags = range(2, max_lag)
        tau = []
        
        for lag in lags:
            # Calculate R/S statistic
            rs_values = []
            for start in range(0, len(series) - lag):
                subseries = series[start:start + lag]
                mean = np.mean(subseries)
                deviations = subseries - mean
                Z = np.cumsum(deviations)
                R = np.max(Z) - np.min(Z)
                S = np.std(subseries, ddof=1)
                if S != 0:
                    rs_values.append(R / S)
            
            if rs_values:
                tau.append(np.mean(rs_values))
        
        if len(tau) > 0:
            # Fit log(R/S) = log(c) + H*log(lag)
            log_lags = np.log(list(lags)[:len(tau)])
            log_tau = np.log(tau)
            
            try:
                poly = np.polyfit(log_lags, log_tau, 1)
                return poly[0]  # Hurst exponent
            except:
                return 0.5
        
        return 0.5
    
    def extract_pair_features(self, series1, series2, prices_df):
        """Extract comprehensive features for ML model"""
        features = {}
        
        try:
            # Basic correlation
            features['correlation'] = series1.corr(series2)
            features['correlation_60d'] = series1.tail(60).corr(series2.tail(60)) if len(series1) >= 60 else features['correlation']
            features['correlation_20d'] = series1.tail(20).corr(series2.tail(20)) if len(series1) >= 20 else features['correlation']
            
            # Cointegration test
            X = series1.values.reshape(-1, 1)
            y = series2.values
            model = LinearRegression()
            model.fit(X, y)
            hedge_ratio = model.coef_[0]
            spread = y - hedge_ratio * series1.values
            
            # Stationarity test with error handling
            try:
                from statsmodels.tsa.stattools import adfuller
                adf_result = adfuller(spread)
                features['adf_pvalue'] = adf_result[1]
                features['adf_statistic'] = adf_result[0]
            except:
                features['adf_pvalue'] = 0.5  # Default to non-stationary
                features['adf_statistic'] = 0.0
            
            # Spread characteristics with nan handling
            features['spread_mean'] = np.nan_to_num(np.mean(spread), 0)
            features['spread_std'] = np.nan_to_num(np.std(spread), 1)
            features['spread_skew'] = np.nan_to_num(stats.skew(spread), 0)
            features['spread_kurtosis'] = np.nan_to_num(stats.kurtosis(spread), 0)
            
            # Mean reversion metrics
            features['half_life'] = self.calculate_half_life(spread)
            features['hurst_exponent'] = self.calculate_hurst_exponent(spread)
            
            # Recent performance
            recent_spread = spread[-20:] if len(spread) >= 20 else spread
            if len(recent_spread) > 1:
                features['recent_crossings'] = np.sum(np.diff(np.sign(recent_spread - np.mean(spread))) != 0)
            else:
                features['recent_crossings'] = 0
            
            # Volatility ratio with zero division handling
            vol1 = series1.pct_change().std()
            vol2 = series2.pct_change().std()
            if vol2 > 0:
                features['vol_ratio'] = vol1 / vol2
            else:
                features['vol_ratio'] = 1.0
            
            # Market regime features
            market_returns = prices_df.pct_change().mean(axis=1)
            features['market_correlation'] = series1.pct_change().corr(market_returns)
            
            # Beta calculation with error handling
            try:
                returns1 = series1.pct_change().dropna()
                market_ret_clean = market_returns.dropna()
                if len(returns1) > 10 and market_ret_clean.var() > 0:
                    features['market_beta'] = np.cov(returns1, market_ret_clean)[0, 1] / market_ret_clean.var()
                else:
                    features['market_beta'] = 1.0
            except:
                features['market_beta'] = 1.0
            
            # Price level features
            if series2.iloc[-1] != 0:
                features['price_ratio'] = series1.iloc[-1] / series2.iloc[-1]
            else:
                features['price_ratio'] = 1.0
                
            price_ratios = series1 / series2.replace(0, np.nan)
            features['price_ratio_std'] = np.nan_to_num(price_ratios.std(), 0.1)
            
            # Replace any remaining inf/nan values
            for key in features:
                if np.isnan(features[key]) or np.isinf(features[key]):
                    features[key] = 0.0 if 'correlation' not in key else 0.5
                    
        except Exception as e:
            # Return default features if extraction fails
            print(f"Feature extraction error: {e}")
            return {
                'correlation': 0.5, 'correlation_60d': 0.5, 'correlation_20d': 0.5,
                'adf_pvalue': 0.5, 'adf_statistic': 0.0,
                'spread_mean': 0.0, 'spread_std': 1.0,
                'spread_skew': 0.0, 'spread_kurtosis': 0.0,
                'half_life': 60.0, 'hurst_exponent': 0.5,
                'recent_crossings': 0, 'vol_ratio': 1.0,
                'market_correlation': 0.0, 'market_beta': 1.0,
                'price_ratio': 1.0, 'price_ratio_std': 0.1
            }
        
        return features
    
    def generate_training_labels(self, series1, series2, forward_days=20):
        """Generate labels for training: future profitability of the pair"""
        X = series1.values.reshape(-1, 1)
        y = series2.values
        model = LinearRegression()
        model.fit(X, y)
        hedge_ratio = model.coef_[0]
        
        spread = y - hedge_ratio * series1.values
        spread_mean = np.mean(spread)
        spread_std = np.std(spread)
        
        # Calculate z-score
        zscore = (spread - spread_mean) / spread_std
        
        # Simulate simple trading over forward_days
        position = 0
        pnl = 0
        
        for i in range(len(zscore) - forward_days):
            # Entry logic
            if position == 0:
                if zscore[i] > self.zscore_entry:
                    position = -1  # Short spread
                elif zscore[i] < -self.zscore_entry:
                    position = 1   # Long spread
            
            # Exit logic
            elif position != 0:
                if abs(zscore[i]) < self.zscore_exit:
                    # Calculate PnL
                    exit_spread = spread[i]
                    entry_spread = spread[i - 1]  # Simplified
                    if position == 1:
                        pnl += (exit_spread - entry_spread) / spread_std
                    else:
                        pnl += (entry_spread - exit_spread) / spread_std
                    position = 0
        
        # Return normalized PnL as label
        return pnl / max(forward_days / 20, 1)  # Normalize by time
    
    def train_ml_model(self, data):
        """Train Random Forest model on historical pair features"""
        print("Training ML model for pair selection...")
        
        all_features = []
        all_labels = []
        pair_names = []
        
        symbols = data.columns.tolist()
        
        # Generate features and labels for all pairs
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                if len(data) > self.lookback_window + 20:  # Need enough data
                    # Use multiple time windows for training
                    for start_idx in range(self.lookback_window, len(data) - 40, 20):
                        end_idx = start_idx + 20
                        
                        window_data = data.iloc[start_idx-self.lookback_window:end_idx]
                        series1 = window_data[symbols[i]]
                        series2 = window_data[symbols[j]]
                        
                        # Extract features
                        features = self.extract_pair_features(series1, series2, window_data)
                        feature_array = np.array(list(features.values()))
                        
                        # Generate label (future profitability)
                        if end_idx + 20 <= len(data):
                            future_data = data.iloc[start_idx:end_idx+20]
                            label = self.generate_training_labels(
                                future_data[symbols[i]], 
                                future_data[symbols[j]]
                            )
                            
                            if not np.isnan(feature_array).any() and not np.isnan(label):
                                all_features.append(feature_array)
                                all_labels.append(label)
                                pair_names.append(f"{symbols[i]}-{symbols[j]}")
        
        if len(all_features) > 10:  # Need minimum samples
            # Convert to arrays
            X = np.array(all_features)
            y = np.array(all_labels)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Train model
            self.rf_model.fit(X_train, y_train)
            
            # Evaluate
            train_score = self.rf_model.score(X_train, y_train)
            test_score = self.rf_model.score(X_test, y_test)
            
            print(f"ML Model Training Complete:")
            print(f"  Train R²: {train_score:.3f}")
            print(f"  Test R²: {test_score:.3f}")
            
            # Feature importance
            feature_names = list(list(features.keys()))
            importance = self.rf_model.feature_importances_
            self.feature_importance = dict(zip(feature_names, importance))
            
            # Print top features
            print("\nTop 5 Most Important Features:")
            sorted_features = sorted(self.feature_importance.items(), 
                                   key=lambda x: x[1], reverse=True)
            for feat, imp in sorted_features[:5]:
                print(f"  {feat}: {imp:.3f}")
            
            return True
        
        print("Insufficient data for ML training")
        return False
    
    def rank_pairs_with_ml(self, data):
        """Use ML model to rank pairs by predicted profitability"""
        symbols = data.columns.tolist()
        pair_scores = []
        
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                series1 = data[symbols[i]]
                series2 = data[symbols[j]]
                
                # Extract features
                features = self.extract_pair_features(series1, series2, data)
                feature_array = np.array(list(features.values())).reshape(1, -1)
                
                if not np.isnan(feature_array).any():
                    # Scale features
                    feature_scaled = self.scaler.transform(feature_array)
                    
                    # Predict profitability
                    predicted_score = self.rf_model.predict(feature_scaled)[0]
                    
                    pair_scores.append({
                        'pair': (symbols[i], symbols[j]),
                        'ml_score': predicted_score,
                        'adf_pvalue': features['adf_pvalue'],
                        'correlation': features['correlation'],
                        'half_life': features['half_life']
                    })
        
        # Sort by ML score
        pair_scores.sort(key=lambda x: x['ml_score'], reverse=True)
        
        return pair_scores
    
    def find_pairs(self, data, use_ml=True):
        """Identify and rank cointegrated pairs using ML"""
        if use_ml and hasattr(self, 'rf_model'):
            # Train ML model on historical data
            self.train_ml_model(data)
            
            # Rank pairs using ML
            ranked_pairs = self.rank_pairs_with_ml(data)
            
            print(f"\nML-Ranked Pairs (Top 5):")
            print("-" * 60)
            
            self.pairs = []
            for i, pair_info in enumerate(ranked_pairs[:5]):
                pair = pair_info['pair']
                print(f"{i+1}. {pair[0]}-{pair[1]}:")
                print(f"   ML Score: {pair_info['ml_score']:.3f}")
                print(f"   P-value: {pair_info['adf_pvalue']:.3f}")
                print(f"   Correlation: {pair_info['correlation']:.3f}")
                print(f"   Half-life: {pair_info['half_life']:.1f} days")
                
                # Add pairs with positive ML score and reasonable p-value
                if pair_info['ml_score'] > 0 and pair_info['adf_pvalue'] < 0.1:
                    # Get hedge ratio for trading
                    series1 = data[pair[0]]
                    series2 = data[pair[1]]
                    X = series1.values.reshape(-1, 1)
                    y = series2.values
                    model = LinearRegression()
                    model.fit(X, y)
                    hedge_ratio = model.coef_[0]
                    
                    self.pairs.append({
                        'pair': pair,
                        'hedge_ratio': hedge_ratio,
                        'ml_score': pair_info['ml_score']
                    })
        else:
            # Fallback to traditional method
            print("Using traditional cointegration method...")
            super().find_pairs(data)
        
        print(f"\nSelected {len(self.pairs)} pairs for trading")
        
    def calculate_signals(self, data):
        """Generate trading signals for selected pairs"""
        for pair_info in self.pairs:
            pair = pair_info['pair']
            hedge_ratio = pair_info['hedge_ratio']
            
            series1 = data[pair[0]]
            series2 = data[pair[1]]
            
            # Calculate spread
            spread = series2 - hedge_ratio * series1
            
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
            signals.loc[abs(zscore) < self.zscore_exit, 'position'] = 0
            
            # Store signals
            self.signals[f"{pair[0]}-{pair[1]}"] = signals
    
    def backtest_pair(self, data, pair_name, signals, hedge_ratio):
        """Backtest a single pair with transaction costs"""
        pair = pair_name.split('-')
        series1 = data[pair[0]]
        series2 = data[pair[1]]
        
        # Initialize portfolio
        positions = pd.DataFrame(index=data.index)
        positions['position'] = signals['position'].fillna(0)
        
        # Calculate when we trade
        positions['trade'] = positions['position'].diff().fillna(0)
        
        # Track holding period
        holding_period = 0
        for i in range(1, len(positions)):
            if positions['position'].iloc[i] != 0:
                if positions['position'].iloc[i] == positions['position'].iloc[i-1]:
                    holding_period += 1
                    if holding_period > self.max_holding_period:
                        positions['position'].iloc[i] = 0
                        positions['trade'].iloc[i] = -positions['position'].iloc[i-1]
                        holding_period = 0
                else:
                    holding_period = 1
            else:
                holding_period = 0
        
        # Calculate returns
        returns1 = series1.pct_change()
        returns2 = series2.pct_change()
        
        # Portfolio returns (dollar neutral)
        # When position = 1: long spread (long asset2, short asset1)
        # When position = -1: short spread (short asset2, long asset1)
        portfolio_returns = positions['position'].shift(1) * (returns2 - hedge_ratio * returns1)
        
        # Transaction costs (5 bps each way)
        transaction_costs = abs(positions['trade']) * 0.0005
        
        # Net returns
        net_returns = portfolio_returns - transaction_costs
        
        # Calculate cumulative returns
        cumulative_returns = (1 + net_returns).cumprod()
        
        # Performance metrics
        total_return = cumulative_returns.iloc[-1] - 1
        sharpe_ratio = np.sqrt(252) * net_returns.mean() / net_returns.std() if net_returns.std() > 0 else 0
        max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
        win_rate = (net_returns[net_returns != 0] > 0).mean() if len(net_returns[net_returns != 0]) > 0 else 0
        
        return {
            'cumulative_returns': cumulative_returns,
            'net_returns': net_returns,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': (positions['trade'] != 0).sum() // 2
        }
    
    def run_full_backtest(self, data):
        """Run backtest on all selected pairs"""
        results = {}
        
        for pair_info in self.pairs:
            pair = pair_info['pair']
            pair_name = f"{pair[0]}-{pair[1]}"
            
            if pair_name in self.signals:
                results[pair_name] = self.backtest_pair(
                    data, 
                    pair_name, 
                    self.signals[pair_name],
                    pair_info['hedge_ratio']
                )
                
                # Add ML score to results
                results[pair_name]['ml_score'] = pair_info['ml_score']
        
        return results
    
    def plot_results(self, data, results):
        """Visualize backtest results with ML insights"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('ML-Enhanced Pairs Trading Results', fontsize=16)
        
        # Plot 1: Cumulative returns by pair
        ax1 = axes[0, 0]
        for pair_name, result in results.items():
            ax1.plot(result['cumulative_returns'], label=f"{pair_name} (ML: {result['ml_score']:.2f})")
        ax1.set_title('Cumulative Returns by Pair')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Feature importance
        ax2 = axes[0, 1]
        if self.feature_importance:
            features = list(self.feature_importance.keys())[:10]  # Top 10
            importances = [self.feature_importance[f] for f in features]
            y_pos = np.arange(len(features))
            ax2.barh(y_pos, importances)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(features)
            ax2.set_xlabel('Importance')
            ax2.set_title('ML Feature Importance (Top 10)')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: ML Score vs Actual Performance
        ax3 = axes[1, 0]
        ml_scores = [r['ml_score'] for r in results.values()]
        actual_returns = [r['total_return'] for r in results.values()]
        ax3.scatter(ml_scores, actual_returns, s=100, alpha=0.6)
        ax3.set_xlabel('ML Predicted Score')
        ax3.set_ylabel('Actual Return')
        ax3.set_title('ML Predictions vs Actual Performance')
        ax3.grid(True, alpha=0.3)
        
        # Add trend line
        if len(ml_scores) > 1:
            z = np.polyfit(ml_scores, actual_returns, 1)
            p = np.poly1d(z)
            ax3.plot(sorted(ml_scores), p(sorted(ml_scores)), "r--", alpha=0.8)
        
        # Plot 4: Performance metrics comparison
        ax4 = axes[1, 1]
        metrics = ['Sharpe Ratio', 'Total Return', 'Win Rate']
        x = np.arange(len(results))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            if metric == 'Sharpe Ratio':
                values = [r['sharpe_ratio'] for r in results.values()]
            elif metric == 'Total Return':
                values = [r['total_return'] * 100 for r in results.values()]
            else:  # Win Rate
                values = [r['win_rate'] * 100 for r in results.values()]
            
            ax4.bar(x + i * width, values, width, label=metric)
        
        ax4.set_xlabel('Pair')
        ax4.set_ylabel('Value')
        ax4.set_title('Performance Metrics by Pair')
        ax4.set_xticks(x + width)
        ax4.set_xticklabels(results.keys(), rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    """Demonstrate ML-enhanced pairs trading system"""
    print("=" * 60)
    print("ML-ENHANCED PAIRS TRADING SYSTEM")
    print("=" * 60)
    
    # Initialize system
    system = MLEnhancedPairsTradingSystem(
        lookback_window=60,
        zscore_entry=2.0,
        zscore_exit=0.5,
        max_holding_period=20,
        position_size=0.1
    )
    
    # Generate data
    print("\nGenerating synthetic market data...")
    data = system.generate_synthetic_data(n_days=500)
    print(f"Created {len(data)} days of data for {len(data.columns)} assets")
    
    # Split data for training and testing
    train_data = data.iloc[:400]  # 80% for training
    test_data = data.iloc[300:]   # Last 40% for testing (with overlap for lookback)
    
    # Find pairs using ML
    print("\nFinding pairs with ML ranking...")
    system.find_pairs(train_data, use_ml=True)
    
    # Generate signals on test data
    print("\nGenerating trading signals...")
    system.calculate_signals(test_data)
    
    # Run backtest
    print("\nRunning backtest with transaction costs...")
    results = system.run_full_backtest(test_data)
    
    # Display results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS (ML-SELECTED PAIRS)")
    print("=" * 60)
    
    for pair_name, result in sorted(results.items(), 
                                   key=lambda x: x[1]['sharpe_ratio'], 
                                   reverse=True):
        print(f"\n{pair_name}:")
        print(f"  ML Score: {result['ml_score']:.3f}")
        print(f"  Total Return: {result['total_return']*100:.1f}%")
        print(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {result['max_drawdown']*100:.1f}%")
        print(f"  Win Rate: {result['win_rate']*100:.1f}%")
        print(f"  Number of Trades: {result['num_trades']}")
    
    # Plot results
    system.plot_results(test_data, results)
    
    # Risk warnings
    print("\n" + "=" * 60)
    print("IMPORTANT NOTES:")
    print("=" * 60)
    print("1. This is an educational demonstration using synthetic data")
    print("2. The ML model is trained on limited synthetic patterns")
    print("3. Real markets have additional complexities not modeled here")
    print("4. Always validate ML predictions with domain knowledge")
    print("5. Past performance does not guarantee future results")

if __name__ == "__main__":
    main()