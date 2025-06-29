# Pairs Trading - Statistical Arbitrage Strategy with ML Enhancement

A comprehensive Python implementation of a pairs trading strategy enhanced with Machine Learning for quantitative finance education. This project demonstrates the integration of traditional statistical arbitrage with modern ML techniques using Random Forest for pair selection and ranking.

## Overview

This codebase implements a complete pairs trading system that identifies and trades mean-reverting relationships between cointegrated asset pairs. The strategy is based on the principle that certain asset pairs maintain stable long-term relationships, and deviations from these relationships present profitable trading opportunities.

## 🚀 New: Machine Learning Enhancement

The ML-enhanced version (`pairs-trading-ml.py`) adds:

### ML-Powered Pair Selection
- **Random Forest** model to predict pair profitability
- **Feature Engineering** with 15+ indicators per pair
- **Automated ranking** of pairs by expected performance
- **Feature importance** analysis for interpretability

### Advanced Features Extracted
- Correlation metrics (multiple timeframes)
- Mean reversion indicators (half-life, Hurst exponent)
- Market regime features
- Spread characteristics (skew, kurtosis)
- Recent performance metrics

## Key Concepts Covered

### Statistical & Mathematical Concepts

1. **Cointegration Analysis**
   - Engle-Granger two-step method
   - Linear regression for hedge ratio calculation
   - Augmented Dickey-Fuller (ADF) test for stationarity testing

2. **Statistical Measures**
   - Z-score normalization for spread analysis
   - Rolling mean and standard deviation calculations
   - Correlation analysis between asset pairs

3. **Risk Management**
   - Position sizing with dollar-neutral portfolios
   - Maximum holding period constraints
   - Transaction cost modeling (5 basis points)

4. **Performance Analytics**
   - Sharpe Ratio calculation
   - Maximum drawdown analysis
   - Win rate and profit factor metrics
   - Cumulative returns tracking

### Trading Strategy Components

- **Entry Signals**: Positions opened when z-score exceeds ±2.0 standard deviations
- **Exit Signals**: Positions closed when z-score returns to ±0.5 or after 20 days
- **Position Sizing**: 10% of capital per position (configurable)
- **Portfolio Construction**: Dollar-neutral long/short positions

## Getting Started

### Prerequisites

Ensure you have Python 3.7+ installed. Install required dependencies:

```bash
pip install numpy pandas scipy scikit-learn matplotlib statsmodels
```

For the ML-enhanced version, you already have all required dependencies!

### Running the Code

1. Clone or download the repository:
```bash
git clone https://github.com/yourusername/pairs-trading.git
cd pairs-trading
```

2. Run the basic version:
```bash
python pairs-trading.py
```

Or run the ML-enhanced version:
```bash
python pairs-trading-ml.py
```

The script will:
- Generate synthetic correlated asset data
- Identify cointegrated pairs (ML version: rank by predicted profitability)
- Generate trading signals
- Execute backtests with transaction costs
- Display performance metrics and visualizations
- (ML version) Show feature importance and ML model performance

### Expected Output

The program generates:
1. **Console Output**: 
   - Cointegrated pairs found
   - Performance metrics for each pair
   - Risk analysis summary

2. **Visualization**: A 4-panel plot showing:
   - Cumulative returns by pair
   - Z-score evolution for best pair
   - Performance metrics comparison
   - Drawdown analysis

## 📁 Code Structure

### Basic Version (pairs-trading.py)
```
├── StatisticalArbitrageSystem (Main Class)
│   ├── generate_synthetic_data()    # Creates correlated price series
│   ├── test_cointegration()         # Tests for cointegration
│   ├── find_pairs()                 # Identifies tradeable pairs
│   ├── calculate_signals()          # Generates buy/sell signals
│   ├── backtest_pair()              # Simulates trading
│   ├── run_full_backtest()          # Orchestrates full backtest
│   └── plot_results()               # Creates visualizations
└── Main execution block
```

### ML-Enhanced Version (pairs-trading-ml.py)
```
├── MLEnhancedPairsTradingSystem (Main Class)
│   ├── All methods from basic version PLUS:
│   ├── extract_pair_features()      # Engineer 15+ ML features
│   ├── train_ml_model()             # Train Random Forest
│   ├── rank_pairs_with_ml()         # ML-based pair ranking
│   ├── calculate_half_life()        # Mean reversion speed
│   ├── calculate_hurst_exponent()   # Trend vs mean reversion
│   └── Enhanced plot_results()      # ML insights visualization
└── Main execution block
```

## Customization

Key parameters can be adjusted in the `StatisticalArbitrageSystem` initialization:

```python
system = StatisticalArbitrageSystem(
    lookback_window=60,        # Days for rolling statistics
    zscore_entry=2.0,          # Entry threshold
    zscore_exit=0.5,           # Exit threshold
    max_holding_period=20,     # Maximum days to hold position
    position_size=0.1          # 10% of capital per position
)
```

## Understanding the Results

### Performance Metrics
- **Sharpe Ratio**: Risk-adjusted returns (>1.0 is good, >2.0 is excellent)
- **Total Return**: Cumulative percentage gain/loss
- **Win Rate**: Percentage of profitable trades
- **Max Drawdown**: Largest peak-to-trough decline

### Risk Considerations
The code includes warnings about:
- Model risk (relationship breakdown)
- Execution risk (slippage, market impact)
- Market risk (systematic events)

## Educational Value

This implementation is ideal for:
- Understanding statistical arbitrage concepts
- Learning quantitative trading fundamentals
- Exploring time series analysis in finance
- Studying risk management techniques
- Building foundation for more complex strategies

## Extending the Code

Consider these enhancements:
1. Use real market data (e.g., via yfinance)
2. Implement more sophisticated cointegration tests (Johansen test)
3. ✅ **DONE**: Added machine learning for pair selection (see pairs-trading-ml.py)
4. Include more risk metrics (VaR, CVaR)
5. Add real-time trading capabilities
6. Implement portfolio-level risk management
7. Add deep learning models (LSTM for spread prediction)
8. Implement reinforcement learning for dynamic thresholds
9. Add sentiment analysis from news data

## Disclaimer

This code is for educational purposes only. Trading involves significant risk of loss. Always thoroughly backtest and validate any strategy before using real capital.

## Further Reading

- Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*
- Engle, R. F., & Granger, C. W. (1987). "Co-integration and error correction"
- Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). "Pairs trading: Performance of a relative-value arbitrage rule"

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is open source and available under the MIT License.