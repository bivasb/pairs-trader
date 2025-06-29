# Pairs Trading - Statistical Arbitrage Strategy

A comprehensive Python implementation of a pairs trading strategy for quantitative finance education. This project demonstrates key concepts in statistical arbitrage, cointegration testing, and systematic trading.

## Overview

This codebase implements a complete pairs trading system that identifies and trades mean-reverting relationships between cointegrated asset pairs. The strategy is based on the principle that certain asset pairs maintain stable long-term relationships, and deviations from these relationships present profitable trading opportunities.

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

### Running the Code

1. Clone or download the repository:
```bash
git clone https://github.com/yourusername/pairs-trading.git
cd pairs-trading
```

2. Run the main script:
```bash
python pairs-trading.py
```

The script will:
- Generate synthetic correlated asset data
- Identify cointegrated pairs
- Generate trading signals
- Execute backtests with transaction costs
- Display performance metrics and visualizations

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

```
pairs-trading.py
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
1. Use real market data (e.g., via barchart)
2. Implement more sophisticated cointegration tests
3. Add machine learning for pair selection
4. Include more risk metrics (VaR, CVaR)
5. Add real-time trading capabilities
6. Implement portfolio-level risk management

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