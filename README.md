# Heikin-Ashi Weighted Trend-Following Strategy

A sophisticated backtesting framework for a weighted Heikin-Ashi trend-following trading strategy that combines Heikin-Ashi candle analysis with RSI-based momentum signals. The strategy uses optimized parameters to identify entry and exit points in financial markets.

## ⚠️ Disclaimer

This software is for educational and research purposes only. It is not financial advice, and trading involves significant risk of loss. Past performance does not guarantee future results. Always conduct your own research and consider consulting with a qualified financial advisor before making trading decisions.

## Features

- **Heikin-Ashi Candle Analysis**: Uses smoothed Heikin-Ashi candles for trend identification
- **Weighted Scoring System**: Recency-weighted analysis of recent candle patterns
- **Momentum Tracking**: Bonus/penalty system for accelerating/decelerating trends
- **RSI Integration**: Combines Heikin-Ashi signals with RSI-based entry/exit confirmations
- **Doji Pattern Recognition**: Enhanced signals from doji patterns at specific lookback positions
- **SAMBO Optimization**: Advanced parameter optimization using Successive Halving with Bayesian Optimization
- **Comprehensive Backtesting**: Built on the Backtesting.py framework with detailed statistics
- **Visualization**: Generates interactive plots and heatmaps of optimization results
- **Numba Optimization**: High-performance calculations using JIT compilation

## Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`:
  - numpy
  - pandas
  - pandas-ta
  - psutil
  - numba
  - backtesting
  - sambo >= 1.25.2

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Backtesting

Run the strategy on a CSV file containing OHLCV data:

```bash
python heikinAshi.py path/to/your/data.csv
```

### Supported CSV Formats

The script automatically detects CSV format based on column names:

- **IVV.csv format**: Columns include `Date`, `Open`, `High`, `Low`, `Close`, `Volume`
- **IVV2.csv format**: Columns include `time`, `open`, `high`, `low`, `close`, `volume`

### Example Output

The script generates:
- Backtest statistics and performance metrics
- Interactive HTML plot of trades
- Optimization heatmap showing parameter performance
- Top 100 parameter combinations (saved as CSV)

## Strategy Details

### Core Components

1. **Heikin-Ashi Candles**: Smoothed price representation that filters noise and highlights trends
2. **Weighted Bull/Bear Systems**: Separate scoring systems for entry (bullish) and exit (bearish) signals
3. **Momentum Tracking**: Analyzes acceleration/deceleration in recent candles
4. **RSI Confirmation**: Uses RSI levels to confirm oversold/overbought conditions

### Entry Logic

- Analyzes the last 4 Heikin-Ashi candles with recency weighting
- Applies momentum bonuses for accelerating bullish trends
- Checks for doji patterns at position -6 for confirmation
- Combines with RSI oversold signals
- Triggers entry when combined score ≥ 1.0

### Exit Logic

- Similar analysis but focused on bearish patterns
- RSI overbought confirmation
- Triggers exit when combined score ≥ 1.0

### Risk Management

- ATR-based stop losses (default 1.5x ATR)
- Position sizing based on available capital

## Parameters

### Optimization Ranges

The strategy optimizes the following parameters using SAMBO:

- **ATR Period**: 8-20 (volatility lookback)
- **Bull Weights**: 0.15-0.50 (entry signal weights for 4 recent candles)
- **Bear Weights**: 0.10-0.40 (exit signal weights)
- **Doji Weights**: 0.30-0.50 (doji pattern confirmation)
- **Momentum Bonuses**: 0.00-0.15 (acceleration rewards)
- **Momentum Penalties**: 0.00-0.15 (deceleration penalties)
- **Doji Body Fraction**: 0.10-0.30 (doji size threshold as % of ATR)
- **Prior Index**: -8 to -4 (doji lookback position)
- **Stop ATR Multiplier**: Fixed at 1.5 (risk management)
- **RSI Parameters**: Period 10-80, thresholds 20-80, weights 0.00-0.25

### Default Configuration

For non-optimized runs, uses predefined parameter values (see commented section in code).

## Output Files

- `HeikinAshi_weighted_{date}.html`: Interactive trade plot
- `HeikinAshi_weighted_heatmap_{date}.html`: Parameter optimization heatmap
- Top 100 parameter sets printed to console (can be saved as CSV)

## Performance Metrics

The backtest provides comprehensive statistics including:
- Total return percentage
- Sharpe ratio
- Maximum drawdown
- Win rate
- Profit factor
- Trade statistics (total trades, winning/losing trades)
- Risk-adjusted metrics

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Technical Notes

- Uses Numba JIT compilation for performance-critical calculations
- Implements multiprocessing for optimization on Unix systems
- Automatically adjusts process priority for background execution
- Supports large-scale parameter optimization (200,000+ evaluations)
