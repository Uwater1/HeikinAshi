# HeikinAshi_Weighted.py
# Weighted Bull/Bear implementation with separate entry/exit systems
# Usage: python HeikinAshi_weighted.py {data}.csv

import numpy as np
import pandas as pd
import pandas_ta as ta
import sys
import os
import psutil
import logging
import multiprocessing as mp
from datetime import date
from numba import jit
from backtesting import Backtest, Strategy
from backtesting.lib import plot_heatmaps


# Silence tqdm progress bars emitted by Backtest.run / optimize
os.environ.setdefault("TQDM_DISABLE", "1")

# Limit third-party logging noise to errors only
logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")

if os.name == 'posix' and __name__ == '__main__':
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass  # Already set

# ========================================
# Numba-Optimized Functions
# ========================================

@jit(nopython=True, cache=True)
def _compute_tr_numba(high, low, close):
    """Numba-optimized true range calculation."""
    n = len(close)
    tr = np.empty(n)
    for i in range(n):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            tr[i] = max(high[i] - low[i],
                        abs(high[i] - close[i-1]),
                        abs(low[i] - close[i-1]))
    return tr

@jit(nopython=True, cache=True)
def _compute_heikin_ashi_numba(o, h, l, c):
    """Numba-optimized Heikin-Ashi calculation."""
    n = len(c)
    ha_open = np.empty(n)
    ha_close = np.empty(n)
    ha_high = np.empty(n)
    ha_low = np.empty(n)

    for i in range(n):
        ha_c = (o[i] + h[i] + l[i] + c[i]) / 4.0
        if i == 0:
            ha_o = (o[i] + c[i]) / 2.0
        else:
            ha_o = (ha_open[i-1] + ha_close[i-1]) / 2.0

        ha_h = max(h[i], ha_o, ha_c)
        ha_l = min(l[i], ha_o, ha_c)

        ha_open[i] = ha_o
        ha_close[i] = ha_c
        ha_high[i] = ha_h
        ha_low[i] = ha_l

    return ha_open, ha_high, ha_low, ha_close

@jit(nopython=True, cache=True)
def _compute_score_numba(ha_open, ha_close, atr_cur, weights, doji_weight, doji_body_frac, prior_idx,
                        weight_bull_bonus=0.1, weight_bear_bonus=0.1,
                        weight_bull_penalty=0.05, weight_bear_penalty=0.05, is_entry=True):
    """Numba-optimized score calculation with momentum tracking for bars 3 and 4.
    Calculates weighted trading score based on:
    - Recent 4 HA candles (body size normalized by ATR, weighted by recency)
    - Doji patterns at position -6 (bonus for bullish/bearish confirmation)
    - Momentum tracking: bonus/penalty for bars 3 and 4 based on size comparison
    Returns composite score used for entry/exit decisions when exceeding thresholds
    """
    score = 0.0
    lookback = 4

    # Pre-allocate arrays
    bar_sizes = np.empty(lookback, dtype=np.float32)

    # Find selected bars: skip reds before first desired, then include all remaining
    selected = []
    started = False
    for i in range(lookback):
        idx = -(i + 1)
        is_desired = (is_entry and ha_close[idx] > ha_open[idx]) or (not is_entry and ha_close[idx] < ha_open[idx])
        if is_desired:
            started = True
            selected.append(i)
        elif not started:
            continue  # skip initial non-desired
        else:
            selected.append(i)  # include non-desired after starting

    # Compute signed norms and bar sizes only for selected bars
    for i in selected:
        idx = -(i + 1)
        ha_o = ha_open[idx]
        ha_c = ha_close[idx]
        if is_entry:
            body = ha_c - ha_o  # positive for green, negative for red
        else:
            body = ha_o - ha_c  # positive for red, negative for green
        bar_size = body / atr_cur
        bar_sizes[i] = bar_size + 0.01 if bar_size > 0 else 0

    # Score only for selected bars
    for j, i in enumerate(selected):
        score += weights[j] * bar_sizes[i]

    len_selected = len(selected)

    # Momentum tracking for the 3rd and 4th selected bars
    if len_selected >= 3:
        # For 3rd selected bar
        current_idx = selected[2]
        prev_indices = selected[0:2]
        max_prev_size = max(bar_sizes[prev_indices[0]], bar_sizes[prev_indices[1]])
        min_prev_size = min(bar_sizes[prev_indices[0]], bar_sizes[prev_indices[1]])
        current_size = bar_sizes[current_idx]

        bonus_weight = weight_bull_bonus if is_entry else weight_bear_bonus
        penalty_weight = weight_bull_penalty if is_entry else weight_bear_penalty

        if current_size > max_prev_size:
            bonus = (current_size - max_prev_size) * bonus_weight
            score += bonus
        elif current_size < min_prev_size:
            penalty = (min_prev_size - current_size) * penalty_weight
            score -= penalty

    if len_selected >= 4:
        # For 4th selected bar
        current_idx = selected[3]
        prev_indices = selected[0:3]
        max_prev_size = max(bar_sizes[prev_indices[0]], bar_sizes[prev_indices[1]], bar_sizes[prev_indices[2]])
        min_prev_size = min(bar_sizes[prev_indices[0]], bar_sizes[prev_indices[1]], bar_sizes[prev_indices[2]])
        current_size = bar_sizes[current_idx]

        if current_size > max_prev_size:
            bonus = (current_size - max_prev_size) * bonus_weight
            score += bonus
        elif current_size < min_prev_size:
            penalty = (min_prev_size - current_size) * penalty_weight
            score -= penalty

    # Doji bonus logic 
    try:
        prior_body = abs(ha_close[prior_idx] - ha_open[prior_idx])
        if prior_body < doji_body_frac * atr_cur:
            score += doji_weight
        else:
            # Check fallback positions based on len_selected
            max_offset = 3 - len_selected
            for offset in range(1, max_offset + 1):
                prior_body = abs(ha_close[prior_idx + offset] - ha_open[prior_idx + offset])
                if prior_body < doji_body_frac * atr_cur:
                    score += doji_weight
                    break
    except:
        pass

    return score

# ========================================
# Data Loading
# ========================================

def load_csv(path):
    """Load CSV data and prepare for backtesting."""
    # Try to detect the CSV format by checking column names
    try:
        # First try reading with Date column (IVV.csv format)
        df = pd.read_csv(path, nrows=5)  # Read just header to detect format
        if "Date" in df.columns:
            # IVV.csv format
            df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
            df = df.rename(columns={"Close/Last": "Close"})
        elif "time" in df.columns:
            # IVV2.csv format
            df = pd.read_csv(path, parse_dates=["time"], index_col="time")
            df = df.rename(columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close"
            })
        else:
            raise ValueError("Unknown CSV format - neither 'Date' nor 'time' column found")
    except Exception as e:
        raise ValueError(f"Failed to load CSV: {e}")

    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.sort_index()
    df = df.dropna()
    return df

# ========================================
# Indicator Functions
# ========================================

def indicator_rsi(close, length=14):
    """Compute RSI using pandas-ta."""
    close_series = pd.Series(close)
    rsi = ta.rsi(close_series, length=length)
    return rsi.to_numpy()

def indicator_atr(high, low, close, length=14):
    """ATR with Numba optimization."""
    high = np.asarray(high, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    close = np.asarray(close, dtype=np.float32)

    tr = _compute_tr_numba(high, low, close)
    atr = pd.Series(tr).rolling(window=length, min_periods=1).mean().to_numpy()
    return atr

def compute_heikin_ashi(df):
    """Compute Heikin-Ashi candles and add to dataframe."""
    try:
        ha = ta.ha(df['Open'], df['High'], df['Low'], df['Close'])
        df['HA_open'] = ha.iloc[:, 0].values
        df['HA_high'] = ha.iloc[:, 1].values
        df['HA_low'] = ha.iloc[:, 2].values
        df['HA_close'] = ha.iloc[:, 3].values
    except Exception:
        # Fallback with Numba optimization
        ha_open, ha_high, ha_low, ha_close = _compute_heikin_ashi_numba(
            df['Open'].to_numpy(),
            df['High'].to_numpy(),
            df['Low'].to_numpy(),
            df['Close'].to_numpy()
        )
        df['HA_open'] = ha_open
        df['HA_high'] = ha_high
        df['HA_low'] = ha_low
        df['HA_close'] = ha_close
    return df

# ========================================
# Strategy
# ========================================

class HeikinAshiWeightedStrategy(Strategy):
    """Weighted Heikin-Ashi trend-following strategy with separate bull/bear systems."""

    # ATR period
    atr_period = 14

    # Bull weights for entry (scaled by 0.25 / 0.8 as per user requirement)
    weight_bull_1 = 0.3
    weight_bull_2 = 0.25
    weight_bull_3 = 0.3
    weight_bull_4 = 0.35
    weight_bull_doji = 0.4

    # Bear weights for exit (scaled by 0.25 * 0.8 as per user requirement)
    weight_bear_1 = 0.25
    weight_bear_2 = 0.20
    weight_bear_3 = 0.25
    weight_bear_4 = 0.30
    weight_bear_doji = 0.3

    # Stop in ATR multiples
    stop_atr_mult = 1.5

    # Doji threshold (fraction of ATR)
    doji_body_frac = 0.20

    # Prior index for doji pattern detection
    prior_idx = -6

    # Momentum tracking parameters
    weight_bull_bonus = 0.1    # Bonus weight for accelerating bullish momentum
    weight_bear_bonus = 0.1     # Bonus weight for accelerating bearish momentum
    weight_bull_penalty = 0.05 # Penalty weight for decelerating bullish momentum
    weight_bear_penalty = 0.05 # Penalty weight for decelerating bearish momentum

    # RSI parameters
    rsi_period = 14                    # RSI calculation window
    rsi_lookback = 8                   # Bars to look back for min/max
    rsi_bull_threshold = 30            # Oversold threshold for entry
    rsi_bear_threshold = 70            # Overbought threshold for exit
    weight_rsi_bull = 15               # Weight for RSI entry signal (scaled by 100)
    weight_rsi_bear = 15               # Weight for RSI exit signal (scaled by 100)

    def init(self):
        # Scale weights from integers (hundredths) to floats
        self.weight_bull_1 = self.weight_bull_1 / 100.0
        self.weight_bull_2 = self.weight_bull_2 / 100.0
        self.weight_bull_3 = self.weight_bull_3 / 100.0
        self.weight_bull_4 = self.weight_bull_4 / 100.0
        self.weight_bull_doji = self.weight_bull_doji / 100.0
        self.weight_bear_1 = self.weight_bear_1 / 100.0
        self.weight_bear_2 = self.weight_bear_2 / 100.0
        self.weight_bear_3 = self.weight_bear_3 / 100.0
        self.weight_bear_4 = self.weight_bear_4 / 100.0
        self.weight_bear_doji = self.weight_bear_doji / 100.0
        self.weight_bull_bonus = self.weight_bull_bonus / 100.0
        self.weight_bear_bonus = self.weight_bear_bonus / 100.0
        self.weight_bull_penalty = self.weight_bull_penalty / 100.0
        self.weight_bear_penalty = self.weight_bear_penalty / 100.0
        self.doji_body_frac = self.doji_body_frac / 100.0
        self.stop_atr_mult = self.stop_atr_mult / 100.0

        # Scale RSI weights from integers (hundredths) to floats
        self.weight_rsi_bull = self.weight_rsi_bull / 100.0
        self.weight_rsi_bear = self.weight_rsi_bear / 100.0

        # Pre-compute weight arrays for performance
        self.bull_weights_arr = np.array([self.weight_bull_1, self.weight_bull_2, self.weight_bull_3, self.weight_bull_4], dtype=np.float32)
        self.bear_weights_arr = np.array([self.weight_bear_1, self.weight_bear_2, self.weight_bear_3, self.weight_bear_4], dtype=np.float32)

        # Register ATR indicator
        self.atr = self.I(lambda: indicator_atr(
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_period
        ))

        # Register RSI indicator
        self.rsi = self.I(lambda: indicator_rsi(
            self.data.Close,
            self.rsi_period
        ))

        # Register HA columns as indicators
        self.ha_open = self.I(lambda: np.asarray(self.data.HA_open, dtype=np.float32))
        self.ha_close = self.I(lambda: np.asarray(self.data.HA_close, dtype=np.float32))
        self.ha_high = self.I(lambda: np.asarray(self.data.HA_high, dtype=np.float32))
        self.ha_low = self.I(lambda: np.asarray(self.data.HA_low, dtype=np.float32))

    def _is_green(self, idx):
        """Check if candle at idx is green."""
        return float(self.ha_close[idx]) > float(self.ha_open[idx])

    def _is_red(self, idx):
        """Check if candle at idx is red."""
        return float(self.ha_close[idx]) < float(self.ha_open[idx])

    def _ha_body(self, idx):
        """Get HA body size at idx."""
        return float(self.ha_close[idx]) - float(self.ha_open[idx])

    def _get_atr_cur(self, price):
        """Get current ATR value with fallback."""
        atr_cur = float(self.atr[-1] if len(self.atr) > 0 else price * 0.01)
        return max(atr_cur, price * 0.01)

    def compute_entry_score(self, price):
        """Compute weighted entry score using bull weights."""
        atr_cur = self._get_atr_cur(price)
        bull_weights = self.bull_weights_arr
        return float(_compute_score_numba(
            self.ha_open, self.ha_close, np.float32(atr_cur), bull_weights,
            np.float32(self.weight_bull_doji), np.float32(self.doji_body_frac), self.prior_idx,
            np.float32(self.weight_bull_bonus), np.float32(self.weight_bear_bonus),
            np.float32(self.weight_bull_penalty), np.float32(self.weight_bear_penalty),
            is_entry=True
        ))

    def compute_exit_score(self, price):
        """Compute weighted exit score using bear weights."""
        atr_cur = self._get_atr_cur(price)
        bear_weights = self.bear_weights_arr
        return float(_compute_score_numba(
            self.ha_open, self.ha_close, np.float32(atr_cur), bear_weights,
            np.float32(self.weight_bear_doji), np.float32(self.doji_body_frac), self.prior_idx,
            np.float32(self.weight_bull_bonus), np.float32(self.weight_bear_bonus),
            np.float32(self.weight_bull_penalty), np.float32(self.weight_bear_penalty),
            is_entry=False
        ))

    def compute_rsi_entry_score(self):
        """Compute RSI-based entry score (oversold signal).
        
        Looks back rsi_lookback bars to find minimum RSI.
        If min RSI is below threshold, returns weighted distance as score.
        More oversold = higher score contribution.
        """
        if len(self.rsi) < self.rsi_lookback + 1:
            return 0.0
        
        # Get last rsi_lookback bars of RSI (exclude current bar)
        lookback_window = self.rsi[-(self.rsi_lookback + 1):-1]
        
        # Handle NaN values
        valid_rsi = lookback_window[~np.isnan(lookback_window)]
        if len(valid_rsi) == 0:
            return 0.0
        
        min_rsi = np.min(valid_rsi)
        
        # Distance below threshold = bullish signal
        # More oversold = higher score
        distance = self.rsi_bull_threshold - min_rsi
        
        if distance > 0:
            return distance * self.weight_rsi_bull
        return 0.0

    def compute_rsi_exit_score(self):
        """Compute RSI-based exit score (overbought signal).
        
        Looks back rsi_lookback bars to find maximum RSI.
        If max RSI is above threshold, returns weighted distance as score.
        More overbought = higher score contribution.
        """
        if len(self.rsi) < self.rsi_lookback + 1:
            return 0.0
        
        # Get last rsi_lookback bars of RSI (exclude current bar)
        lookback_window = self.rsi[-(self.rsi_lookback + 1):-1]
        
        # Handle NaN values
        valid_rsi = lookback_window[~np.isnan(lookback_window)]
        if len(valid_rsi) == 0:
            return 0.0
        
        max_rsi = np.max(valid_rsi)
        
        # Distance above threshold = bearish signal
        # More overbought = higher score
        distance = max_rsi - self.rsi_bear_threshold
        
        if distance > 0:
            return distance * self.weight_rsi_bear
        return 0.0

    def next(self):
        """Execute strategy logic on each bar."""
        price = float(self.data.Close[-1])
        atr_cur = float(self.atr[-1] if len(self.atr) > 0 else 0.0)

        # Entry logic - combine HA and RSI scores, trigger when total >= 1.0
        if not self.position:
            ha_entry_score = self.compute_entry_score(price)
            rsi_entry_score = self.compute_rsi_entry_score()
            total_entry_score = ha_entry_score + rsi_entry_score
            
            if total_entry_score >= 1.0:  # Fixed threshold for entry
                sl_price = max(price * 0.95, price - self.stop_atr_mult * atr_cur) if atr_cur > 0 else price * 0.97
                try:
                    self.buy(sl=sl_price)
                except Exception:
                    self.buy()

        # Exit logic - combine HA and RSI scores, trigger when total >= 1.0
        else:
            ha_exit_score = self.compute_exit_score(price)
            rsi_exit_score = self.compute_rsi_exit_score()
            total_exit_score = ha_exit_score + rsi_exit_score
            
            if total_exit_score >= 1.0:  # Fixed threshold for exit
                try:
                    self.position.close()
                except Exception:
                    try:
                        self.sell()
                    except Exception:
                        pass

# ========================================
# SAMBO Optimization Analysis
# ========================================

def analyze_sambo_results(optimize_result):
    """Analyze and display SAMBO optimization results."""
    # Check if optimize_result is None or empty
    if not optimize_result:
        print("No SAMBO optimization results available.")
        return

    # Check if optimize_result has the expected structure for SAMBO
    if not hasattr(optimize_result, 'keys') or 'xv' not in optimize_result:
        print("No SAMBO optimization results available.")
        return

    print("\n=== SAMBO Optimization Analysis ===")

    # Extract optimization results from SAMBO format
    xv = optimize_result.get('xv', [])
    funv = optimize_result.get('funv', [])

    if len(xv) > 0 and len(funv) > 0:
        # Convert to DataFrame for analysis
        import pandas as pd

        # Create DataFrame from parameter vectors
        param_names = ['atr_period', 'weight_bull_1', 'weight_bull_2', 'weight_bull_3', 'weight_bull_4',
                      'weight_bull_doji', 'weight_bear_1', 'weight_bear_2', 'weight_bear_3', 'weight_bear_4',
                      'weight_bear_doji', 'weight_bull_bonus', 'weight_bear_bonus',
                      'weight_bull_penalty', 'weight_bear_penalty', 'doji_body_frac', 'prior_idx', 'stop_atr_mult']

        # Convert parameter vectors to DataFrame
        history_data = []
        for i, params in enumerate(xv):
            row = dict(zip(param_names, params))
            row['Return [%]'] = funv[i]
            history_data.append(row)

        history_df = pd.DataFrame(history_data)

        print(f"Total evaluations: {len(history_df)}")
        print(f"Best result found: {history_df['Return [%]'].max():.2f}%")

        # Show parameter evolution
        print("\nParameter evolution:")
        for param in param_names:
            if param in history_df.columns:
                print(f"  {param}: {history_df[param].min():.3f} -> {history_df[param].max():.3f}")

        # Show optimization statistics
        print(f"\nOptimization Statistics:")
        print(f"  Total evaluations: {len(history_df)}")
        print(f"  Best Return [%]: {history_df['Return [%]'].max():.2f}")
        print(f"  Worst Return [%]: {history_df['Return [%]'].min():.2f}")
        print(f"  Average Return [%]: {history_df['Return [%]'].mean():.2f}")

        # Show success status if available
        if 'success' in optimize_result:
            print(f"  Success: {optimize_result['success']}")
        if 'message' in optimize_result:
            print(f"  Message: {optimize_result['message']}")
    else:
        print("No optimization history available in SAMBO result.")


# ========================================
# Main Backtest Runner
# ========================================

def run(path):
    """Load data, run backtest with optimization."""
    # print("Loading data...")
    df = load_csv(path)

    # print("Computing Heikin-Ashi...")
    df = compute_heikin_ashi(df)

    # Prepare dataframe for backtesting
    df_bt = df[['Open', 'High', 'Low', 'Close', 'Volume',
                'HA_open', 'HA_high', 'HA_low', 'HA_close']].copy()
    df_bt.index = pd.to_datetime(df_bt.index)

    # print("Initializing backtest...")
    bt = Backtest(df_bt, HeikinAshiWeightedStrategy,
                  cash=100000,
                  commission=0.001,
                  exclusive_orders=True,
                  finalize_trades=True) # Just to prevent extra log

    # Set low process priority
    try:
        p = psutil.Process(os.getpid())
        p.nice(0)
    except Exception as e:
        print(f"Warning: Could not set process priority: {e}\n")

    # Use random optimization method
    stats, heatmap, optimize_result = bt.optimize(
        atr_period=(8, 20),
        weight_bull_1=(15, 40),  # 0.20 to 0.35
        weight_bull_2=(15, 40),  # 0.15 to 0.30
        weight_bull_3=(20, 40),  # 0.20 to 0.35
        weight_bull_4=(30, 60),  # 0.35 to 0.50
        weight_bull_doji=(0, 50),  # 0.35 to 0.45
        weight_bear_1=(5, 30),  # 0.10 to 0.25
        weight_bear_2=(5, 30),  # 0.10 to 0.25
        weight_bear_3=(5, 30),  # 0.10 to 0.25
        weight_bear_4=(5, 30),  # 0.10 to 0.25
        weight_bear_doji=(0, 50),  # 0.30 to 0.40
        weight_bull_bonus=(0, 15),  # 0.00 to 0.15
        weight_bear_bonus=(0, 15),  # 0.00 to 0.15
        weight_bull_penalty=(0, 15),  # 0.00 to 0.15
        weight_bear_penalty=(0, 15),  # 0.00 to 0.15
        doji_body_frac=(10, 30),  # 0.10 to 0.30
        prior_idx=(-10, -3),  # -8 to -4 (look back 4-8 candles)
        stop_atr_mult=(100, 200),
        # RSI parameters
        rsi_period=(8, 120),  # RSI calculation window
        rsi_lookback=(4, 12),  # Bars to look back for min/max
        rsi_bull_threshold=(20, 50),  # Oversold threshold (20-50)
        rsi_bear_threshold=(50, 80),  # Overbought threshold (50-80)
        weight_rsi_bull=(0, 20),  # 0.00 to 0.25 weight
        weight_rsi_bear=(0, 20),  # 0.00 to 0.25 weight
        maximize='Return [%]',
        method="sambo",
        max_tries=30000,  # Increased for additional RSI parameters
        random_state=1,
        return_heatmap=True,
        return_optimization=True
    )

    '''
    stats, heatmap = bt.optimize(
        atr_period = 14,
        weight_bull_1=[0.3],
        weight_bull_2=[0.25],
        weight_bull_3=[0.3],
        weight_bull_4=[0.35],
        weight_bull_doji=[0.4],
        weight_bear_1=[0.2],
        weight_bear_2=[0.2],
        weight_bear_3=[0.25],
        weight_bear_4=[0.3],
        weight_bear_doji=[0.3],
        weight_bull_bonus=[0.1],
        weight_bear_bonus=[0.1],
        weight_bull_penalty=[0.05],
        weight_bear_penalty=[0.05],
        stop_atr_mult=[1.5],
        maximize='Return [%]',
        return_heatmap=True
    )
    '''

    print("--- Optimization Complete ---\n")
    print(stats)

    print("\n--- Best Parameters ---")
    st = stats._strategy
    print(f"  atr_period: {st.atr_period}")
    print(f"  weight_bull_1: {st.weight_bull_1} | weight_bull_2: {st.weight_bull_2}")
    print(f"  weight_bull_3: {st.weight_bull_3} | weight_bull_4: {st.weight_bull_4}")
    print(f"  weight_bull_doji: {st.weight_bull_doji}")
    print(f"  weight_bull_bonus: {st.weight_bull_bonus} | weight_bear_bonus: {st.weight_bear_bonus}")
    print(f"  weight_bull_penalty: {st.weight_bull_penalty} | weight_bear_penalty: {st.weight_bear_penalty}")
    print(f"  weight_bear_1: {st.weight_bear_1} | weight_bear_2: {st.weight_bear_2}")
    print(f"  weight_bear_3: {st.weight_bear_3} | weight_bear_4: {st.weight_bear_4}")
    print(f"  weight_bear_doji: {st.weight_bear_doji}")
    print(f"  doji_body_frac: {st.doji_body_frac}")
    print(f"  prior_idx: {st.prior_idx}")
    print(f"  stop_atr_mult: {st.stop_atr_mult}")
    print(f"    rsi_period: {st.rsi_period}")
    print(f"    rsi_lookback: {st.rsi_lookback}")
    print(f"    rsi_bull_threshold: {st.rsi_bull_threshold} | rsi_bear_threshold: {st.rsi_bear_threshold}")
    print(f"    weight_rsi_bull: {st.weight_rsi_bull:.3f} | weight_rsi_bear: {st.weight_rsi_bear:.3f}")

    plot_filename = f"HeikinAshi_weighted_{date.today()}.html"
    heatmap_filename = f"HeikinAshi_weighted_heatmap_{date.today()}.html"
    bt.plot(filename=plot_filename)
    plot_heatmaps(heatmap, filename=heatmap_filename)

    print(f"Plot saved as: {plot_filename}  ||  Heatmap saved as: {heatmap_filename}")

    # Analyze SAMBO optimization results
    analyze_sambo_results(optimize_result)

    print("\n--- Top 100 parameter sets (by Return [%]): (.csv) ---")
    top_df = heatmap.sort_values(ascending=False).iloc[:100].reset_index()
    print(top_df.to_csv(index=False,float_format='%.2f'))

# ========================================
# CLI Entry Point
# ========================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python HeikinAshi_weighted.py {data}.csv")
        sys.exit(1)
    else:
        run(sys.argv[1])
