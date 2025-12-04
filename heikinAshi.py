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
def _compute_score_numba(ha_open, ha_close, atr_cur, weights, doji_weight, doji_threshold,
                        context_quality, weight_bull_bonus=0.1, weight_bear_bonus=0.1,
                        weight_bull_penalty=0.05, weight_bear_penalty=0.05, is_entry=True,
                        use_legacy_doji=False, legacy_doji_body_frac=0.20, legacy_doji_position=-6):
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

    # Momentum tracking for the 3rd and 4th selected bars
    if len(selected) >= 3:
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

    if len(selected) >= 4:
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

    # Context-specific doji integration
    if use_legacy_doji:
        # Legacy binary doji logic (exact reproduction of original)
        try:
            prior_idx = legacy_doji_position
            prior_body = abs(ha_close[prior_idx] - ha_open[prior_idx])
            if prior_body < legacy_doji_body_frac * atr_cur:
                if is_entry:
                    if ha_close[prior_idx] < ha_open[prior_idx]:
                        score += doji_weight
                else:
                    if ha_close[prior_idx] > ha_open[prior_idx]:
                        score += doji_weight
        except:
            pass
    else:
        # New quality-based logic
        if context_quality > doji_threshold:
            if is_entry:
                # Bullish hammer reduces entry score (bearish signal for entry)
                score -= doji_weight * context_quality
            else:
                # Bearish momentum loss increases exit score (bearish signal for exit)
                score += doji_weight * context_quality

    return score

@jit(nopython=True, cache=True)
def get_bearish_momentum_loss_quality_numba(ha_open, ha_close, ha_high, ha_low, doji_lookback_bear=5, idx=0):
    """
    Detects loss of growing momentum in bullish sequences.
    Only used for exit signals (is_entry = False).
    Returns 0.0-1.0 quality score.
    """
    # 1. Analyze bullish sequence before current momentum system
    bullish_sequence = []
    red_emergence_count = 0

    for i in range(doji_lookback_bear):
        pos = idx - i
        if pos < 0:
            continue

        is_green = ha_close[pos] > ha_open[pos]
        body_size = abs(ha_close[pos] - ha_open[pos])

        if is_green:
            bullish_sequence.append(body_size)
        else:
            red_emergence_count += 1

    if len(bullish_sequence) < 2:
        return 0.0

    # 2. Body shrinkage calculation
    current_body = bullish_sequence[0]  # Most recent green body
    avg_prior_bodies = 0.0
    for i in range(1, len(bullish_sequence)):
        avg_prior_bodies += bullish_sequence[i]
    avg_prior_bodies /= (len(bullish_sequence) - 1)

    shrinkage_score = 0.0
    if current_body < avg_prior_bodies:
        shrinkage_score = 1.0 - (current_body / avg_prior_bodies)

    # 3. Red emergence penalty (red bars in bullish context)
    red_penalty = min(1.0, red_emergence_count * 0.25)

    # 4. Upper shadow analysis (weak bullish conviction)
    upper_shadow_penalty = 0.0
    if ha_close[idx] > ha_open[idx]:  # Current is green
        body_size = ha_close[idx] - ha_open[idx]
        upper_shadow = ha_high[idx] - ha_close[idx]
        if upper_shadow > body_size * 0.3:  # Significant upper shadow
            upper_shadow_penalty = 0.3

    # Combined score: shrinkage is primary signal
    return min(1.0, (shrinkage_score * 0.5) + (red_penalty * 0.3) + (upper_shadow_penalty * 0.2))

@jit(nopython=True, cache=True)
def get_bullish_hammer_quality_numba(ha_open, ha_close, ha_high, ha_low, low_prices, atr_cur, doji_lookback_bull=15, hammer_atr_factor=0.5, idx=0):
    """
    Detects hammer patterns with sharp rejection at support.
    Only used for entry signals (is_entry = True).
    Returns 0.0-1.0 quality score.
    """
    ha_o, ha_c, ha_h, ha_l = ha_open[idx], ha_close[idx], ha_high[idx], ha_low[idx]

    body_size = abs(ha_c - ha_o)
    lower_shadow = min(ha_o, ha_c) - ha_l
    upper_shadow = ha_h - max(ha_o, ha_c)
    total_range = ha_h - ha_l

    if total_range == 0 or atr_cur == 0:
        return 0.0

    # 1. Hammer pattern recognition
    hammer_score = 0.0
    if lower_shadow > body_size * 2 and upper_shadow < body_size * 0.5:
        # Classic hammer: long lower wick, small body, small upper wick
        hammer_score = min(1.0, lower_shadow / (atr_cur * hammer_atr_factor))

    # 2. Support proximity for double bottom confirmation
    support_level = ha_l  # Initialize with current low
    for i in range(min(doji_lookback_bull, idx + 1)):
        pos = idx - i
        if pos >= 0 and low_prices[pos] < support_level:
            support_level = low_prices[pos]

    dist_from_support = ha_l - support_level
    tolerance = atr_cur * 0.3

    support_score = 0.0
    if dist_from_support <= tolerance:
        support_score = 1.0 - (dist_from_support / tolerance)

    # 3. Body smallness relative to range
    body_ratio = body_size / total_range
    small_body_bonus = 0.0
    if body_ratio < 0.3:  # Body less than 30% of total range
        small_body_bonus = 1.0 - (body_ratio / 0.3)

    # Combined score: hammer pattern is most important for entry
    return min(1.0, (hammer_score * 0.5) + (support_score * 0.3) + (small_body_bonus * 0.2))

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

    # Bear weights for exit (scaled by 0.25 * 0.8 as per user requirement)
    weight_bear_1 = 0.25
    weight_bear_2 = 0.20
    weight_bear_3 = 0.25
    weight_bear_4 = 0.30

    # Stop in ATR multiples
    stop_atr_mult = 1.5

    # Legacy doji parameters (for backward compatibility)
    doji_body_frac = 0.20       # Original doji body fraction (now optimizable)
    legacy_doji_position = -6   # Original doji position (now optimizable)

    # Context-specific doji parameters
    # Bearish momentum loss parameters (for exit signals only)
    doji_lookback_bear = 5      # Lookback periods for analyzing bullish sequences
    doji_threshold_bear = 0.4   # Quality threshold for bearish momentum loss

    # Bullish hammer parameters (for entry signals only)
    doji_lookback_bull = 15     # Lookback periods for support proximity
    doji_threshold_bull = 0.4   # Quality threshold for bullish hammer
    hammer_atr_factor = 0.5     # Hammer shadow length as fraction of ATR

    # Separate weights for context-specific signals
    weight_bear_momentum = 0.0  # Weight for bearish momentum loss (0 = legacy mode)
    weight_bull_hammer = 0.0    # Weight for bullish hammer rejection (0 = legacy mode)

    # Momentum tracking parameters
    weight_bull_bonus = 0.1    # Bonus weight for accelerating bullish momentum
    weight_bear_bonus = 0.1     # Bonus weight for accelerating bearish momentum
    weight_bull_penalty = 0.05 # Penalty weight for decelerating bullish momentum
    weight_bear_penalty = 0.05 # Penalty weight for decelerating bearish momentum

    def init(self):
        # Scale weights from integers (hundredths) to floats
        self.weight_bull_1 = self.weight_bull_1 / 100.0
        self.weight_bull_2 = self.weight_bull_2 / 100.0
        self.weight_bull_3 = self.weight_bull_3 / 100.0
        self.weight_bull_4 = self.weight_bull_4 / 100.0
        self.weight_bear_1 = self.weight_bear_1 / 100.0
        self.weight_bear_2 = self.weight_bear_2 / 100.0
        self.weight_bear_3 = self.weight_bear_3 / 100.0
        self.weight_bear_4 = self.weight_bear_4 / 100.0
        self.weight_bull_bonus = self.weight_bull_bonus / 100.0
        self.weight_bear_bonus = self.weight_bear_bonus / 100.0
        self.weight_bull_penalty = self.weight_bull_penalty / 100.0
        self.weight_bear_penalty = self.weight_bear_penalty / 100.0
        self.stop_atr_mult = self.stop_atr_mult / 100.0

        # Scale legacy doji parameters
        self.doji_body_frac = self.doji_body_frac / 100.0

        # Scale new doji parameters
        self.doji_threshold_bear = self.doji_threshold_bear / 100.0
        self.doji_threshold_bull = self.doji_threshold_bull / 100.0
        self.hammer_atr_factor = self.hammer_atr_factor / 100.0
        self.weight_bear_momentum = self.weight_bear_momentum / 100.0
        self.weight_bull_hammer = self.weight_bull_hammer / 100.0

        # Register ATR indicator
        self.atr = self.I(lambda: indicator_atr(
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_period
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

    def compute_entry_score(self):
        """Compute weighted entry score with legacy or quality-based doji detection."""
        atr_cur = float(self.atr[-1] if len(self.atr) > 0 else 1.0)
        if atr_cur <= 0:
            atr_cur = 1.0

        # Determine which doji system to use
        use_legacy = (self.weight_bull_hammer == 0.0)

        if use_legacy:
            # Legacy mode: no quality computation needed
            bull_quality = 0.0
            doji_weight = 0.4  # Use default legacy weight
            doji_threshold = 0.0  # Not used in legacy mode
        else:
            # Quality mode: compute bullish hammer quality
            bull_quality = get_bullish_hammer_quality_numba(
                self.ha_open, self.ha_close, self.ha_high, self.ha_low,
                self.data.Low, np.float32(atr_cur),
                self.doji_lookback_bull, self.hammer_atr_factor, idx=-1
            )
            doji_weight = self.weight_bull_hammer
            doji_threshold = self.doji_threshold_bull

        bull_weights = np.array([self.weight_bull_1, self.weight_bull_2, self.weight_bull_3, self.weight_bull_4], dtype=np.float32)
        return float(_compute_score_numba(
            self.ha_open, self.ha_close, np.float32(atr_cur), bull_weights,
            np.float32(doji_weight), np.float32(doji_threshold),
            np.float32(bull_quality),
            np.float32(self.weight_bull_bonus), np.float32(self.weight_bear_bonus),
            np.float32(self.weight_bull_penalty), np.float32(self.weight_bear_penalty),
            is_entry=True, use_legacy_doji=use_legacy,
            legacy_doji_body_frac=self.doji_body_frac, legacy_doji_position=self.legacy_doji_position
        ))

    def compute_exit_score(self):
        """Compute weighted exit score with legacy or quality-based doji detection."""
        atr_cur = float(self.atr[-1] if len(self.atr) > 0 else 1.0)
        if atr_cur <= 0:
            atr_cur = 1.0

        # Determine which doji system to use
        use_legacy = (self.weight_bear_momentum == 0.0)

        if use_legacy:
            # Legacy mode: no quality computation needed
            bear_quality = 0.0
            doji_weight = 0.3  # Use default legacy weight
            doji_threshold = 0.0  # Not used in legacy mode
        else:
            # Quality mode: compute bearish momentum loss quality
            bear_quality = get_bearish_momentum_loss_quality_numba(
                self.ha_open, self.ha_close, self.ha_high, self.ha_low,
                self.doji_lookback_bear, idx=-1
            )
            doji_weight = self.weight_bear_momentum
            doji_threshold = self.doji_threshold_bear

        bear_weights = np.array([self.weight_bear_1, self.weight_bear_2, self.weight_bear_3, self.weight_bear_4], dtype=np.float32)
        return float(_compute_score_numba(
            self.ha_open, self.ha_close, np.float32(atr_cur), bear_weights,
            np.float32(doji_weight), np.float32(doji_threshold),
            np.float32(bear_quality),
            np.float32(self.weight_bull_bonus), np.float32(self.weight_bear_bonus),
            np.float32(self.weight_bull_penalty), np.float32(self.weight_bear_penalty),
            is_entry=False, use_legacy_doji=use_legacy,
            legacy_doji_body_frac=self.doji_body_frac, legacy_doji_position=self.legacy_doji_position
        ))

    def next(self):
        """Execute strategy logic on each bar."""
        price = float(self.data.Close[-1])
        atr_cur = float(self.atr[-1] if len(self.atr) > 0 else 0.0)

        # Entry logic - use bull weights, trigger when score >= 1.0
        if not self.position:
            entry_score = self.compute_entry_score()
            if entry_score >= 1.0:  # Fixed threshold for entry
                sl_price = max(price * 0.95, price - self.stop_atr_mult * atr_cur) if atr_cur > 0 else price * 0.97
                try:
                    self.buy(sl=sl_price)
                except Exception:
                    self.buy()

        # Exit logic - use bear weights, trigger when score >= 1.0
        else:
            exit_score = self.compute_exit_score()
            if exit_score >= 1.0:  # Fixed threshold for exit
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
                      'weight_bear_1', 'weight_bear_2', 'weight_bear_3', 'weight_bear_4',
                      'doji_lookback_bear', 'doji_threshold_bear', 'doji_lookback_bull', 'doji_threshold_bull',
                      'hammer_atr_factor', 'weight_bear_momentum', 'weight_bull_hammer',
                      'weight_bull_bonus', 'weight_bear_bonus', 'weight_bull_penalty', 'weight_bear_penalty', 'stop_atr_mult']

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

    #'''
    # Use random optimization method with full backward compatibility
    stats, heatmap, optimize_result = bt.optimize(
        atr_period=(8, 20),
        weight_bull_1=(25, 35),  # 0.25 to 0.35
        weight_bull_2=(15, 30),  # 0.15 to 0.25
        weight_bull_3=(25, 35),  # 0.30 to 0.40
        weight_bull_4=(35, 50),  # 0.40 to 0.50
        weight_bear_1=(10, 25),  # 0.15 to 0.25
        weight_bear_2=(10, 25),  # 0.10 to 0.20
        weight_bear_3=(10, 25),  # 0.15 to 0.20
        weight_bear_4=(10, 25),  # 0.15 to 0.25
        # Legacy doji parameters (include original values)
        doji_body_frac=(10, 30),        # 0.10-0.30 (includes 0.20)
        legacy_doji_position=(-8, -4),  # -8 to -4 (includes -6)
        # Context-specific doji parameters (include original hardcoded values)
        doji_lookback_bear=(3, 8),      # 3-8 (includes 5)
        doji_threshold_bear=(20, 60),   # 0.2-0.6 bear quality threshold
        doji_lookback_bull=(10, 25),    # 10-25 (includes 15)
        doji_threshold_bull=(30, 60),   # 0.3-0.6 bull quality threshold
        hammer_atr_factor=(30, 70),     # 0.3-0.7 ATR fraction (includes 0.5)
        # New weights (0 enables legacy mode)
        weight_bear_momentum=(0, 50),   # 0.0-0.5 (0 = legacy mode)
        weight_bull_hammer=(0, 60),     # 0.0-0.6 (0 = legacy mode)
        weight_bull_bonus=(0, 15),  # 0.05 to 0.15
        weight_bear_bonus=(0, 15),  # 0.05 to 0.15
        weight_bull_penalty=(0, 15),  # 0.00 to 0.10
        weight_bear_penalty=(0, 15),  # 0.00 to 0.10
        stop_atr_mult=150,
        maximize='Return [%]',
        method="sambo",
        max_tries=50000,
        random_state=1,
        return_heatmap=True,
        return_optimization=True
    )
    #'''

    '''
    stats, heatmap = bt.optimize(
        atr_period = 14,
        weight_bull_1=[0.3],
        weight_bull_2=[0.25],
        weight_bull_3=[0.3],
        weight_bull_4=[0.35],
        weight_bear_1=[0.2],
        weight_bear_2=[0.2],
        weight_bear_3=[0.25],
        weight_bear_4=[0.3],
        doji_lookback_bear=[5],
        doji_threshold_bear=[0.4],
        doji_lookback_bull=[15],
        doji_threshold_bull=[0.4],
        hammer_atr_factor=[0.5],
        weight_bear_momentum=[0.35],
        weight_bull_hammer=[0.45],
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
    print(f"  weight_bear_1: {st.weight_bear_1} | weight_bear_2: {st.weight_bear_2}")
    print(f"  weight_bear_3: {st.weight_bear_3} | weight_bear_4: {st.weight_bear_4}")
    print(f"  doji_lookback_bear: {st.doji_lookback_bear} | doji_threshold_bear: {st.doji_threshold_bear}")
    print(f"  doji_lookback_bull: {st.doji_lookback_bull} | doji_threshold_bull: {st.doji_threshold_bull}")
    print(f"  hammer_atr_factor: {st.hammer_atr_factor}")
    print(f"  weight_bear_momentum: {st.weight_bear_momentum} | weight_bull_hammer: {st.weight_bull_hammer}")
    print(f"  weight_bull_bonus: {st.weight_bull_bonus} | weight_bear_bonus: {st.weight_bear_bonus}")
    print(f"  weight_bull_penalty: {st.weight_bull_penalty} | weight_bear_penalty: {st.weight_bear_penalty}")
    print(f"  stop_atr_mult: {st.stop_atr_mult}")

    plot_filename = f"HeikinAshi_weighted_{date.today()}.html"
    bt.plot(filename=plot_filename)

    print(f"Plot saved as: {plot_filename}")

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
