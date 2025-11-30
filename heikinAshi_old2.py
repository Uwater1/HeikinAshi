# HeikinAshi_Optimized.py
# Optimized with 4-Bar Lookback, Normalized Weights, Dual Stop Loss, and Smart Entries
# Usage: python HeikinAshi_Optimized.py {data}.csv

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

# Silence tqdm progress bars
os.environ.setdefault("TQDM_DISABLE", "1")
logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")

# Linux/Mac multiprocessing fix
if os.name == 'posix' and __name__ == '__main__':
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass

# ========================================
# Numba-Optimized Math Core
# ========================================

@jit(nopython=True, cache=True)
def _compute_tr_numba(high, low, close):
    n = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    return tr

@jit(nopython=True, cache=True)
def _compute_ha_numba(o, h, l, c):
    n = len(c)
    ha_o = np.empty(n)
    ha_c = np.empty(n)
    ha_h = np.empty(n)
    ha_l = np.empty(n)
    
    # Initialize first candle
    ha_c[0] = (o[0] + h[0] + l[0] + c[0]) / 4.0
    ha_o[0] = (o[0] + c[0]) / 2.0
    ha_h[0] = h[0]
    ha_l[0] = l[0]
    
    for i in range(1, n):
        ha_c[i] = (o[i] + h[i] + l[i] + c[i]) / 4.0
        ha_o[i] = (ha_o[i-1] + ha_c[i-1]) / 2.0
        ha_h[i] = max(h[i], ha_o[i], ha_c[i])
        ha_l[i] = min(l[i], ha_o[i], ha_c[i])
        
    return ha_o, ha_h, ha_l, ha_c

@jit(nopython=True, cache=True)
def _calc_weighted_score(ha_open, ha_close, atr_arr, weights, idx):
    """
    Calculates the 'Trend Strength Ratio' using 4 bars.
    Returns: (Weighted Average Body Size) / ATR
    Range is typically -2.0 to 2.0. Positive = Bullish.
    """
    score_numerator = 0.0
    weight_sum = 0.0
    
    # weights is array: [w_recent, w_mid1, w_mid2, w_old] (lookback 4)
    # idx is current index. We look at idx, idx-1, idx-2, idx-3
    
    # Check bounds
    if idx < 3:
        return 0.0
        
    atr = atr_arr[idx]
    if atr <= 0: 
        atr = 1.0
        
    # Iterate backwards: 0=recent, 1=mid1, 2=mid2, 3=old
    for i in range(4):
        curr_idx = idx - i
        # Calculate Body (Close - Open)
        body = ha_close[curr_idx] - ha_open[curr_idx]
        
        # Add to weighted sum
        score_numerator += body * weights[i]
        weight_sum += weights[i]
        
    if weight_sum == 0:
        return 0.0
        
    avg_body = score_numerator / weight_sum
    return avg_body / atr

# ========================================
# Data Processing (User Provided)
# ========================================

def load_csv(path):
    """Load CSV data and prepare for backtesting."""
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    df = df.rename(columns={"Close/Last": "Close"})
    
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.sort_index()
    df = df.dropna()
    return df

def prepare_data(path):
    # Wrapper to add indicators to the user's load_csv
    df = load_csv(path)
    
    # Compute ATR using pandas rolling
    h, l, c = df.High.values, df.Low.values, df.Close.values
    tr = _compute_tr_numba(h, l, c)
    atr = pd.Series(tr, index=df.index).rolling(window=14).mean()
    df['ATR'] = atr
    
    # Compute Heikin Ashi
    hao, hah, hal, hac = _compute_ha_numba(
        df.Open.values, df.High.values, df.Low.values, df.Close.values
    )
    df['HA_Open'] = hao
    df['HA_High'] = hah
    df['HA_Low'] = hal
    df['HA_Close'] = hac
    
    # Drop only rows where ATR is NaN (first 13 rows)
    df = df.dropna(subset=['ATR'])
    return df

# ========================================
# Advanced Strategy
# ========================================

class OptimizedHeikinAshi(Strategy):
    """
    Optimized Heikin Ashi Strategy
    1. Normalized Weights (Body % of ATR) using 4 Bars
    2. Separate Entry vs Exit hardness
    3. Dual Stops (Fixed + Trailing)
    4. Smart Stop Orders for 'Anticipated' entry
    """
    
    # --- Optimization Parameters ---
    
    # Entry Weights (0=Recent ... 3=Oldest)
    entry_w1 = 0.4
    entry_w2 = 0.3
    entry_w3 = 0.2
    entry_w4 = 0.1
    
    # Exit Weights (Separate hardness)
    exit_w1 = 0.5
    exit_w2 = 0.3
    exit_w3 = 0.1
    exit_w4 = 0.1
    
    # Thresholds (Normalized: 0.5 = 50% ATR avg body)
    entry_thresh = 0.6
    exit_thresh = 0.4
    
    # Stops
    fixed_stop_mult = 3.0
    trail_stop_mult = 2.0
    
    # Smart Entry: If score is within X% of threshold, place Stop Order
    anticipation_factor = 0.9 

    def init(self):
        # Pre-cast arrays for Numba speed
        self.hao = self.I(lambda: self.data.HA_Open, name="HA_Open")
        self.hac = self.I(lambda: self.data.HA_Close, name="HA_Close")
        self.atr = self.I(lambda: self.data.ATR, name="ATR")
        
        # Track Highest High for Trailing Stop
        self.highest_high = 0.0

    def next(self):
        idx = len(self.data) - 1
        if idx < 10: return
        
        atr = self.atr[-1]
        close = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        
        # --- 1. Compute Scores (4-Bar Lookback) ---
        
        # Entry Score
        w_entry = np.array([self.entry_w1, self.entry_w2, self.entry_w3, self.entry_w4], dtype=np.float64)
        entry_score = _calc_weighted_score(self.hao, self.hac, self.atr, w_entry, idx)
        
        # Exit Score (Separate Weights)
        w_exit = np.array([self.exit_w1, self.exit_w2, self.exit_w3, self.exit_w4], dtype=np.float64)
        raw_exit_score = _calc_weighted_score(self.hao, self.hac, self.atr, w_exit, idx)
        
        # --- 2. Position Management (Exit) ---
        
        if self.position:
            # Update Highest High for Trailing Stop
            if high > self.highest_high:
                self.highest_high = high
                
            # A. Trailing Stop Logic (Chandelier Exit)
            trail_price = self.highest_high - (self.trail_stop_mult * atr)
            
            # B. Fixed Stop Logic & Update
            for trade in self.trades:
                current_sl = trade.sl if trade.sl else 0
                # Move stop UP only
                new_sl = max(current_sl, trail_price)
                trade.sl = new_sl

            # C. Signal Exit (Soft Exit based on Heikin Ashi Reversal)
            # If trend turns Red (negative score) and magnitude > threshold
            if raw_exit_score < -self.exit_thresh:
                self.position.close()
                return

        # --- 3. Entry Logic ---
        
        if not self.position:
            # Reset Highest High tracking
            self.highest_high = high
            
            # Standard Fixed Stop distance
            stop_dist = self.fixed_stop_mult * atr
            
            # A. Standard Strong Entry (Close > Threshold)
            if entry_score >= self.entry_thresh:
                sl_price = close - stop_dist
                # Ensure stop loss is positive
                if sl_price > 0:
                    self.buy(sl=sl_price)
                
            # B. Anticipatory Entry (Smart Stop Order)
            # Logic: If we are close (anticipation_factor), place a STOP order slightly above High.
            # This captures the breakout ("Wanted Price") needed to confirm the trend.
            elif entry_score > (self.entry_thresh * self.anticipation_factor):
                
                wanted_price = high + (0.1 * atr) 
                sl_price = wanted_price - stop_dist
                
                # Ensure stop loss is positive
                if sl_price > 0:
                    # Valid only for next bar
                    self.buy(stop=wanted_price, sl=sl_price)

# ========================================
# Execution
# ========================================

def run_optimization(path):
    print(f"Loading {path}...")
    df = prepare_data(path)
    
    bt = Backtest(df, OptimizedHeikinAshi, cash=100000, commission=0.001)
    
    print("Starting Optimization (Targeting ~250k iterations)...")
    
    # Optimization Space
    # Refined based on previous result: older weights (w3, w4) were significant.
    # Entry Weights: 3*3*3*3 = 81
    # Exit Weights: 2*2*2*1 = 8
    # Thresholds: 5 * 3 = 15
    # Stops: 3 * 3 = 9
    # Anticipation: 3
    # Total: 81 * 8 * 15 * 9 * 3 = 262,440 iterations.
    
    '''  
    stats, heatmap = bt.optimize(  
        # Entry Weights (Focus on granularity)
        entry_w1 = [0.1, 0.2, 0.3], 
        entry_w2 = [0.1, 0.2, 0.3],
        entry_w3 = [0.2, 0.3, 0.4], # Weighted higher based on previous results
        entry_w4 = [0.2, 0.3, 0.4], # Weighted higher based on previous results
        
        # Exit Weights (Simpler)
        exit_w1 = [0.5, 0.7],       # Exit needs fast reaction (recent weights higher)
        exit_w2 = [0.2, 0.3],
        exit_w3 = [0.1, 0.2],
        exit_w4 = [0.1],
        
        # Thresholds (Normalized Units)
        entry_thresh = [0.4, 0.5, 0.6, 0.7, 0.8],
        exit_thresh  = [0.2, 0.3, 0.4],
        
        # Stops
        trail_stop_mult = [2.0, 3.0, 4.0],
        fixed_stop_mult = [2.0, 3.0, 4.0],
        
        # Smart Entry
        anticipation_factor = [0.85, 0.9, 0.95],
        
        maximize='Return [%]',
        return_heatmap=True
    )
    '''

    stats, heatmap = bt.optimize(  
        # Entry Weights (Focus on granularity)
        entry_w1 = 0.3, 
        entry_w2 = 0.3,
        entry_w3 = 0.4, # Weighted higher based on previous results
        entry_w4 = 0.4, # Weighted higher based on previous results
        
        # Exit Weights (Simpler)
        exit_w1 = 0.7,       # Exit needs fast reaction (recent weights higher)
        exit_w2 = 0.3,
        exit_w3 = 0.2,
        exit_w4 = 0.1,
        
        # Thresholds (Normalized Units)
        entry_thresh = [0.4, 0.5, 0.6, 0.7, 0.8],
        exit_thresh  = [0.2, 0.3, 0.4],
        
        # Stops
        trail_stop_mult = 4.0,
        fixed_stop_mult = 2.0,
        
        # Smart Entry
        anticipation_factor = [0.85, 0.9, 0.95],
        
        maximize='Return [%]',
        return_heatmap=True
    )
    
    print("\n--- Optimization Complete ---")
    print(stats)
    
    print("\n--- Best Parameters ---")
    st = stats._strategy
    print(f"  Entry Weights: {st.entry_w1}, {st.entry_w2}, {st.entry_w3}, {st.entry_w4}")
    print(f"  Exit Weights: {st.exit_w1}, {st.exit_w2}, {st.exit_w3}, {st.exit_w4}")
    print(f"  Thresholds: Entry {st.entry_thresh} | Exit {st.exit_thresh}")
    print(f"  Stops: Fixed {st.fixed_stop_mult} | Trail {st.trail_stop_mult}")
    print(f"  Anticipation: {st.anticipation_factor}")
    
    # Required Output Format
    plot_filename = f"HeikinAshi_optimized_{date.today()}.html"
    heatmap_filename = f"HeikinAshi_heatmap_{date.today()}.html"
    bt.plot(filename=plot_filename)
    plot_heatmaps(heatmap, filename=heatmap_filename)
    print(f"Plot saved as: {plot_filename}  ||  Heatmap saved as: {heatmap_filename}")
    print("\n--- Top 120 parameter sets (by Return [%]): (.csv) ---")
    top_df = heatmap.sort_values(ascending=False).iloc[:120].reset_index()
    print(top_df.to_csv(index=False,float_format='%.2f'))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python heikinAshi_Optimized.py {data}.csv")
    else:
        run_optimization(sys.argv[1])