# HeikinAshi_Combined.py
# Combined implementation with Numba optimization and weighted strategy
# Usage: python HeikinAshi_Combined.py {data}.csv

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

# ========================================
# Data Loading
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
    """Weighted Heikin-Ashi trend-following strategy."""
    
    # ATR period
    atr_period = 14

    # Candle weights (later candles contribute more)
    weight_1 = 0.15
    weight_2 = 0.20
    weight_3 = 0.25
    weight_4 = 0.30

    # Bonuses
    weight_doji = 0.20
    weight_volume = 0.20

    # Thresholds
    entry_threshold = 1.0
    exit_threshold = 1.0

    # Stop in ATR multiples
    stop_atr_mult = 2.0

    # Doji threshold (fraction of ATR)
    doji_body_frac = 0.20

    # Volume lookback for MA
    vol_ma_period = 20

    def init(self):
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

        # Volume array accessor
        self.vol = self.I(lambda: np.asarray(self.data.Volume, dtype=np.float32))

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
        """Compute weighted entry score using up to 4 most recent HA candles."""
        atr_cur = float(self.atr[-1] if len(self.atr) > 0 else 1.0)
        if atr_cur <= 0:
            atr_cur = 1.0

        weights = [self.weight_1, self.weight_2, self.weight_3, self.weight_4]
        score = 0.0

        # Lookback up to 4 bars
        for i in range(4):
            idx = -(i + 1)
            try:
                ha_o = float(self.ha_open[idx])
                ha_c = float(self.ha_close[idx])
            except Exception:
                break

            body = ha_c - ha_o
            if body > 0:  # green candle
                norm_body = body / atr_cur
                norm_body = min(norm_body, 3.0)
                score += weights[i] * norm_body

        # Doji bonus: check bar -6 for small red/doji
        try:
            prior_idx = -6
            prior_body = abs(float(self.ha_close[prior_idx]) - float(self.ha_open[prior_idx]))
            if prior_body < self.doji_body_frac * atr_cur and (float(self.ha_close[prior_idx]) < float(self.ha_open[prior_idx])):
                score += self.weight_doji
        except Exception:
            pass

        # Volume bonus
        try:
            vol_cur = float(self.vol[-1])
            if len(self.vol) >= self.vol_ma_period + 1:
                vol_ma = float(np.nanmean(self.vol[-(self.vol_ma_period + 1):-1]))
            else:
                vol_ma = float(np.nanmean(self.vol))

            if vol_ma > 0 and vol_cur > 1.2 * vol_ma:
                score += self.weight_volume
        except Exception:
            pass

        return score

    def compute_exit_score(self):
        """Compute weighted exit score using up to 4 most recent HA candles."""
        atr_cur = float(self.atr[-1] if len(self.atr) > 0 else 1.0)
        if atr_cur <= 0:
            atr_cur = 1.0

        weights = [self.weight_1, self.weight_2, self.weight_3, self.weight_4]
        score = 0.0

        for i in range(4):
            idx = -(i + 1)
            try:
                ha_o = float(self.ha_open[idx])
                ha_c = float(self.ha_close[idx])
            except Exception:
                break

            body = ha_o - ha_c  # positive if red
            if body > 0:
                norm_body = body / atr_cur
                norm_body = min(norm_body, 3.0)
                score += weights[i] * norm_body

        # Doji bonus: check bar -6 for small green doji
        try:
            prior_idx = -6
            prior_body = abs(float(self.ha_close[prior_idx]) - float(self.ha_open[prior_idx]))
            if prior_body < self.doji_body_frac * atr_cur and (float(self.ha_close[prior_idx]) > float(self.ha_open[prior_idx])):
                score += self.weight_doji
        except Exception:
            pass

        # Volume bonus on red bars
        try:
            vol_cur = float(self.vol[-1])
            if len(self.vol) >= self.vol_ma_period + 1:
                vol_ma = float(np.nanmean(self.vol[-(self.vol_ma_period + 1):-1]))
            else:
                vol_ma = float(np.nanmean(self.vol))

            if (float(self.ha_close[-1]) < float(self.ha_open[-1])) and vol_ma > 0 and vol_cur > 1.2 * vol_ma:
                score += self.weight_volume
        except Exception:
            pass

        return score

    def next(self):
        """Execute strategy logic on each bar."""
        price = float(self.data.Close[-1])
        atr_cur = float(self.atr[-1] if len(self.atr) > 0 else 0.0)

        # Entry logic
        if not self.position:
            entry_score = self.compute_entry_score()

            if entry_score >= self.entry_threshold:
                sl_price = max(0.0, price - self.stop_atr_mult * atr_cur) if atr_cur > 0 else max(0.0, price * 0.98)
                
                try:
                    self.buy(sl=sl_price)
                except Exception:
                    self.buy()

        # Exit logic
        else:
            exit_score = self.compute_exit_score()
            if exit_score >= self.exit_threshold:
                try:
                    self.position.close()
                except Exception:
                    try:
                        self.sell()
                    except Exception:
                        pass

# ========================================
# Main Backtest Runner
# ========================================

def run(path):
    """Load data, run backtest with optimization."""
    print("Loading data...")
    df = load_csv(path)
    
    print("Computing Heikin-Ashi...")
    df = compute_heikin_ashi(df)
    
    # Prepare dataframe for backtesting
    df_bt = df[['Open', 'High', 'Low', 'Close', 'Volume',
                'HA_open', 'HA_high', 'HA_low', 'HA_close']].copy()
    df_bt.index = pd.to_datetime(df_bt.index)
    
    print("Initializing backtest...")
    bt = Backtest(df_bt, HeikinAshiWeightedStrategy,
                  cash=100000,
                  commission=0.001,
                  exclusive_orders=True,
                  finalize_trades=True) # Just to prevent extra log
    
    # Set low process priority
    try:
        p = psutil.Process(os.getpid())
        p.nice(10)
        print("✓ Process priority set to low\n")
    except Exception as e:
        print(f"Warning: Could not set process priority: {e}\n")
    
    print("--- Starting Optimization ---")
    print("This may take 10-20 minutes depending on your system...\n")


    stats, heatmap = bt.optimize(
        weight_1=[0.2, 0.25, 0.25, 0.3],
        weight_2=[0.2, 0.25, 0.3],
        weight_3=[0.15, 0.2, 0.25],
        weight_4=[0.2, 0.25, 0.3],
        weight_doji=[0.1, 0.2, 0.3],
        weight_volume= [0.05, 0.1, 0.15],
        entry_threshold=[0.7, 0.8, 0.9, 1.0],
        exit_threshold=[1.0, 1.1, 1.2],
        stop_atr_mult=[2.0, 2.5, 3.0, 3.5, 4.0],        
        maximize='Return [%]',
        return_heatmap=True
    )

    '''
    stats, heatmap = bt.optimize(
        weight_1=[0.2, 0.25],
        weight_2=0.25,
        weight_3=0.25,
        weight_4=0.25,
        weight_doji=0.25,
        weight_volume= 0.1,
        entry_threshold=[0.7, 0.9],
        exit_threshold=[1.0, 1.2],
        stop_atr_mult=[2.0, 3.0],        
        maximize='Return [%]',
        return_heatmap=True
    )
    '''

    
    print("\n--- Optimization Complete ---\n")
    print(stats)
    
    print("\n--- Best Parameters ---")
    st = stats._strategy
    print(f"  weight_1: {st.weight_1}")
    print(f"  weight_2: {st.weight_2}")
    print(f"  weight_3: {st.weight_3}")
    print(f"  weight_4: {st.weight_4}")
    print(f"  weight_doji: {st.weight_doji}")
    print(f"  weight_volume: {st.weight_volume}")
    print(f"  entry_threshold: {st.entry_threshold}")
    print(f"  exit_threshold: {st.exit_threshold}")
    print(f"  stop_atr_mult: {st.stop_atr_mult}")
    
    print("\nPlotting results...")
    plot_filename = f"HeikinAshi_optimized_{date.today()}.html"
    bt.plot(filename=plot_filename)
    plot_heatmaps(heatmap, filename=heatmap_filename)
    print(f"Plot saved as: {plot_filename}")
    print(f"Heatmap saved as: {heatmap_filename}")

# ========================================
# CLI Entry Point
# ========================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python HeikinAshi_Combined.py {data}.csv")
        sys.exit(1)
    else:
        run(sys.argv[1])
