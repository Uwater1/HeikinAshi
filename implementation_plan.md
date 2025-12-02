# Implementation Plan: Replace entry_threshold/exit_threshold with Weighted Bull/Bear System

## Current Issue Analysis
The current implementation uses a single scoring system with `entry_threshold` and `exit_threshold` parameters, but this is logically incorrect because:
- Entry conditions (bullish) and exit conditions (bearish) should use different weighted systems
- The current system applies the same weights to both entry and exit calculations
- Bullish market conditions require different weighting than bearish conditions

## Proposed Solution
Replace the current threshold system with separate weighted systems:
- **Bull weights** (for entry): `weight_bull_1`, `weight_bull_2`, `weight_bull_3`, `weight_bull_4` (range: 0.7-0.9)
- **Bear weights** (for exit): `weight_bear_1`, `weight_bear_2`, `weight_bear_3`, `weight_bear_4` (range: 1.1-1.3)

## Implementation Steps

### 1. Modify Strategy Class Parameters
Replace current parameters:
```python
# Remove these:
entry_threshold = 1.0
exit_threshold = 1.0

# Add these:
weight_bull_1 = 0.7
weight_bull_2 = 0.8
weight_bull_3 = 0.85
weight_bull_4 = 0.9
weight_bear_1 = 1.1
weight_bear_2 = 1.2
weight_bear_3 = 1.25
weight_bear_4 = 1.3
```

### 2. Update _compute_score_numba Function
Modify the function signature and logic to accept separate bull and bear weights:
```python
@jit(nopython=True, cache=True)
def _compute_score_numba(ha_open, ha_close, atr_cur, bull_weights, bear_weights, doji_body_frac, weight_doji, is_entry=True):
    """Numba-optimized score calculation with separate bull/bear weights."""
    score = 0.0
    lookback = 4

    weights = bull_weights if is_entry else bear_weights

    for i in range(lookback):
        idx = -(i + 1)
        ha_o = ha_open[idx]
        ha_c = ha_close[idx]
        body = ha_c - ha_o

        if is_entry:
            if body > 0:  # green candle
                norm_body = body / atr_cur
                if norm_body < 0.3:
                    norm_body = 0.3
                score += weights[i] * norm_body
        else:
            body = ha_o - ha_c  # positive if red
            if body > 0:
                norm_body = body / atr_cur
                if norm_body < 0.3:
                    norm_body = 0.3
                score += weights[i] * norm_body

    # Doji bonus: check bar -6
    try:
        prior_idx = -6
        prior_body = abs(ha_close[prior_idx] - ha_open[prior_idx])
        if prior_body < doji_body_frac * atr_cur:
            if is_entry:
                if ha_close[prior_idx] < ha_open[prior_idx]:
                    score += weight_doji
            else:
                if ha_close[prior_idx] > ha_open[prior_idx]:
                    score += weight_doji
    except:
        pass

    return score
```

### 3. Update Strategy Methods
Modify `compute_entry_score` and `compute_exit_score` methods:
```python
def compute_entry_score(self):
    """Compute weighted entry score using bull weights."""
    atr_cur = float(self.atr[-1] if len(self.atr) > 0 else 1.0)
    if atr_cur <= 0:
        atr_cur = 1.0

    bull_weights = np.array([self.weight_bull_1, self.weight_bull_2, self.weight_bull_3, self.weight_bull_4], dtype=np.float32)
    return float(_compute_score_numba(
        self.ha_open, self.ha_close, np.float32(atr_cur), bull_weights,
        np.array([self.weight_bear_1, self.weight_bear_2, self.weight_bear_3, self.weight_bear_4], dtype=np.float32),
        np.float32(self.doji_body_frac), np.float32(self.weight_doji),
        is_entry=True
    ))

def compute_exit_score(self):
    """Compute weighted exit score using bear weights."""
    atr_cur = float(self.atr[-1] if len(self.atr) > 0 else 1.0)
    if atr_cur <= 0:
        atr_cur = 1.0

    bear_weights = np.array([self.weight_bear_1, self.weight_bear_2, self.weight_bear_3, self.weight_bear_4], dtype=np.float32)
    return float(_compute_score_numba(
        self.ha_open, self.ha_close, np.float32(atr_cur),
        np.array([self.weight_bull_1, self.weight_bull_2, self.weight_bull_3, self.weight_bull_4], dtype=np.float32),
        bear_weights,
        np.float32(self.doji_body_frac), np.float32(self.weight_doji),
        is_entry=False
    ))
```

### 4. Update Optimization Parameters
Replace the optimization parameters in the `run` function:
```python
stats, heatmap = bt.optimize(
    atr_period = [10, 14, 20, 30],
    weight_bull_1=[0.7, 0.75, 0.8, 0.85, 0.9],
    weight_bull_2=[0.7, 0.75, 0.8, 0.85, 0.9],
    weight_bull_3=[0.7, 0.75, 0.8, 0.85, 0.9],
    weight_bull_4=[0.7, 0.75, 0.8, 0.85, 0.9],
    weight_bear_1=[1.1, 1.15, 1.2, 1.25, 1.3],
    weight_bear_2=[1.1, 1.15, 1.2, 1.25, 1.3],
    weight_bear_3=[1.1, 1.15, 1.2, 1.25, 1.3],
    weight_bear_4=[1.1, 1.15, 1.2, 1.25, 1.3],
    weight_bull_doji=[0.25, 0.3, 0.35, 0.4],
    weight_bear_doji=[0.25, 0.3, 0.35, 0.4],
    stop_atr_mult=[1.3, 1.5, 2.0],
    maximize='Return [%]',
    return_heatmap=True
)
```

### 5. Update Entry/Exit Logic
Modify the `next` method to use the new scoring system:
```python
def next(self):
    """Execute strategy logic on each bar."""
    price = float(self.data.Close[-1])
    atr_cur = float(self.atr[-1] if len(self.atr) > 0 else 0.0)

    # Entry logic - use bull weights
    if not self.position:
        entry_score = self.compute_entry_score()
        # Entry threshold can be a fixed value or another parameter
        if entry_score >= 1.0:  # This threshold can be adjusted
            sl_price = max(price * 0.95, price - self.stop_atr_mult * atr_cur) if atr_cur > 0 else price * 0.97
            try:
                self.buy(sl=sl_price)
            except Exception:
                self.buy()

    # Exit logic - use bear weights
    else:
        exit_score = self.compute_exit_score()
        # Exit threshold can be a fixed value or another parameter
        if exit_score >= 1.0:  # This threshold can be adjusted
            try:
                self.position.close()
            except Exception:
                try:
                    self.sell()
                except Exception:
                    pass
```

## Benefits of This Approach
1. **Logical Separation**: Clear distinction between bullish entry conditions and bearish exit conditions
2. **Flexible Weighting**: Different weight ranges for bull (0.7-0.9) and bear (1.1-1.3) conditions
3. **Backtest Compatibility**: Maintains compatibility with existing backtesting framework
4. **Optimization Ready**: New parameters can be optimized using the same framework

## Migration Path
1. Implement the changes in a new version of the file
2. Test with existing data to ensure compatibility
3. Run optimization to find optimal weight combinations
4. Compare performance with previous version