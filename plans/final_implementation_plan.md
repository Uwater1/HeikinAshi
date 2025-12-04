# Final Implementation Plan: Weighted Bull/Bear System

## Current Understanding
The current system uses:
- `weight_1`, `weight_2`, `weight_3`, `weight_4` = [0.15, 0.20, 0.25, 0.30]
- `weight_doji` = 0.30
- Entry/Exit thresholds = 1.0
- Entry score = weight_doji + weight_1*size_1 + weight_2*size_2 + weight_3*size_3 + weight_4*size_4
- Trigger when score ≥ threshold (currently both thresholds = 1.0)

## Required Changes
Replace with separate bull/bear weighted systems:

### 1. New Parameters
```python
# Bull weights for entry (scaled by 0.7-0.9)
weight_bull_1 = 0.15 * 0.7  # 0.105
weight_bull_2 = 0.20 * 0.7  # 0.140
weight_bull_3 = 0.25 * 0.7  # 0.175
weight_bull_4 = 0.30 * 0.7  # 0.210
weight_bull_doji = 0.30 * 0.7  # 0.210

# Bear weights for exit (scaled by 1.1-1.3)
weight_bear_1 = 0.15 * 1.1  # 0.165
weight_bear_2 = 0.20 * 1.1  # 0.220
weight_bear_3 = 0.25 * 1.1  # 0.275
weight_bear_4 = 0.30 * 1.1  # 0.330
weight_bear_doji = 0.30 * 1.1  # 0.330
```

### 2. Modified _compute_score_numba Function
```python
@jit(nopython=True, cache=True)
def _compute_score_numba(ha_open, ha_close, atr_cur, weights, doji_weight, doji_body_frac, is_entry=True):
    """Numba-optimized score calculation with separate weights for bull/bear."""
    score = 0.0
    lookback = 4

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
                    score += doji_weight
            else:
                if ha_close[prior_idx] > ha_open[prior_idx]:
                    score += doji_weight
    except:
        pass

    return score
```

### 3. Updated Strategy Methods
```python
def compute_entry_score(self):
    """Compute weighted entry score using bull weights."""
    atr_cur = float(self.atr[-1] if len(self.atr) > 0 else 1.0)
    if atr_cur <= 0:
        atr_cur = 1.0

    bull_weights = np.array([self.weight_bull_1, self.weight_bull_2, self.weight_bull_3, self.weight_bull_4], dtype=np.float32)
    return float(_compute_score_numba(
        self.ha_open, self.ha_close, np.float32(atr_cur), bull_weights,
        np.float32(self.weight_bull_doji), np.float32(self.doji_body_frac),
        is_entry=True
    ))

def compute_exit_score(self):
    """Compute weighted exit score using bear weights."""
    atr_cur = float(self.atr[-1] if len(self.atr) > 0 else 1.0)
    if atr_cur <= 0:
        atr_cur = 1.0

    bear_weights = np.array([self.weight_bear_1, self.weight_bear_2, self.weight_bear_3, self.weight_bear_4], dtype=np.float32)
    return float(_compute_score_numba(
        self.ha_open, self.ha_close, np.float32(atr_cur), bear_weights,
        np.float32(self.weight_bear_doji), np.float32(self.doji_body_frac),
        is_entry=False
    ))
```

### 4. Updated Entry/Exit Logic
```python
def next(self):
    """Execute strategy logic on each bar."""
    price = float(self.data.Close[-1])
    atr_cur = float(self.atr[-1] if len(self.atr) > 0 else 0.0)

    # Entry logic - use bull weights, trigger when score >= 1.0
    if not self.position:
        entry_score = self.compute_entry_score()
        if entry_score >= 1.0:  # Fixed threshold
            sl_price = max(price * 0.95, price - self.stop_atr_mult * atr_cur) if atr_cur > 0 else price * 0.97
            try:
                self.buy(sl=sl_price)
            except Exception:
                self.buy()

    # Exit logic - use bear weights, trigger when score >= 1.0
    else:
        exit_score = self.compute_exit_score()
        if exit_score >= 1.0:  # Fixed threshold
            try:
                self.position.close()
            except Exception:
                try:
                    self.sell()
                except Exception:
                    pass
```

### 5. Optimization Parameters
```python
stats, heatmap = bt.optimize(
    atr_period = [10, 14, 20, 30],
    weight_bull_1=[0.105, 0.140, 0.175, 0.210],
    weight_bull_2=[0.105, 0.140, 0.175, 0.210],
    weight_bull_3=[0.105, 0.140, 0.175, 0.210],
    weight_bull_4=[0.105, 0.140, 0.175, 0.210],
    weight_bull_doji=[0.210, 0.250, 0.280, 0.300],
    weight_bear_1=[0.165, 0.220, 0.275, 0.330],
    weight_bear_2=[0.165, 0.220, 0.275, 0.330],
    weight_bear_3=[0.165, 0.220, 0.275, 0.330],
    weight_bear_4=[0.165, 0.220, 0.275, 0.330],
    weight_bear_doji=[0.330, 0.360, 0.390, 0.420],
    stop_atr_mult=[1.3, 1.5, 2.0],
    maximize='Return [%]',
    return_heatmap=True
)
```

## Key Benefits
1. **Logical Separation**: Clear distinction between bullish entry and bearish exit conditions
2. **Proper Scaling**: Weights are scaled appropriately for bull (0.7-0.9) and bear (1.1-1.3) conditions
3. **Consistent Threshold**: Both entry and exit use the same threshold (1.0) but with different weighted systems
4. **Backward Compatibility**: Maintains the same scoring logic but with proper weight separation

## Implementation Approach
1. Create a new version of the file with the updated logic
2. Test with existing data to ensure the scoring works correctly
3. Run optimization to find optimal weight combinations
4. Compare performance with the previous version