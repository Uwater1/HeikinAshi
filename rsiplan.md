# Comprehensive RSI Implementation Plan

## Phase 1: Understanding Current Architecture

### Current Strategy Flow
1. **Initialization (`init`)**: 
   - Computes ATR indicator using Numba-optimized functions
   - Registers Heikin-Ashi OHLC as indicators
   - Pre-computes weight arrays for bull/bear scoring

2. **Scoring System (`_compute_score_numba`)**:
   - Looks back 4 bars, filters by color (green for entry, red for exit)
   - Normalizes bar sizes by ATR
   - Applies recency weights (more recent = higher weight)
   - Adds momentum bonuses/penalties for bars 3-4
   - Adds doji pattern bonus at prior_idx (-6 by default)

3. **Entry/Exit Logic (`next`)**:
   - Entry: Bull score ≥ 1.0 triggers buy with stop loss
   - Exit: Bear score ≥ 1.0 triggers position close

### Key Design Patterns
- **Numba optimization** for hot loops (scoring, TR, HA calculations)
- **Weight arrays pre-computed** in init() for performance
- **Separate entry/exit systems** with different weight parameters
- **Score accumulation** model where multiple signals add to threshold

---

## Phase 2: RSI Implementation Strategy

### User said:
when we set RSI to be large (for ep RSI 50) then its a better indicator for bottom. So make a comprehensive plan on how to implement this. Here's the basic: Use the rsi in pandas-ta with length subject to bt.optimization. 2. as RSI large, 30 70 are no longer the optimum, so bt.optimize it as well. 3. Each time, lookback 8 bars and find the minimum (for entry) / maximum (for exit) of RSI x, then subtract this to overbought / oversold threathhold. Then times this value with rsi_bull_weight or rsi_bear_weight, add to the score

### Conceptual Overview
Your RSI approach adds a **divergence-based signal**:
- **Entry**: When RSI dips low (oversold region), it indicates potential bottom → bullish
- **Exit**: When RSI peaks high (overbought region), it indicates potential top → bearish

The innovation is making thresholds **adaptive** rather than fixed (30/70):
- Look back 8 bars to find min/max RSI
- Calculate **distance from threshold** as the signal strength
- Weight and add to existing score

---

## Phase 3: Detailed Implementation Plan

### 3.1 Add RSI Indicator (in `init`)

```python
def init(self):
    # ... existing code ...
    
    # RSI indicator
    self.rsi = self.I(lambda: indicator_rsi(
        self.data.Close,
        self.rsi_period
    ))
```

**New helper function** (add after `indicator_atr`):
```python
def indicator_rsi(close, length=14):
    """Compute RSI using pandas-ta."""
    close_series = pd.Series(close)
    rsi = ta.rsi(close_series, length=length)
    return rsi.to_numpy()
```

**Why pandas-ta?** 
- Already imported and used for HA calculation
- Well-tested implementation
- No need for custom Numba optimization (RSI computed once per backtest)

---

### 3.2 Add RSI Parameters to Strategy Class

```python
class HeikinAshiWeightedStrategy(Strategy):
    # ... existing parameters ...
    
    # RSI parameters
    rsi_period = 14                    # RSI calculation window
    rsi_lookback = 8                   # Bars to look back for min/max
    rsi_bull_threshold = 30            # Oversold threshold for entry
    rsi_bear_threshold = 70            # Overbought threshold for exit
    weight_rsi_bull = 0.15             # Weight for RSI entry signal
    weight_rsi_bear = 0.15             # Weight for RSI exit signal
```

**Optimization ranges** (add to `bt.optimize` call):
```python
rsi_period=(8, 21),              # 8-21 period RSI
rsi_lookback=(4, 12),            # 4-12 bars lookback
rsi_bull_threshold=(20, 40),     # 20-40 oversold (scaled by 100)
rsi_bear_threshold=(60, 80),     # 60-80 overbought (scaled by 100)
weight_rsi_bull=(5, 25),         # 0.05-0.25 weight (scaled by 100)
weight_rsi_bear=(5, 25),         # 0.05-0.25 weight (scaled by 100)
```

**Scale in init()**:
```python
def init(self):
    # ... existing scaling ...
    
    self.rsi_bull_threshold = self.rsi_bull_threshold / 100.0
    self.rsi_bear_threshold = self.rsi_bear_threshold / 100.0
    self.weight_rsi_bull = self.weight_rsi_bull / 100.0
    self.weight_rsi_bear = self.weight_rsi_bear / 100.0
```

---

### 3.3 Create RSI Scoring Functions

**Option A: Pure Python (Simpler, Fast Enough)**
```python
def compute_rsi_entry_score(self):
    """Compute RSI-based entry score (oversold signal)."""
    if len(self.rsi) < self.rsi_lookback + 1:
        return 0.0
    
    # Get last rsi_lookback bars of RSI
    lookback_window = self.rsi[-(self.rsi_lookback + 1):-1]
    min_rsi = np.min(lookback_window)
    
    # Distance below threshold = bullish signal
    # More oversold = higher score
    distance = self.rsi_bull_threshold - min_rsi
    
    if distance > 0:
        return distance * self.weight_rsi_bull
    return 0.0

def compute_rsi_exit_score(self):
    """Compute RSI-based exit score (overbought signal)."""
    if len(self.rsi) < self.rsi_lookback + 1:
        return 0.0
    
    # Get last rsi_lookback bars of RSI
    lookback_window = self.rsi[-(self.rsi_lookback + 1):-1]
    max_rsi = np.max(lookback_window)
    
    # Distance above threshold = bearish signal
    # More overbought = higher score
    distance = max_rsi - self.rsi_bear_threshold
    
    if distance > 0:
        return distance * self.weight_rsi_bear
    return 0.0
```

**Option B: Numba-Optimized (If Performance Critical)**
```python
@jit(nopython=True, cache=True)
def _compute_rsi_score_numba(rsi_arr, lookback, threshold, weight, is_entry):
    """
    Numba-optimized RSI score calculation.
    
    Args:
        rsi_arr: Full RSI array
        lookback: Number of bars to look back
        threshold: Oversold (entry) or overbought (exit) level
        weight: Multiplier for signal strength
        is_entry: True for entry (look for min), False for exit (look for max)
    
    Returns:
        Weighted RSI score contribution
    """
    n = len(rsi_arr)
    if n < lookback + 1:
        return 0.0
    
    # Extract lookback window (exclude current bar)
    window = rsi_arr[-(lookback + 1):-1]
    
    if is_entry:
        # Entry: find minimum RSI, check if below threshold
        extreme_rsi = np.min(window)
        distance = threshold - extreme_rsi
    else:
        # Exit: find maximum RSI, check if above threshold
        extreme_rsi = np.max(window)
        distance = extreme_rsi - threshold
    
    if distance > 0:
        return distance * weight
    return 0.0
```

**Usage in strategy**:
```python
def compute_rsi_entry_score(self):
    """Compute RSI-based entry score."""
    return float(_compute_rsi_score_numba(
        self.rsi,
        self.rsi_lookback,
        np.float32(self.rsi_bull_threshold),
        np.float32(self.weight_rsi_bull),
        is_entry=True
    ))

def compute_rsi_exit_score(self):
    """Compute RSI-based exit score."""
    return float(_compute_rsi_score_numba(
        self.rsi,
        self.rsi_lookback,
        np.float32(self.rsi_bear_threshold),
        np.float32(self.weight_rsi_bear),
        is_entry=False
    ))
```

---

### 3.4 Integrate RSI into Entry/Exit Logic

**Modify `next()` method**:
```python
def next(self):
    """Execute strategy logic on each bar."""
    price = float(self.data.Close[-1])
    atr_cur = float(self.atr[-1] if len(self.atr) > 0 else 0.0)

    # Entry logic
    if not self.position:
        ha_entry_score = self.compute_entry_score(price)
        rsi_entry_score = self.compute_rsi_entry_score()
        total_entry_score = ha_entry_score + rsi_entry_score
        
        if total_entry_score >= 1.0:
            sl_price = max(price * 0.95, price - self.stop_atr_mult * atr_cur) if atr_cur > 0 else price * 0.97
            try:
                self.buy(sl=sl_price)
            except Exception:
                self.buy()

    # Exit logic
    else:
        ha_exit_score = self.compute_exit_score(price)
        rsi_exit_score = self.compute_rsi_exit_score()
        total_exit_score = ha_exit_score + rsi_exit_score
        
        if total_exit_score >= 1.0:
            try:
                self.position.close()
            except Exception:
                try:
                    self.sell()
                except Exception:
                    pass
```

---

### 3.5 Update Optimization Parameters

**In `run()` function**, add to `bt.optimize()`:
```python
stats, heatmap, optimize_result = bt.optimize(
    # ... existing parameters ...
    
    # RSI parameters
    rsi_period=(8, 21),              # RSI window
    rsi_lookback=(4, 12),            # Lookback for min/max
    rsi_bull_threshold=(20, 40),     # Oversold threshold (20-40)
    rsi_bear_threshold=(60, 80),     # Overbought threshold (60-80)
    weight_rsi_bull=(5, 25),         # Bull weight (0.05-0.25)
    weight_rsi_bear=(5, 25),         # Bear weight (0.05-0.25)
    
    # ... rest of parameters ...
)
```

---

### 3.6 Update Best Parameters Display

**In `run()` function after optimization**:
```python
print("\n--- Best Parameters ---")
st = stats._strategy
print(f"  atr_period: {st.atr_period}")
# ... existing parameter prints ...

# Add RSI parameters
print(f"\n  RSI Configuration:")
print(f"    rsi_period: {st.rsi_period}")
print(f"    rsi_lookback: {st.rsi_lookback}")
print(f"    rsi_bull_threshold: {st.rsi_bull_threshold:.2f} | rsi_bear_threshold: {st.rsi_bear_threshold:.2f}")
print(f"    weight_rsi_bull: {st.weight_rsi_bull:.3f} | weight_rsi_bear: {st.weight_rsi_bear:.3f}")
```

---

## Phase 4: Testing & Validation Strategy

### 4.1 Unit Testing
```python
# Test RSI calculation matches expected values
def test_rsi_calculation():
    close_prices = np.array([44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64])
    rsi = indicator_rsi(close_prices, length=14)
    assert len(rsi) == len(close_prices)
    assert 0 <= rsi[-1] <= 100

# Test score calculation logic
def test_rsi_scoring():
    # Mock RSI array with known min/max
    rsi_arr = np.array([50, 45, 40, 35, 30, 25, 30, 35, 40])  # Min at index 5 = 25
    score = _compute_rsi_score_numba(rsi_arr, lookback=8, threshold=30.0, weight=0.15, is_entry=True)
    expected = (30.0 - 25.0) * 0.15  # = 0.75
    assert abs(score - expected) < 0.01
```

### 4.2 Integration Testing
1. **Baseline run**: Run without RSI (set weights to 0) to verify no regression
2. **RSI-only run**: Set HA weights to 0, only use RSI to verify signal generation
3. **Combined run**: Full optimization with both systems

### 4.3 Performance Validation
- **Before/after timing**: Measure `next()` execution time
- **Expected impact**: <5% slowdown (RSI computed once, scoring is simple)
- **Memory check**: Ensure RSI array doesn't cause issues

---

## Phase 5: Optimization Considerations

### Search Space Impact
**Current parameters**: ~18 dimensions  
**With RSI**: ~24 dimensions

**Implications**:
- SAMBO can handle this (designed for 20-100 dimensions)
- May need to increase `max_tries` from 10,000 → 15,000-20,000
- Consider two-stage optimization:
  1. Coarse grid for RSI parameters
  2. Fine-tune all parameters together

### Parameter Interactions to Watch
1. **RSI period vs ATR period**: Longer RSI may need longer ATR
2. **RSI weight vs HA weights**: May compete or complement
3. **Lookback vs RSI period**: Lookback should be ≤ RSI period for stability

---

## Phase 6: Alternative Designs (Future Enhancements) (Not implemented yet)

### Option 1: RSI Divergence Detection
Instead of absolute levels, detect divergence:
- Price makes lower low, RSI makes higher low → bullish divergence
- More sophisticated but requires more parameters

### Option 2: RSI Slope
Check if RSI is turning up (entry) or down (exit):
```python
rsi_slope = self.rsi[-1] - self.rsi[-2]
if rsi_slope > 0:  # RSI turning up
    score += rsi_slope * weight
```

### Option 3: Multi-Timeframe RSI
Compute RSI on different periods, combine signals:
- Fast RSI (7-period) for immediate turning points
- Slow RSI (21-period) for trend confirmation

---

## Phase 7: Implementation Checklist

### Code Changes
- [ ] Add `indicator_rsi()` function after `indicator_atr()`
- [ ] Add RSI parameters to `HeikinAshiWeightedStrategy` class
- [ ] Add RSI indicator registration in `init()`
- [ ] Add parameter scaling for RSI in `init()`
- [ ] Create `compute_rsi_entry_score()` method
- [ ] Create `compute_rsi_exit_score()` method
- [ ] Create `_compute_rsi_score_numba()` if using Numba
- [ ] Modify `next()` to include RSI scores
- [ ] Update `bt.optimize()` parameter ranges
- [ ] Update best parameters display

### Testing
- [ ] Run baseline test (RSI weights = 0)
- [ ] Run RSI-only test (HA weights = 0)
- [ ] Run full optimization
- [ ] Compare results to current version
- [ ] Profile performance impact

### Documentation
- [ ] Comment RSI logic clearly
- [ ] Document threshold reasoning
- [ ] Add usage examples
- [ ] Update performance report with RSI impact

---

## Phase 8: Expected Behavior & Tuning

### Signal Interpretation
**Entry (Bullish RSI)**:
- RSI lookback min = 25, threshold = 30 → distance = 5 → score += 5 × 0.15 = 0.75
- Combined with HA score ≥ 0.25 → triggers entry

**Exit (Bearish RSI)**:
- RSI lookback max = 75, threshold = 70 → distance = 5 → score += 5 × 0.15 = 0.75
- Combined with HA score ≥ 0.25 → triggers exit

### Tuning Strategy
1. **Start conservative**: Low RSI weights (0.05-0.10)
2. **Observe interaction**: Check if RSI helps or hurts win rate
3. **Adjust thresholds**: If RSI rarely fires, widen ranges
4. **Balance weights**: If RSI dominates, reduce weights

---

## Recommendation: Start with Option A (Pure Python)

**Rationale**:
1. RSI score calculation is simple (min/max + arithmetic)
2. Only runs once per bar in `next()` (not in hot loop)
3. Easier to debug and validate
4. Can always optimize to Numba later if profiling shows bottleneck

**If performance becomes an issue** (unlikely), upgrade to Option B with Numba.

---

## Summary

This plan adds RSI as a **complementary signal** to the existing Heikin-Ashi momentum system:

- **HA system**: Captures bar-by-bar momentum trends
- **RSI system**: Captures mean-reversion / overbought-oversold conditions
- **Combined**: More robust entry/exit decisions

The adaptive threshold approach (finding min/max over lookback) is clever because:
- Traditional RSI 30/70 fails with higher periods (RSI 50 rarely reaches 30)
- This dynamically adjusts to recent price action
- Distance metric allows for graduated signal strength (not just binary)

**Next step**: Implement Phase 3 (core functionality), test with Phase 4, then optimize with Phase 5.