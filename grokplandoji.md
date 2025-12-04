## Detailed Implementation Plan for Asymmetric Doji Scoring

Based on your confirmation to implement the asymmetric system from dojiResearch.md, here's the comprehensive plan to replace the current binary doji logic with quality-based asymmetric scoring:

### Current State Analysis
- **Existing Doji Logic**: Simple binary check at position -6: if body < 0.20 * ATR, add weight_bull_doji (for entry on bearish doji) or weight_bear_doji (for exit on bullish doji)
- **Hardcoded Parameters**: doji_body_frac = 0.20, fixed position -6
- **Symmetric Weights**: weight_bull_doji and weight_bear_doji treated similarly

### Proposed Asymmetric Implementation

#### 1. New Quality Scoring Functions
Implement three core functions as numba-compatible helpers:

**Volatility-Normalized Body Quality**:
```python
@jit(nopython=True, cache=True)
def get_volatility_normalized_score(body_size, atr_value, sensitivity_factor=0.1):
    """Returns 0.0-1.0 score based on body size relative to ATR."""
    max_allowed_body = atr_value * sensitivity_factor
    if max_allowed_body == 0 or body_size > max_allowed_body:
        return 0.0
    return 1.0 - (body_size / max_allowed_body)
```

**Bearish Rollover Quality** (adapted for numba):
```python
@jit(nopython=True, cache=True)
def get_bear_rollover_quality_numba(ha_open, ha_close, ha_low, atr_cur, lookback=3, idx=0):
    """Detects momentum decay at tops. Returns 0.0-1.0."""
    # Body shrinkage detection
    recent_bodies = np.empty(lookback)
    for i in range(lookback):
        pos = idx - i - 1
        if pos >= 0:
            recent_bodies[i] = abs(ha_close[pos] - ha_open[pos])
        else:
            recent_bodies[i] = 0.0
    
    avg_recent_body = np.mean(recent_bodies) if np.any(recent_bodies > 0) else 1.0
    current_body = abs(ha_close[idx] - ha_open[idx])
    
    momentum_score = 0.0
    if current_body < avg_recent_body:
        momentum_score = 1.0 - (current_body / avg_recent_body)
    
    # Shadow emergence (lower shadow on green candle)
    shadow_penalty = 0.0
    if ha_close[idx] > ha_open[idx]:  # Green candle
        lower_shadow = ha_open[idx] - ha_low[idx]
        if lower_shadow > (current_body * 0.2):
            shadow_penalty = 0.5
    
    return min(1.0, (momentum_score * 0.7) + (shadow_penalty * 0.3))
```

**Bullish Rejection Quality** (adapted for numba):
```python
@jit(nopython=True, cache=True)  
def get_bull_rejection_quality_numba(ha_open, ha_close, ha_high, ha_low, low_prices, atr_cur, support_lookback=20, idx=0):
    """Detects sharp rejection near support. Returns 0.0-1.0."""
    ha_o, ha_c, ha_l = ha_open[idx], ha_close[idx], ha_low[idx]
    
    # Hammer shape detection
    body_size = abs(ha_c - ha_o)
    lower_shadow = min(ha_o, ha_c) - ha_l
    shadow_score = min(1.0, lower_shadow / (atr_cur * 0.5))
    
    # Support proximity
    recent_lows = low_prices[max(0, idx-support_lookback):idx+1]
    support_level = np.min(recent_lows) if len(recent_lows) > 0 else ha_l
    dist_from_support = ha_l - support_level
    tolerance = atr_cur * 0.5
    support_score = 0.0
    if dist_from_support < tolerance:
        support_score = 1.0 - (dist_from_support / tolerance)
    
    return (shadow_score * 0.6) + (support_score * 0.4)
```

#### 2. Strategy Parameter Updates
Replace current doji parameters with research-based ones:

```python
# Replace these:
weight_bull_doji = 0.4
weight_bear_doji = 0.3
doji_body_frac = 0.20

# With these:
weight_bear_rollover = 0.25    # Lower magnitude for slow tops
weight_bull_rejection = 0.45   # Higher magnitude for fast bottoms
sensitivity_factor = 0.10       # Max body as fraction of ATR
support_lookback = 20           # Periods to look back for support
bear_lookback = 3               # Periods for bear momentum calculation
doji_threshold = 0.4            # Minimum quality score to trigger
```

#### 3. Modified Score Computation
Update `_compute_score_numba` to accept pre-computed quality scores:

```python
@jit(nopython=True, cache=True)
def _compute_score_numba(ha_open, ha_close, atr_cur, weights, 
                        bear_rollover_quality, bull_rejection_quality,
                        weight_bear_rollover, weight_bull_rejection, 
                        doji_threshold, is_entry=True):
    # ... existing score calculation ...
    
    # Replace binary doji logic with quality-based asymmetric scoring
    if is_entry and bear_rollover_quality > doji_threshold:
        score -= weight_bear_rollover * bear_rollover_quality
    elif not is_entry and bull_rejection_quality > doji_threshold:
        score += weight_bull_rejection * bull_rejection_quality
    
    return score
```

#### 4. Strategy Class Updates
- Add quality computation in `next()` method before calling score functions
- Pass computed qualities to `_compute_score_numba`
- Update parameter initialization and scaling

#### 5. Optimization Ranges
```python
bt.optimize(
    # ... existing params ...
    weight_bear_rollover=(15, 35),    # 0.15-0.35
    weight_bull_rejection=(35, 55),   # 0.35-0.55  
    sensitivity_factor=(0.05, 0.20),  # 5%-20% of ATR
    support_lookback=(10, 30),        # 10-30 periods
    bear_lookback=(2, 5),             # 2-5 periods
    doji_threshold=(0.3, 0.6),        # 0.3-0.6
    # ... other params ...
)
```

### Key Benefits
1. **Asymmetric Market Behavior**: Separate logic for slow tops vs fast bottoms
2. **Quality-Based Scoring**: Eliminates binary optimization cliffs
3. **Volatility Adaptation**: ATR-normalized thresholds
4. **Pattern Recognition**: Hammer and rollover detection
5. **Better Optimization Surface**: Continuous scores improve parameter search

### Implementation Phases
1. **Phase 1**: Add new parameters and helper functions
2. **Phase 2**: Implement quality computation in strategy
3. **Phase 3**: Replace binary logic with quality scoring
4. **Phase 4**: Update optimization and validate results

Are you pleased with this implementation plan? If yes, I can switch to code mode to begin implementation. If you'd like any modifications, please let me know.