# Heikin-Ashi Momentum Tracking Enhancement Plan

## Current Implementation Analysis

The current `_compute_score_numba` function calculates scores based on:
- 4 recent Heikin-Ashi candles (bars 1-4, with bar 1 being most recent)
- Each bar's body size is normalized by ATR and weighted by recency
- Fixed weights are applied to each bar position
- No momentum tracking or dynamic weight adjustment

## Problem Identification

The backtest failures indicate the system fails to track loss of momentum. The current implementation:
- Treats all bars equally in terms of momentum
- Doesn't account for accelerating/decelerating trends
- Misses opportunities to adjust weights based on relative bar sizes

## Proposed Solution

### Momentum Tracking Logic

For bars 3 and 4 (indices 2 and 3 in 0-based array), implement dynamic weight adjustments:

1. **Bonus Weight**: When bar 3 or 4 has size greater than all previous bars
   - Formula: `(size_of_bar - max(size_bar1, size_bar2)) * weight_bonus`
   - This captures accelerating momentum

2. **Penalty Weight**: When bar 3 or 4 has size less than minimum of bars 1 and 2
   - Formula: `(min(size_bar1, size_bar2) - size_of_bar) * weight_penalty`
   - This captures decelerating momentum

### Implementation Details

#### Modified `_compute_score_numba` Function

```python
@jit(nopython=True, cache=True)
def _compute_score_numba(ha_open, ha_close, atr_cur, weights, doji_weight, doji_body_frac,
                        weight_bonus=0.1, weight_penalty=0.05, is_entry=True):
    """Enhanced score calculation with momentum tracking for bars 3 and 4."""
    score = 0.0
    lookback = 4

    # Store bar sizes for momentum comparison
    bar_sizes = np.empty(lookback, dtype=np.float32)

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

        # Store normalized body size for momentum analysis
        bar_sizes[i] = abs(norm_body)

    # Momentum tracking for bars 3 and 4 (indices 2 and 3)
    for bar_idx in [2, 3]:  # bars 3 and 4 (0-based: 0=bar1, 1=bar2, 2=bar3, 3=bar4)
        current_size = bar_sizes[bar_idx]
        max_prev_size = max(bar_sizes[0], bar_sizes[1])  # max of bars 1 and 2
        min_prev_size = min(bar_sizes[0], bar_sizes[1])  # min of bars 1 and 2

        # Bonus for accelerating momentum
        if current_size > max_prev_size:
            bonus = (current_size - max_prev_size) * weight_bonus
            score += bonus

        # Penalty for decelerating momentum
        elif current_size < min_prev_size:
            penalty = (min_prev_size - current_size) * weight_penalty
            score -= penalty

    # Doji bonus logic (unchanged)
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

### Parameter Additions

Add two new parameters to the strategy class:
- `weight_bonus`: Controls the magnitude of momentum acceleration bonus (default: 0.1)
- `weight_penalty`: Controls the magnitude of momentum deceleration penalty (default: 0.05)

### Strategy Class Updates

1. Add new parameters to `HeikinAshiWeightedStrategy` class
2. Update `compute_entry_score()` and `compute_exit_score()` methods to pass new parameters
3. Include new parameters in optimization ranges

### Expected Benefits

1. **Better Trend Detection**: Accelerating trends get higher scores, improving entry timing
2. **Early Exit on Weakening Momentum**: Decelerating trends get lower scores, improving exit timing
3. **Reduced False Signals**: Penalty for weak bars reduces whipsaw trades
4. **Enhanced Risk Management**: Dynamic weighting adapts to changing market conditions

### Implementation Steps

1. ✅ Analyze current implementation
2. ✅ Design momentum tracking logic
3. ⏳ Modify `_compute_score_numba` function
4. ⏳ Update strategy class with new parameters
5. ⏳ Test with sample data
6. ⏳ Optimize parameter ranges
7. ⏳ Validate with backtesting

### Risk Considerations

- **Overfitting**: New parameters increase optimization space
- **Parameter Sensitivity**: Need careful tuning of bonus/penalty weights
- **Backward Compatibility**: Ensure existing functionality remains intact

### Validation Plan

1. Run backtests with new implementation
2. Compare performance metrics vs. original
3. Analyze specific trade cases where momentum tracking should help
4. Verify no regression in existing profitable scenarios