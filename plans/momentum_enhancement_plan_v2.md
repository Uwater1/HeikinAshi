# Heikin-Ashi Momentum Tracking Enhancement Plan (v2)

## Updated Based on User Requirements

### Key Changes from Original Plan:
1. **Separate bonus/penalty weights**: `weight_bull_bonus` and `weight_bear_penalty` (4 parameters total)
2. **Bar 4 penalty calculation**: Uses `min(size_bar1, size_bar2, size_bar3)` instead of just bars 1 and 2
3. **Performance optimization**: Focus on maintaining running speed

## Implementation Details

### Modified `_compute_score_numba` Function

```python
@jit(nopython=True, cache=True)
def _compute_score_numba(ha_open, ha_close, atr_cur, weights, doji_weight, doji_body_frac,
                        weight_bull_bonus=0.1, weight_bear_penalty=0.05,
                        is_entry=True):
    """Enhanced score calculation with momentum tracking for bars 3 and 4."""
    score = 0.0
    lookback = 4

    # Pre-allocate array for bar sizes (performance optimization)
    bar_sizes = np.empty(lookback, dtype=np.float32)

    # Process each bar and calculate base score
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

        # Store absolute normalized body size for momentum analysis
        bar_sizes[i] = abs(norm_body)

    # Momentum tracking for bars 3 and 4 (indices 2 and 3)
    # Use efficient array operations for performance
    if is_entry:
        # Bullish momentum tracking
        weight_bonus = weight_bull_bonus
        # For bar 3 (index 2)
        current_size = bar_sizes[2]
        max_prev_size = max(bar_sizes[0], bar_sizes[1])  # max of bars 1 and 2

        if current_size > max_prev_size:
            bonus = (current_size - max_prev_size) * weight_bonus
            score += bonus

        # For bar 4 (index 3)
        current_size = bar_sizes[3]
        # Use min of bars 1, 2, and 3 as requested
        min_prev_size = min(bar_sizes[0], bar_sizes[1], bar_sizes[2])

        if current_size > max(bar_sizes[0], bar_sizes[1], bar_sizes[2]):
            bonus = (current_size - max(bar_sizes[0], bar_sizes[1], bar_sizes[2])) * weight_bonus
            score += bonus
        elif current_size < min_prev_size:
            penalty = (min_prev_size - current_size) * weight_bear_penalty
            score -= penalty

    else:
        # Bearish momentum tracking
        weight_penalty = weight_bear_penalty
        # For bar 3 (index 2)
        current_size = bar_sizes[2]
        max_prev_size = max(bar_sizes[0], bar_sizes[1])  # max of bars 1 and 2

        if current_size > max_prev_size:
            bonus = (current_size - max_prev_size) * weight_penalty
            score += bonus

        # For bar 4 (index 3)
        current_size = bar_sizes[3]
        # Use min of bars 1, 2, and 3 as requested
        min_prev_size = min(bar_sizes[0], bar_sizes[1], bar_sizes[2])

        if current_size > max(bar_sizes[0], bar_sizes[1], bar_sizes[2]):
            bonus = (current_size - max(bar_sizes[0], bar_sizes[1], bar_sizes[2])) * weight_penalty
            score += bonus
        elif current_size < min_prev_size:
            penalty = (min_prev_size - current_size) * weight_penalty
            score -= penalty

    # Doji bonus logic (unchanged for performance)
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

### Parameter Additions to Strategy Class

```python
# Add to HeikinAshiWeightedStrategy class
weight_bull_bonus = 0.1    # Bonus weight for accelerating bullish momentum
weight_bear_penalty = 0.05 # Penalty weight for decelerating bearish momentum
```

### Performance Optimizations

1. **Pre-allocated arrays**: Use `np.empty()` instead of dynamic lists
2. **Efficient comparisons**: Use `max()`/`min()` functions instead of loops
3. **Conditional logic**: Only perform momentum calculations when needed
4. **Numba compatibility**: Maintain all operations as numba-optimizable

### Updated Implementation Steps

1. ✅ Analyze current implementation
2. ✅ Design momentum tracking logic with user requirements
3. ⏳ Modify `_compute_score_numba` function with new parameters
4. ⏳ Update strategy class with 4 new parameters
5. ⏳ Add parameters to optimization ranges
6. ⏳ Test with sample data for performance
7. ⏳ Validate backtesting results

### Parameter Optimization Ranges

```python
# Add to optimization parameters
weight_bull_bonus=[0.05, 0.1, 0.15, 0.2],
weight_bear_penalty=[0.025, 0.05, 0.075, 0.1],
```

### Expected Performance Impact

- **Minimal overhead**: Array operations are O(1) for fixed lookback=4
- **Numba optimization**: All operations remain numba-compatible
- **Memory efficient**: Pre-allocated arrays prevent dynamic memory allocation
- **Cache friendly**: Sequential array access patterns

### Validation Plan

1. Compare execution time before/after changes
2. Verify numba compilation succeeds
3. Test with various market conditions
4. Ensure no regression in existing functionality