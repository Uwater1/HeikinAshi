# Refined Doji Improvement Plan - Incorporating User Feedback

## Key Feedback Incorporated:
1. **Doji weight already in optimization**: `weight_doji` is already included in `bt.optimize`
2. **Lookback optimization suggestion**: Test lookback values `[2, 4, 6]`
3. **Focus on practical improvements**: Build on existing optimization framework

## Current State Analysis

### Existing Optimization Parameters (Lines 344, 358)
```python
weight_doji=[0.25, 0.3, 0.35, 0.4]  # Already optimized
```

### Current Hardcoded Issues
1. **Fixed doji position**: `prior_idx = -6` (line 109)
2. **No lookback flexibility**: Always checks only position -6
3. **Binary doji contribution**: Full weight or nothing

## Refined Improvement Plan

### 1. Adaptive Lookback Parameter

**Add to Strategy Class:**
```python
# Add flexible lookback parameter
doji_lookback = 6  # Default to current behavior
```

**Add to Optimization:**
```python
doji_lookback = [2, 4, 6, 8]  # Test different lookback periods
```

**Updated Doji Detection Logic:**
```python
# Replace fixed prior_idx = -6 with:
try:
    best_doji_score = 0.0
    best_doji_pos = None

    # Search through lookback period
    for i in range(1, min(self.doji_lookback + 1, len(ha_open))):
        idx = -i
        prior_body = abs(ha_close[idx] - ha_open[idx])

        if prior_body < doji_body_frac * atr_cur:
            # Calculate doji quality score (0-1, higher is better)
            quality = 1.0 - (prior_body / (doji_body_frac * atr_cur))

            # For entry: prefer bearish dojis (close < open)
            # For exit: prefer bullish dojis (close > open)
            if is_entry and ha_close[idx] < ha_open[idx]:
                doji_score = quality
            elif not is_entry and ha_close[idx] > ha_open[idx]:
                doji_score = quality
            else:
                doji_score = quality * 0.5  # Reduced score for wrong direction

            if doji_score > best_doji_score:
                best_doji_score = doji_score
                best_doji_pos = idx

    # Add scaled doji contribution
    if best_doji_pos is not None:
        score += weight_doji * best_doji_score

except Exception:
    pass  # Maintain current error handling
```

### 2. Enhanced Optimization Strategy

**Updated Optimization Parameters:**
```python
bt.optimize(
    # ... existing parameters ...
    doji_lookback=[2, 4, 6, 8],  # Test different lookback periods
    doji_body_frac=[0.15, 0.20, 0.25],  # Add body fraction optimization
    weight_doji=[0.25, 0.3, 0.35, 0.4],  # Keep existing weight optimization
    # ... other parameters ...
)
```

### 3. Quality-Based Doji Scoring

**Key Improvements:**
1. **Directional Preference**: Higher scores for dojis in the "right" direction
2. **Quality Scaling**: Better quality dojis contribute more
3. **Flexible Lookback**: Search optimal period instead of fixed -6

### 4. Implementation Phases

**Phase 1: Minimal Changes (Backward Compatible)**
```python
# Keep current behavior as default
doji_lookback = 6  # Default maintains current -6 position
doji_body_frac = 0.20  # Current value

# Add to optimization only
doji_lookback = [2, 4, 6]  # Start with user's suggested range
```

**Phase 2: Enhanced Logic**
```python
# Implement quality-based scoring within existing framework
# Maintain same function signature for compatibility
```

## Expected Benefits

1. **Better Signal Quality**: Focus on most relevant dojis in optimal lookback period
2. **Directional Confirmation**: Higher scores for dojis that confirm trend direction
3. **Optimization Potential**: More parameters to fine-tune for specific assets
4. **Backward Compatibility**: Default values maintain current behavior

## Code Changes Required

### In `HeikinAshiWeightedStrategy` class:
```python
# Add new parameters
doji_lookback = 6  # Default maintains current behavior
doji_body_frac = 0.20  # Make configurable

# Add to __init__ or class variables
```

### In `_compute_score_numba` function:
```python
# Replace lines 107-119 with enhanced logic
# Use doji_lookback parameter instead of hardcoded -6
# Implement quality-based scoring
```

### In optimization section:
```python
# Add doji_lookback and doji_body_frac to optimization grid
```

## Validation Approach

1. **Backward Compatibility Test**: Verify default values produce similar results
2. **Optimization Validation**: Run with new parameters to find optimal combinations
3. **Performance Comparison**: Compare new vs old approach across different assets
4. **Robustness Testing**: Validate behavior in different market conditions

## Risk Mitigation

- **Default Values**: Maintain current behavior as defaults
- **Incremental Testing**: Validate each change separately
- **Fallback Logic**: Preserve existing error handling
- **Performance Monitoring**: Track impact on key metrics