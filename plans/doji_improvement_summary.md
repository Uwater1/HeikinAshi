# Doji Improvement Summary - Key Recommendations

## Current Issues
1. **Hardcoded doji_weight**: Fixed at 0.30, should be in optimal range 0.3~0.4
2. **Fixed position detection**: Always checks index -6, missing better opportunities
3. **Binary doji contribution**: Full weight or nothing, no quality scaling
4. **Static doji threshold**: Fixed 0.20 ATR fraction doesn't adapt to volatility

## Key Improvements Needed

### 1. Dynamic Doji Weight Parameter
```python
# Replace fixed weight_doji = 0.30 with:
weight_doji = 0.35  # Start with midpoint of optimal range
# Add to optimization: weight_doji=[0.3, 0.35, 0.4]
```

### 2. Adaptive Doji Position Selection
```python
# Replace fixed prior_idx = -6 with:
# Search last 10 candles for best quality doji
for i in range(1, min(11, len(ha_open))):
    idx = -i
    # Find doji with smallest body relative to ATR
```

### 3. Quality-Based Doji Scoring
```python
# Replace binary bonus with quality scaling:
if best_doji_found:
    quality = 1.0 - (doji_body_size / (doji_body_frac * atr_cur))
    score += weight_doji * quality  # Scale by quality (0-1)
```

### 4. Dynamic Doji Body Fraction
```python
# Make doji threshold adaptive:
doji_body_frac = 0.20  # Keep current but add to optimization
# Add to optimization: doji_body_frac=[0.15, 0.20, 0.25]
```

## Implementation Priority

1. **Quick Win**: Update weight_doji to 0.35 and add to optimization range
2. **Medium Effort**: Implement adaptive position selection (last 10 candles)
3. **Advanced**: Add quality-based scaling for more nuanced signals

## Expected Impact
- Better adaptation to different market conditions
- More relevant doji pattern detection
- Improved strategy performance through quality scaling
- Enhanced optimization potential

## Backward Compatibility
All changes maintain same function signatures and can be made backward compatible with sensible defaults.