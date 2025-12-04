# Doji Implementation Analysis and Improvement Plan

## Current Issues Identified

### 1. Hardcoded Doji Weight
- **Current**: `weight_doji = 0.30` (line 211)
- **Issue**: Fixed value doesn't adapt to market conditions
- **Backtest Suggestion**: Optimal range is 0.3~0.4

### 2. Hardcoded Index for Doji Detection
- **Current**: `prior_idx = -6` (line 109)
- **Issue**: Always checks position -6, which may not be optimal for all market conditions
- **Problem**: This fixed position can miss relevant doji patterns or include irrelevant ones

### 3. Fixed Doji Body Fraction
- **Current**: `doji_body_frac = 0.20` (line 221)
- **Issue**: Static threshold doesn't adapt to volatility regimes

### 4. Binary Doji Bonus
- **Current**: Full `weight_doji` added when doji condition met
- **Issue**: No scaling based on doji quality or market context

## Detailed Analysis

### Current Doji Detection Logic (Lines 107-119)
```python
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
```

### Key Problems:
1. **Fixed Position**: Always uses index -6 regardless of market conditions
2. **Binary Decision**: Either adds full weight_doji or nothing
3. **No Quality Scaling**: All dojis treated equally regardless of body size
4. **No Volatility Adaptation**: Fixed 0.20 ATR fraction threshold
5. **Broad Exception Handling**: Silent failure on any error

## Improvement Recommendations

### 1. Dynamic Doji Weight Parameter

**Proposed Solution**: Make `weight_doji` adaptive based on market volatility

```python
# Replace fixed weight_doji with dynamic calculation
def get_dynamic_doji_weight(atr_cur, atr_ma, volatility_regime):
    """
    Calculate dynamic doji weight based on market conditions
    atr_cur: current ATR value
    atr_ma: moving average of ATR (e.g., 20-period)
    volatility_regime: 'low', 'normal', 'high'
    """
    # Base weight from backtest optimal range
    base_weight = 0.35  # Midpoint of 0.3-0.4 range

    # Adjust based on volatility regime
    if volatility_regime == 'high':
        # In high volatility, dojis are more significant
        weight = base_weight * 1.2
    elif volatility_regime == 'low':
        # In low volatility, dojis are less significant
        weight = base_weight * 0.8
    else:
        weight = base_weight

    # Constrain to optimal range
    return max(0.3, min(0.4, weight))
```

### 2. Adaptive Doji Position Selection

**Proposed Solution**: Find most relevant doji in recent history instead of fixed -6 position

```python
def find_best_doji_position(ha_open, ha_close, atr_cur, doji_body_frac, lookback=10):
    """
    Find the most significant doji in recent history
    Returns: (position, quality_score) where quality_score is 0-1
    """
    best_position = None
    best_quality = 0.0

    for i in range(1, min(lookback, len(ha_open))):
        idx = -i
        body_size = abs(ha_close[idx] - ha_open[idx])
        atr_fraction = body_size / atr_cur

        # Calculate doji quality (0 = perfect doji, 1 = not a doji)
        if atr_fraction < doji_body_frac:
            quality = 1.0 - (atr_fraction / doji_body_frac)  # 0-1 scale
            if quality > best_quality:
                best_quality = quality
                best_position = idx

    return best_position, best_quality
```

### 3. Dynamic Doji Body Fraction

**Proposed Solution**: Make doji threshold adaptive to volatility

```python
def get_dynamic_doji_fraction(atr_cur, atr_std, market_regime):
    """
    Calculate adaptive doji body fraction based on volatility
    """
    # Base fraction
    base_fraction = 0.20

    # Adjust based on ATR volatility
    if atr_std > 0:
        # More volatile markets need larger thresholds
        volatility_factor = atr_std / (atr_cur + 1e-6)
        base_fraction = base_fraction * (1.0 + volatility_factor * 0.5)

    # Constrain to reasonable range
    return max(0.15, min(0.25, base_fraction))
```

### 4. Quality-Based Doji Scoring

**Proposed Solution**: Scale doji contribution by quality

```python
def calculate_doji_contribution(ha_open, ha_close, atr_cur, doji_body_frac,
                              weight_doji, is_entry, lookback=10):
    """
    Calculate doji contribution with quality scaling
    """
    best_position, quality = find_best_doji_position(
        ha_open, ha_close, atr_cur, doji_body_frac, lookback
    )

    if best_position is not None and quality > 0:
        # Check direction for entry/exit
        if is_entry:
            if ha_close[best_position] < ha_open[best_position]:
                # Bearish doji for entry confirmation
                return weight_doji * quality
        else:
            if ha_close[best_position] > ha_open[best_position]:
                # Bullish doji for exit confirmation
                return weight_doji * quality

    return 0.0
```

## Complete Improved Implementation

### Updated `_compute_score_numba` Function

```python
@jit(nopython=True, cache=True)
def _compute_score_numba(ha_open, ha_close, atr_cur, weights, doji_body_frac,
                        weight_doji, is_entry=True):
    """Numba-optimized score calculation with improved doji detection."""
    score = 0.0
    lookback = 4

    # Main candle scoring (unchanged)
    for i in range(lookback):
        idx = -(i + 1)
        ha_o = ha_open[idx]
        ha_c = ha_close[idx]
        body = ha_c - ha_o

        if is_entry:
            if body > 0:  # green candle
                norm_body = body / atr_cur
                if norm_body > 3.0:
                    norm_body = 3.0
                score += weights[i] * norm_body
        else:
            body = ha_o - ha_c  # positive if red
            if body > 0:
                norm_body = body / atr_cur
                if norm_body > 3.0:
                    norm_body = 3.0
                score += weights[i] * norm_body

    # Improved Doji detection
    try:
        # Find best doji in recent history (last 10 candles)
        best_doji_pos = None
        best_doji_quality = 0.0

        for i in range(1, min(11, len(ha_open))):  # Look back up to 10 candles
            idx = -i
            prior_body = abs(ha_close[idx] - ha_open[idx])
            if prior_body < doji_body_frac * atr_cur:
                # Calculate quality: smaller body = higher quality
                quality = 1.0 - (prior_body / (doji_body_frac * atr_cur))
                if quality > best_doji_quality:
                    best_doji_quality = quality
                    best_doji_pos = idx

        # Add quality-scaled doji bonus if found
        if best_doji_pos is not None and best_doji_quality > 0:
            if is_entry:
                if ha_close[best_doji_pos] < ha_open[best_doji_pos]:
                    score += weight_doji * best_doji_quality
            else:
                if ha_close[best_doji_pos] > ha_open[best_doji_pos]:
                    score += weight_doji * best_doji_quality

    except Exception:
        # More specific error handling
        pass

    return score
```

## Strategy Parameter Updates

### Add to HeikinAshiWeightedStrategy class:

```python
# Add adaptive doji parameters
doji_lookback = 10  # Search for dojis in last 10 candles
doji_quality_scaling = True  # Enable quality-based doji scoring
```

### Update optimization parameters:

```python
# Add to optimization grid
doji_lookback = [8, 10, 12]  # Test different lookback periods
doji_body_frac = [0.15, 0.20, 0.25]  # Test different body fractions
```

## Implementation Plan

### Phase 1: Parameter Adaptation
1. Replace fixed `weight_doji = 0.30` with dynamic calculation
2. Add volatility regime detection
3. Implement adaptive doji body fraction

### Phase 2: Position Flexibility
1. Replace fixed `-6` position with adaptive search
2. Implement quality-based doji selection
3. Add lookback parameter for optimization

### Phase 3: Quality Scaling
1. Implement doji quality scoring
2. Replace binary bonus with quality-scaled contribution
3. Add quality threshold parameter

### Phase 4: Testing & Optimization
1. Run backtests with new parameter ranges
2. Validate performance improvements
3. Fine-tune adaptive algorithms

## Expected Benefits

1. **Better Market Adaptation**: Doji detection adapts to volatility regimes
2. **Improved Signal Quality**: Focus on most relevant dojis, not fixed positions
3. **Enhanced Performance**: Quality scaling provides more nuanced signals
4. **Robustness**: Better handles different market conditions
5. **Optimization Potential**: More parameters to fine-tune for specific assets

## Backward Compatibility

The changes maintain the same function signature and can be made backward compatible by:
- Keeping default parameters that replicate current behavior
- Adding new parameters with sensible defaults
- Ensuring optimization grids include current values

## Risk Mitigation

1. **Gradual Rollout**: Implement changes incrementally
2. **Extensive Testing**: Validate each component separately
3. **Fallback Mechanisms**: Maintain current logic as fallback
4. **Performance Monitoring**: Track impact on strategy metrics