# Doji Implementation Plan v1.0 - Asymmetric Momentum-Based Detection

## Executive Summary

Building on the existing momentum tracking system in heikinAshi.py, this plan implements asymmetric doji detection that focuses on **loss of bullish momentum** for bearish signals and **sharp rejection patterns** for bullish signals. The approach moves beyond binary detection to quality-based scoring that incorporates market asymmetry.

## Current Implementation Analysis

### Existing Momentum System
The current code already includes sophisticated momentum tracking:
- **Bonus/Penalty Logic**: Bars 3 and 4 get bonuses for acceleration, penalties for deceleration
- **Size Comparison**: Current bar size vs. previous bars determines momentum strength
- **Separate Bull/Bear Weights**: Different bonus/penalty weights for entry vs exit

### Current Doji Logic (Lines 156-168)
```python
# Simple binary check at fixed position -6
prior_idx = -6
prior_body = abs(ha_close[prior_idx] - ha_open[prior_idx])
if prior_body < doji_body_frac * atr_cur:
    if is_entry and ha_close[prior_idx] < ha_open[prior_idx]:
        score += doji_weight  # Bearish doji for entry
    elif not is_entry and ha_close[prior_idx] > ha_open[prior_idx]:
        score += doji_weight  # Bullish doji for exit
```

**Problems:**
- Fixed position (-6) ignores momentum context
- Binary logic creates optimization cliffs
- No distinction between momentum loss vs. rejection patterns
- Symmetric treatment of asymmetric market behavior

## Asymmetric Doji Detection Strategy

### Core Principle: Context-Specific Detection
- **Exit Signals (is_entry = False)**: Detect loss of growing momentum through body shrinkage and red candle emergence in bullish sequences
- **Entry Signals (is_entry = True)**: Detect hammer patterns with sharp rejection at support levels
- **Quality-Based Scoring**: Both use 0.0-1.0 continuous scoring instead of binary detection

### 1. Bearish Doji Detection (Loss of Bull Momentum) - For Exit Only

**Context**: Used only when `is_entry = False` to detect weakening bullish momentum before reversal

**Focus**: Analyze sequences like `[g_1, g_2, r_1, g_3]` - the part before consecutive red bars that the momentum system already handles

**Key Indicators:**
1. **Body Shrinkage in Bullish Sequence**: Green candles getting progressively smaller
2. **Red Candle Emergence**: Red bars appearing within green-dominated sequences
3. **Low Upper Shadow**: Weak bullish conviction
4. **Momentum Decay**: Current green body smaller than recent green average

**Implementation Concept:**
```python
def get_bearish_momentum_loss_quality(ha_open, ha_close, ha_high, ha_low, lookback=5, idx=0):
    """
    Detects loss of growing momentum in bullish sequences.
    Only used for exit signals (is_entry = False).
    Returns 0.0-1.0 quality score.
    """
    # 1. Analyze bullish sequence before current momentum system
    bullish_sequence = []
    red_emergence_count = 0

    for i in range(lookback):
        pos = idx - i
        if pos < 0:
            continue

        is_green = ha_close[pos] > ha_open[pos]
        body_size = abs(ha_close[pos] - ha_open[pos])

        if is_green:
            bullish_sequence.append(body_size)
        else:
            red_emergence_count += 1

    if len(bullish_sequence) < 2:
        return 0.0

    # 2. Body shrinkage calculation
    current_body = bullish_sequence[0]  # Most recent green body
    avg_prior_bodies = sum(bullish_sequence[1:]) / len(bullish_sequence[1:])

    shrinkage_score = 0.0
    if current_body < avg_prior_bodies:
        shrinkage_score = 1.0 - (current_body / avg_prior_bodies)

    # 3. Red emergence penalty (red bars in bullish context)
    red_penalty = min(1.0, red_emergence_count * 0.25)

    # 4. Upper shadow analysis (weak bullish conviction)
    upper_shadow_penalty = 0.0
    if ha_close[idx] > ha_open[idx]:  # Current is green
        body_size = ha_close[idx] - ha_open[idx]
        upper_shadow = ha_high[idx] - ha_close[idx]
        if upper_shadow > body_size * 0.3:  # Significant upper shadow
            upper_shadow_penalty = 0.3

    # Combined score: shrinkage is primary signal
    return min(1.0, (shrinkage_score * 0.5) + (red_penalty * 0.3) + (upper_shadow_penalty * 0.2))
```

### 2. Bullish Doji Detection (Hammer Rejection) - For Entry Only

**Context**: Used only when `is_entry = True` to detect sharp rejection patterns

**Focus**: Hammer shapes with long lower shadows, especially near support levels

**Key Indicators:**
1. **Hammer Pattern**: Long lower shadow with small body and upper shadow
2. **Support Proximity**: Proximity to recent lows for double bottom potential
3. **Rejection Intensity**: Lower shadow length relative to ATR
4. **Body Smallness**: Small body relative to total range

**Implementation Concept:**
```python
def get_bullish_hammer_quality(ha_open, ha_close, ha_high, ha_low, low_prices, atr_cur, support_lookback=15, idx=0):
    """
    Detects hammer patterns with sharp rejection at support.
    Only used for entry signals (is_entry = True).
    Returns 0.0-1.0 quality score.
    """
    ha_o, ha_c, ha_h, ha_l = ha_open[idx], ha_close[idx], ha_high[idx], ha_low[idx]

    body_size = abs(ha_c - ha_o)
    lower_shadow = min(ha_o, ha_c) - ha_l
    upper_shadow = ha_h - max(ha_o, ha_c)
    total_range = ha_h - ha_l

    if total_range == 0:
        return 0.0

    # 1. Hammer pattern recognition
    hammer_score = 0.0
    if lower_shadow > body_size * 2 and upper_shadow < body_size * 0.5:
        # Classic hammer: long lower wick, small body, small upper wick
        hammer_score = min(1.0, lower_shadow / (atr_cur * 0.5))

    # 2. Support proximity for double bottom confirmation
    recent_lows = low_prices[max(0, idx-support_lookback):idx+1]
    support_level = min(recent_lows) if len(recent_lows) > 0 else ha_l
    dist_from_support = ha_l - support_level
    tolerance = atr_cur * 0.3

    support_score = 0.0
    if dist_from_support <= tolerance:
        support_score = 1.0 - (dist_from_support / tolerance)

    # 3. Body smallness relative to range
    body_ratio = body_size / total_range
    small_body_bonus = 0.0
    if body_ratio < 0.3:  # Body less than 30% of total range
        small_body_bonus = 1.0 - (body_ratio / 0.3)

    # Combined score: hammer pattern is most important for entry
    return min(1.0, (hammer_score * 0.5) + (support_score * 0.3) + (small_body_bonus * 0.2))
```

## Integration with Existing Momentum System

### Context-Specific Quality Computation
```python
def next(self):
    """Enhanced next() method with context-specific doji detection."""

    # Compute quality scores based on context
    if not self.position:
        # For entry: only compute bullish hammer quality
        bull_quality = self.compute_bullish_hammer_quality()
        bear_quality = 0.0  # Not used for entry
    else:
        # For exit: only compute bearish momentum loss quality
        bear_quality = self.compute_bearish_momentum_loss_quality()
        bull_quality = 0.0  # Not used for exit

    # Pass context-appropriate qualities to scoring functions
    entry_score = self.compute_entry_score(bull_quality)  # Only uses bull_quality
    exit_score = self.compute_exit_score(bear_quality)    # Only uses bear_quality
    # ... rest of logic
```

### Enhanced Score Computation
```python
def _compute_score_numba(ha_open, ha_close, ha_high, ha_low, low_prices, atr_cur,
                        weights, context_quality, weight_doji, doji_threshold, is_entry=True):
    # ... existing momentum scoring logic ...

    # Context-specific doji integration
    if context_quality > doji_threshold:
        if is_entry:
            # Bullish hammer reduces entry score (bearish signal for entry)
            score -= weight_doji * context_quality
        else:
            # Bearish momentum loss increases exit score (bearish signal for exit)
            score += weight_doji * context_quality

    return score
```

## Parameter Structure

### Context-Specific Parameters
```python
# Bearish momentum loss parameters (for exit signals only)
doji_lookback_bear = 5      # Lookback periods for analyzing bullish sequences
doji_threshold_bear = 0.4   # Quality threshold for bearish momentum loss

# Bullish hammer parameters (for entry signals only)
doji_lookback_bull = 15     # Lookback periods for support proximity
doji_threshold_bull = 0.4   # Quality threshold for bullish hammer
hammer_atr_factor = 0.5     # Hammer shadow length as fraction of ATR

# Separate weights for context-specific signals
weight_bear_momentum = 0.35 # Weight for bearish momentum loss (exit signal)
weight_bull_hammer = 0.45   # Weight for bullish hammer rejection (entry signal)
```

### Optimization Ranges
```python
bt.optimize(
    # ... existing params ...
    doji_lookback_bear=(3, 8),       # 3-8 periods for bear momentum analysis
    doji_threshold_bear=(0.3, 0.6),  # 0.3-0.6 bear quality threshold
    doji_lookback_bull=(10, 25),     # 10-25 periods for bull support analysis
    doji_threshold_bull=(0.3, 0.6),  # 0.3-0.6 bull quality threshold
    hammer_atr_factor=(0.3, 0.7),    # 0.3-0.7 ATR fraction for hammer
    weight_bear_momentum=(25, 45),   # 0.25-0.45 bear momentum weight
    weight_bull_hammer=(35, 55),     # 0.35-0.55 bull hammer weight
    # ... other params ...
)
```

## Implementation Phases

### Phase 1: Context-Specific Quality Functions
1. Implement `get_bearish_momentum_loss_quality_numba()` - analyzes bullish sequences for momentum decay
2. Implement `get_bullish_hammer_quality_numba()` - detects hammer patterns near support
3. Ensure numba compatibility for performance

### Phase 2: Integration with Existing Logic
1. Modify `_compute_score_numba` to accept single context_quality parameter
2. Update `next()` method to compute appropriate quality based on position context
3. Replace binary doji check with context-aware quality scoring
4. Maintain existing momentum system for consecutive bar handling

### Phase 3: Parameter Optimization & Validation
1. Add context-specific parameters to optimization grid
2. Run backtests to validate asymmetric signal quality
3. Compare with current binary system performance
4. Fine-tune thresholds and weights based on results

## Expected Benefits

### 1. Context-Aware Signal Quality
- **Exit Signals**: Bearish momentum loss detection complements existing consecutive red bar handling
- **Entry Signals**: Hammer patterns provide high-confidence bullish reversal signals
- **No Signal Conflicts**: Different detection logic for different market contexts

### 2. Enhanced Momentum Integration
- **Bearish Detection**: Focuses on pre-consecutive-red sequences that momentum system doesn't handle
- **Complementary Logic**: Works alongside existing bonus/penalty system without overlap
- **Sequence Analysis**: Analyzes `[g_1, g_2, r_1, g_3]` patterns for early warning

### 3. Asymmetric Market Alignment
- **Tops**: Slow momentum decay detection (exit focus)
- **Bottoms**: Sharp rejection pattern detection (entry focus)
- **Quality-Based**: Continuous scoring eliminates binary optimization issues

### 4. Improved Signal Reliability
- **Reduced False Positives**: Context-specific detection minimizes inappropriate signals
- **Higher Conviction**: Quality scores provide confidence levels for each signal
- **Market-Responsive**: Adapts to different volatility and trend conditions

## Risk Mitigation

1. **Backward Compatibility**: Maintain current logic as fallback
2. **Parameter Validation**: Ensure new ranges produce stable results
3. **Performance Monitoring**: Track signal quality metrics
4. **Incremental Rollout**: Test components separately before full integration

## Success Metrics

1. **Improved Sharpe Ratio**: Better risk-adjusted returns
2. **Reduced Drawdown**: Earlier exit from losing positions
3. **Higher Win Rate**: Better entry timing
4. **Smoother Optimization**: Continuous quality scores vs binary cliffs

This plan transforms the current binary doji detection into a sophisticated, momentum-aware system that respects market asymmetry while building on the existing momentum tracking foundation.