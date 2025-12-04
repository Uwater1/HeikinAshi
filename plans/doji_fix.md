# Doji Backward Compatibility Fix Plan

## Problem Analysis

The current asymmetric doji implementation produces worse results than the original system because:

1. **Complete Replacement**: The new quality-based logic entirely replaces the old binary doji detection
2. **Zero Weight Issue**: When `weight_bear_momentum = 0` and `weight_bull_hammer = 0`, there are **no doji signals at all**, whereas the old system always had doji signals when binary conditions were met
3. **Missing Fallback**: No backward compatibility mechanism to restore old behavior

## Root Cause

**Old System Behavior:**
```python
# Always applied when binary conditions met
if is_entry and ha_close[prior_idx] < ha_open[prior_idx]:
    score += weight_bull_doji  # Always contributed when bearish doji detected
elif not is_entry and ha_close[prior_idx] > ha_open[prior_idx]:
    score += weight_bear_doji  # Always contributed when bullish doji detected
```

**New System Behavior:**
```python
# Only applies when quality > threshold AND weight > 0
if context_quality > doji_threshold:
    if is_entry:
        score -= weight_doji * context_quality  # Only if weight_bull_hammer > 0
    else:
        score += weight_doji * context_quality  # Only if weight_bear_momentum > 0
```

When new weights are 0, **no doji contribution occurs**, breaking backward compatibility.

## Solution: Hybrid Fallback System

### Core Principle
- **New weights > 0**: Use quality-based asymmetric detection
- **New weights = 0**: Fall back to original binary detection with old weights
- **Smooth transition**: Allow optimization to choose between old and new paradigms

### Implementation Strategy

#### 1. Make All Hardcoded Values Optimizable
Replace hardcoded values in quality functions with parameters:

```python
@jit(nopython=True, cache=True)
def get_bearish_momentum_loss_quality_numba(ha_open, ha_close, ha_high, ha_low,
                                           lookback=5, idx=0):  # lookback becomes parameter

@jit(nopython=True, cache=True)
def get_bullish_hammer_quality_numba(ha_open, ha_close, ha_high, ha_low, low_prices, atr_cur,
                                    support_lookback=15, hammer_atr_factor=0.5, idx=0):  # Both become parameters
```

#### 2. Modify Score Computation Logic
```python
def _compute_score_numba(ha_open, ha_close, atr_cur, weights, doji_weight, doji_threshold,
                        context_quality, weight_bull_bonus=0.1, weight_bear_bonus=0.1,
                        weight_bull_penalty=0.05, weight_bear_penalty=0.05, is_entry=True,
                        use_legacy_doji=False, legacy_doji_body_frac=0.20, legacy_doji_position=-6):
    # ... existing momentum logic ...

    if use_legacy_doji:
        # Legacy binary doji logic (when new weights are 0)
        try:
            prior_idx = legacy_doji_position  # Now optimizable: -8 to -4 range
            prior_body = abs(ha_close[prior_idx] - ha_open[prior_idx])
            if prior_body < legacy_doji_body_frac * atr_cur:
                if is_entry:
                    if ha_close[prior_idx] < ha_open[prior_idx]:
                        score += doji_weight
                else:
                    if ha_close[prior_idx] > ha_open[prior_idx]:
                        score += doji_weight
        except:
            pass
    else:
        # New quality-based logic
        if context_quality > doji_threshold:
            if is_entry:
                score -= doji_weight * context_quality
            else:
                score += doji_weight * context_quality
```

#### 2. Strategy Class Modifications
```python
def compute_entry_score(self):
    # Determine which doji system to use
    use_legacy = (self.weight_bull_hammer == 0.0)

    if use_legacy:
        # Use legacy binary logic - no quality computation needed
        bull_quality = 0.0  # Not used
    else:
        # Use new quality-based logic
        bull_quality = get_bullish_hammer_quality_numba(...)

    return _compute_score_numba(
        ..., bull_quality, ...,
        use_legacy_doji=use_legacy,
        legacy_doji_body_frac=self.doji_body_frac
    )

def compute_exit_score(self):
    # Similar logic for exit
    use_legacy = (self.weight_bear_momentum == 0.0)
    # ... rest of logic
```

#### 3. Parameter Structure Updates
```python
# Legacy parameters (now optimizable to include original values)
doji_body_frac = 0.20      # Original: 0.20, Range: 0.10-0.30
legacy_doji_position = -6  # Original: -6, Range: -8 to -4

# Quality function parameters (include original hardcoded values)
doji_lookback_bear = 5     # Original: 5, Range: 3-8
doji_lookback_bull = 15    # Original: 15, Range: 10-25
hammer_atr_factor = 0.5    # Original: 0.5, Range: 0.3-0.7

# New weights with 0 as backward-compatible value
weight_bear_momentum = 0.0 # 0 = use legacy, >0 = use new quality
weight_bull_hammer = 0.0   # 0 = use legacy, >0 = use new quality
```

#### 4. Complete Optimization Ranges
```python
bt.optimize(
    # ... existing params ...
    # Legacy parameters (include original values)
    doji_body_frac=(10, 30),        # 0.10-0.30 (includes 0.20)
    legacy_doji_position=(-8, -4),  # -8 to -4 (includes -6)

    # Quality function parameters (include original hardcoded values)
    doji_lookback_bear=(3, 8),      # 3-8 (includes 5)
    doji_lookback_bull=(10, 25),    # 10-25 (includes 15)
    hammer_atr_factor=(30, 70),     # 0.3-0.7 (includes 0.5)

    # New weights (0 enables legacy mode)
    weight_bear_momentum=(0, 50),   # 0.0-0.5 (0 = legacy mode)
    weight_bull_hammer=(0, 60),     # 0.0-0.6 (0 = legacy mode)
    # ... other params ...
)
```

## Implementation Phases

### Phase 1: Make All Hardcoded Values Optimizable
1. Modify quality functions to accept parameters instead of hardcoded values
2. Update `_compute_score_numba` to accept additional legacy parameters
3. Add `legacy_doji_position` parameter for position optimization

### Phase 2: Implement Legacy Fallback Logic
1. Add `use_legacy_doji` detection based on weight values
2. Implement conditional logic: legacy binary vs quality-based scoring
3. Ensure legacy mode uses exact original logic with optimizable parameters

### Phase 3: Update Strategy Methods
1. Modify `compute_entry_score()` and `compute_exit_score()` to pass all parameters
2. Conditionally compute quality scores only when using new system
3. Update parameter passing to numba functions

### Phase 4: Complete Parameter Setup
1. Add all legacy and quality parameters to strategy class
2. Set up proper parameter scaling (/100 for SAMBO integers)
3. Update optimization ranges to include original hardcoded values

### Phase 5: Testing & Validation
1. Test legacy mode (`weights = 0`) reproduces original behavior
2. Test quality mode (`weights > 0`) uses new asymmetric detection
3. Validate SAMBO optimization can find parameter combinations at least as good as original
4. Confirm all hardcoded values (0.2, 6, 5, 15, 0.5) are within optimization ranges

## Expected Outcomes

### Backward Compatibility Achieved
- **Old Performance Restored**: Setting new weights to 0 restores exact old behavior
- **Optimization Flexibility**: Optimizer can choose between paradigms
- **Smooth Migration**: Gradual transition from old to new system possible

### Benefits Maintained
- **Quality Detection**: When weights > 0, sophisticated asymmetric detection active
- **No Performance Regression**: Legacy mode preserves original performance
- **Enhanced Optimization**: Search space includes both old and new approaches

## Risk Mitigation

1. **Exact Legacy Reproduction**: Ensure legacy mode produces identical results to original
2. **Parameter Validation**: Test edge cases where weights are exactly 0
3. **Optimization Stability**: Verify optimizer can navigate between legacy and new regions
4. **Performance Impact**: Quality computation only when needed (weights > 0)

This hybrid approach ensures the new asymmetric doji system can be at least as good as the original while providing enhanced capabilities when the optimizer chooses to use them.