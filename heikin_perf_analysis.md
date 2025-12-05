# HeikinAshi Performance Optimization Report

## Executive Summary
This report identifies 15+ optimization opportunities that could improve execution speed by 30-60% through better algorithms, reduced redundancy, and improved JIT compilation.

---

## Critical Performance Issues

### 1. **Inefficient Bar Selection Logic in `_compute_score_numba`**
**Location:** Lines 95-110  
**Issue:** The bar selection uses a Python list (`selected`) which prevents full Numba optimization
```python
selected = []
started = False
for i in range(lookback):
    idx = -(i + 1)
    is_desired = (is_entry and ha_close[idx] > ha_open[idx]) or ...
    if is_desired:
        started = True
        selected.append(i)
```

**Impact:** HIGH - This causes Numba to fall back to object mode  
**Solution:** Pre-allocate fixed-size array and use index counter
```python
selected = np.empty(lookback, dtype=np.int32)
selected_count = 0
```

---

### 2. **Redundant Bar Size Calculations**
**Location:** Lines 112-121  
**Issue:** Calculates `bar_size` for all selected bars, then accesses them multiple times
```python
for i in selected:
    idx = -(i + 1)
    ha_o = ha_open[idx]
    ha_c = ha_close[idx]
    body = ...
    bar_size = body / atr_cur
    bar_sizes[i] = bar_size + 0.01 if bar_size > 0 else 0
```

**Impact:** MEDIUM - Division and conditional logic repeated unnecessarily  
**Solution:** Vectorize the calculation once before the loop

---

### 3. **Repeated ATR Validation**
**Location:** Lines 213, 232, 252  
**Issue:** Every function checks `atr_cur <= 0` and sets default
```python
atr_cur = float(self.atr[-1] if len(self.atr) > 0 else 1.0)
if atr_cur <= 0:
    atr_cur = 1.0
```

**Impact:** LOW-MEDIUM - Repeated validation across multiple calls per bar  
**Solution:** Validate once in `next()` and pass validated value

---

### 4. **Inefficient Array Conversions**
**Location:** Lines 195-198  
**Issue:** Converting entire arrays to float32 on every `init()` call
```python
self.ha_open = self.I(lambda: np.asarray(self.data.HA_open, dtype=np.float32))
```

**Impact:** LOW - Only happens once, but unnecessary if data already correct type  
**Solution:** Convert during data loading phase, not in strategy init

---

### 5. **Momentum Tracking Redundancy**
**Location:** Lines 128-161  
**Issue:** Identical logic repeated for 3rd and 4th bars with only index changes
```python
if len_selected >= 3:
    current_idx = selected[2]
    prev_indices = selected[0:2]
    max_prev_size = max(bar_sizes[prev_indices[0]], bar_sizes[prev_indices[1]])
    # ... 20 lines ...
if len_selected >= 4:
    current_idx = selected[3]
    prev_indices = selected[0:3]
    max_prev_size = max(bar_sizes[prev_indices[0]], ...)
    # ... same logic ...
```

**Impact:** MEDIUM - Duplicated code prevents optimization  
**Solution:** Extract to separate Numba function or use loop

---

### 6. **Try-Except in Hot Loop**
**Location:** Lines 164-175  
**Issue:** Try-except block in JIT-compiled function forces object mode
```python
try:
    prior_body = abs(ha_close[prior_idx] - ha_open[prior_idx])
    if prior_body < doji_body_frac * atr_cur:
        score += doji_weight
except:
    pass
```

**Impact:** HIGH - Prevents full Numba optimization  
**Solution:** Pre-validate indices or use explicit bounds checking

---

### 7. **Multiple Float Conversions**
**Location:** Lines 204-209, 249-253  
**Issue:** Converting numpy values to float multiple times per call
```python
def _is_green(self, idx):
    return float(self.ha_close[idx]) > float(self.ha_open[idx])

def _is_red(self, idx):
    return float(self.ha_close[idx]) < float(self.ha_open[idx])
```

**Impact:** LOW-MEDIUM - Called frequently but simple operations  
**Solution:** These helper functions are unused - remove them

---

### 8. **Unnecessary Array Pre-allocation**
**Location:** Line 95  
**Issue:** Pre-allocates full `bar_sizes` array but may only use subset
```python
bar_sizes = np.empty(lookback, dtype=np.float32)
```

**Impact:** LOW - Small array, minimal memory impact  
**Solution:** Only allocate for `len(selected)` after selection

---

### 9. **Non-Vectorized ATR Calculation**
**Location:** Lines 41-51  
**Issue:** True Range calculation uses explicit loop instead of numpy operations
```python
for i in range(n):
    if i == 0:
        tr[i] = high[i] - low[i]
    else:
        tr[i] = max(high[i] - low[i], ...)
```

**Impact:** LOW - Already JIT-compiled, but could be faster  
**Solution:** Use numpy operations for vectorization: `np.maximum.reduce()`

---

### 10. **Weight Scaling in init()**
**Location:** Lines 183-194  
**Issue:** Dividing all weights by 100.0 individually
```python
self.weight_bull_1 = self.weight_bull_1 / 100.0
self.weight_bull_2 = self.weight_bull_2 / 100.0
# ... 15 more lines ...
```

**Impact:** NEGLIGIBLE - One-time operation  
**Solution:** Use loop or dict comprehension for cleaner code

---

### 11. **Redundant Score Computation**
**Location:** Lines 211-225, 228-242  
**Issue:** Both entry and exit scores create new numpy arrays every call
```python
bull_weights = np.array([self.weight_bull_1, ...], dtype=np.float32)
bear_weights = np.array([self.weight_bear_1, ...], dtype=np.float32)
```

**Impact:** MEDIUM - Called on every bar during backtest  
**Solution:** Pre-compute weight arrays in `init()` and reuse

---

### 12. **Repeated Position Checks**
**Location:** Lines 253-256  
**Issue:** Nested exception handling with multiple fallback attempts
```python
try:
    self.position.close()
except Exception:
    try:
        self.sell()
    except Exception:
        pass
```

**Impact:** LOW - Only triggered on exit signals  
**Solution:** Check position state before attempting close

---

### 13. **CSV Format Detection Overhead**
**Location:** Lines 180-195  
**Issue:** Reads CSV twice - once for format detection, once for full load
```python
df = pd.read_csv(path, nrows=5)  # First read
# ...
df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")  # Second read
```

**Impact:** LOW - One-time operation at startup  
**Solution:** Use single read with try-except for column renaming

---

### 14. **Unused Helper Functions**
**Location:** Lines 204-213  
**Issue:** `_is_green()`, `_is_red()`, and `_ha_body()` defined but never called
```python
def _is_green(self, idx):
    """Check if candle at idx is green."""
    return float(self.ha_close[idx]) > float(self.ha_open[idx])
```

**Impact:** NONE - Dead code  
**Solution:** Remove to reduce code complexity

---

### 15. **Missing Caching for Pandas Operations**
**Location:** Lines 135-136  
**Issue:** Using pandas for rolling mean in hot path
```python
atr = pd.Series(tr).rolling(window=length, min_periods=1).mean().to_numpy()
```

**Impact:** LOW-MEDIUM - Called once per backtest initialization  
**Solution:** Implement custom rolling mean in Numba

---

## Optimization Priority Matrix

| Priority | Issue | Estimated Speedup | Complexity |
|----------|-------|-------------------|------------|
| 🔴 **P0** | Bar selection logic (#1) | 15-25% | Medium |
| 🔴 **P0** | Try-except in JIT (#6) | 10-20% | Low |
| 🟡 **P1** | Redundant score arrays (#11) | 5-10% | Low |
| 🟡 **P1** | Momentum tracking duplication (#5) | 5-8% | Medium |
| 🟡 **P1** | Redundant bar calculations (#2) | 3-7% | Low |
| 🟢 **P2** | ATR validation (#3) | 2-4% | Low |
| 🟢 **P2** | Float conversions (#7) | 1-3% | Low |
| ⚪ **P3** | All others | <2% each | Varies |

---

## Recommended Implementation Plan

### Phase 1: Critical Fixes (Target: 25-35% speedup)
1. **Rewrite bar selection with fixed arrays**
   - Replace Python list with pre-allocated numpy array
   - Use counter instead of dynamic append
   - Ensure full nopython mode

2. **Remove try-except from JIT function**
   - Add explicit bounds checking before doji logic
   - Use conditional: `if prior_idx >= -len(ha_close):`

3. **Pre-compute weight arrays**
   - Create `self.bull_weights_arr` and `self.bear_weights_arr` in init()
   - Pass references instead of creating new arrays

### Phase 2: Medium Impact (Target: 10-15% speedup)
4. **Consolidate momentum tracking**
   - Create separate Numba function for momentum bonus/penalty
   - Call in loop for bars 3+ instead of duplicating code

5. **Eliminate redundant calculations**
   - Validate ATR once in `next()`
   - Remove unused helper functions
   - Cache bar size calculations

### Phase 3: Polish (Target: 5-10% speedup)
6. **Vectorize ATR calculation**
   - Replace loop with `np.maximum.reduce([array1, array2, array3])`

7. **Optimize data loading**
   - Single-pass CSV reading
   - Direct type conversion during load

---

## Code Smell Summary

- **Premature Type Conversion:** Converting types multiple times
- **Dead Code:** Unused helper functions
- **Copy-Paste Programming:** Duplicated momentum logic
- **Exception-Driven Flow:** Using try-except for control flow
- **Magic Numbers:** Hardcoded thresholds (1.0, 0.95, etc.)
- **God Function:** `_compute_score_numba` does too many things

---

## Testing Strategy

After each optimization:
1. **Correctness Test:** Compare results with original on sample dataset
2. **Performance Test:** Time 1000 iterations of score computation
3. **Profile:** Use `line_profiler` to verify hotspot elimination
4. **Regression Test:** Ensure optimization results match baseline

---

## Expected Overall Impact

**Conservative Estimate:** 30-40% speed improvement  
**Optimistic Estimate:** 50-60% speed improvement  

Primary gains from:
- Eliminating Python object mode in Numba (15-25%)
- Reducing redundant calculations (10-15%)
- Better memory access patterns (5-10%)

---

## Additional Recommendations

1. **Use `@njit` instead of `@jit(nopython=True)`** - More explicit
2. **Add `fastmath=True`** flag to Numba functions for floating-point speed
3. **Consider `parallel=True`** for True Range calculation if dataset is large
4. **Profile with `py-spy`** to identify any remaining bottlenecks
5. **Add timing decorators** to measure actual impact of changes

---

## Conclusion

The code has good foundations with Numba usage, but several anti-patterns prevent full optimization. The most critical issues are in the score computation function where Python object mode is triggered. Fixing the top 5 issues should yield 30-40% improvement with moderate effort.

The optimization work should be done incrementally with testing after each change to ensure correctness is maintained while improving performance.