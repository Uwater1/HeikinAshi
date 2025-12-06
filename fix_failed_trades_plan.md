# Fix Failed Trades Issue

## Problem
The backtest shows 5 failed trades occurring within the first 60 bars of the program start. This is likely due to stop loss prices being too close to entry prices early in the backtest when ATR values are small.

## Root Cause
- Early in the backtest, ATR calculation uses limited historical data
- Stop loss calculation: `sl_price = max(price * 0.95, price - self.stop_atr_mult * atr_cur)`
- When ATR is small, `price - self.stop_atr_mult * atr_cur` is very close to `price`
- The backtesting library may reject stop losses that are too close to entry price
- The try-except block catches this and falls back to `self.buy()` without stop loss

## Solution
Instead of fixing the stop loss distance, implement a simpler solution: **Skip trading in the first 50 bars** to allow indicators to stabilize.

## Implementation Plan

### Code Changes Required

1. **Add bar counter in `init()` method:**
   ```python
   def init(self):
       # ... existing code ...
       self.bar_count = 0
   ```

2. **Add early return in `next()` method:**
   ```python
   def next(self):
       self.bar_count += 1
       if self.bar_count <= 50:
           return
       # ... rest of existing code ...
   ```

### Testing
- Run backtest with the modified code
- Verify that no trades occur in the first 50 bars
- Check that trading resumes normally after bar 50
- Ensure overall backtest results remain valid

### Benefits
- Eliminates failed trades early in backtest
- Allows ATR and RSI indicators to stabilize with more historical data
- Simple implementation with minimal risk of side effects
- Maintains strategy logic for the majority of the backtest period

## Alternative Solutions (if needed)
If skipping trades is not desired, the stop loss calculation could be modified to ensure minimum distance:

```python
atr_stop = self.stop_atr_mult * atr_cur
min_stop_distance = price * 0.02  # 2% minimum
sl_price = price - max(atr_stop, min_stop_distance)