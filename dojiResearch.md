This report synthesizes the transition from binary Doji detection to a **Quality-Based Asymmetric Scoring System**.

### Executive Summary

Standard Heikin Ashi (HA) algorithms often fail because they treat all Dojis equally and assume market tops and bottoms behave symmetrically. Research indicates two critical pivots for improvement:

1.  **Volatility Normalization:** A "small body" is relative. It must be defined by the Average True Range (ATR), not fixed points.
2.  **Market Asymmetry:**
      * **Tops (Bearish):** Characterized by **Momentum Decay** (shrinking bodies, "Rollover").
      * **Bottoms (Bullish):** Characterized by **Violent Rejection** (long shadows, "Sharp Turns", support interaction).

-----

### Observation 1: The "Quality Score" Math

**Problem:** A binary check (`if body < threshold`) creates optimization "cliffs." A body slightly above the threshold gets 0 weight, while one slightly below gets full weight.
**Solution:** Use Linear Decay. The closer the body is to 0, the higher the score.

**Formula:**
$$Q_{body} = 1.0 - \left( \frac{\text{Body Size}}{\text{ATR} \times \text{Factor}} \right)$$
*If the body size exceeds the limit, the score is 0. If the body is 0 (flat), the score is 1.*

**Implementation Code:**

```python
def get_volatility_normalized_score(self, body_size, atr_value, sensitivity_factor=0.1):
    """
    Returns a float 0.0 to 1.0 based on how 'doji-like' the bar is relative to ATR.
    sensitivity_factor: Max body size as a fraction of ATR (e.g., 0.1 = 10% of ATR).
    """
    max_allowed_body = atr_value * sensitivity_factor
    
    if max_allowed_body == 0: return 0.0
    if body_size > max_allowed_body: return 0.0
    
    # Linear Decay: 1.0 for perfect doji, 0.0 for body at limit
    return 1.0 - (body_size / max_allowed_body)
```

-----

### Observation 2: Bearish Asymmetry (The "Rollover")

**Market Behavior:** Bull trends rarely stop abruptly. They run out of steam. We look for **Deceleration**.
**Key Indicators:**

1.  **Body Shrinkage:** Current Green body is smaller than the average of previous Green bodies.
2.  **Shadow Emergence:** Green HA candles should have *no* lower shadow. If one appears, buying pressure is weakening.

**Implementation Code:**

```python
def get_bear_rollover_quality(self, lookback=3):
    """
    Detects momentum loss at the top.
    Returns 0.0 to 1.0 based on deceleration magnitude.
    """
    # 1. Fetch recent bodies to establish a baseline
    recent_bodies = [abs(self.ha_close[-i] - self.ha_open[-i]) for i in range(1, lookback+1)]
    avg_recent_body = sum(recent_bodies) / lookback if recent_bodies else 1.0
    
    current_body = abs(self.ha_close[0] - self.ha_open[0])
    
    # 2. Calculate Deceleration (Linear Decay)
    # Score is high if current body is tiny compared to recent average
    momentum_score = 0.0
    if current_body < avg_recent_body:
        momentum_score = 1.0 - (current_body / avg_recent_body)
        
    # 3. Check for "Rotting" Signal (Lower Shadow on Green Candle)
    # In a strong uptrend, HA Green candles have flat bottoms. 
    # A lower shadow indicates weakness.
    shadow_penalty_score = 0.0
    ha_open, ha_close = self.ha_open[0], self.ha_close[0]
    
    if ha_close > ha_open: # If currently Green
        lower_shadow = ha_open - self.ha_low[0]
        # If lower shadow exists and is significant (> 20% of body)
        if lower_shadow > (current_body * 0.2):
            shadow_penalty_score = 0.5 # Boost the bearish signal
            
    # Weighted combination
    return min(1.0, (momentum_score * 0.7) + (shadow_penalty_score * 0.3))
```

-----

### Observation 3: Bullish Asymmetry (The "Sharp Turn")

**Market Behavior:** Panic selling creates V-bottoms. We look for **Structural Rejection**.
**Key Indicators:**

1.  **The Hammer (Pin Bar):** Long lower shadow indicating intraday rejection.
2.  **Support Proximity:** A Doji floating in mid-air is a "falling knife." A Doji at a previous low is a "Double Bottom."

[Image of double bottom chart pattern with hammer candle]

**Implementation Code:**

```python
def get_bull_rejection_quality(self, support_lookback=20):
    """
    Detects sharp rejection near structural support.
    """
    ha_o, ha_c = self.ha_open[0], self.ha_close[0]
    ha_l = self.ha_low[0]
    current_atr = self.atr[0]
    
    # 1. Shadow Intensity (The Hammer Shape)
    # A huge lower shadow relative to ATR implies massive rejection
    lower_shadow = min(ha_o, ha_c) - ha_l
    
    # Cap the score: If shadow is > 50% of ATR, max score achieved
    shadow_score = min(1.0, lower_shadow / (current_atr * 0.5))
    
    # 2. Structural Support (Double Bottom Detection)
    # Calculate distance from the lowest low of the last 'n' periods
    recent_period_low = min(self.data.low.get(ago=-1, size=support_lookback))
    current_low = self.data.low[0]
    
    dist_from_support = current_low - recent_period_low
    
    # Score is 1.0 if we touched the previous low, 0.0 if we are far away
    support_score = 0.0
    tolerance = current_atr * 0.5 # Defined "Zone" of support
    
    if dist_from_support < tolerance:
        support_score = 1.0 - (dist_from_support / tolerance)
        
    # Final Bull Score: Shadow is vital, Support is confirmation
    return (shadow_score * 0.6) + (support_score * 0.4)
```

-----

### Observation 4: Integration Strategy

Instead of a single `weight_doji`, the final scoring system should separate the weights to account for the asymmetric nature of the market.

**Proposed Parameter Structure:**

  * `weight_bear_rollover` (Optimized, usually lower magnitude because tops are slow)
  * `weight_bull_rejection` (Optimized, usually higher magnitude because bottoms are fast)

**Logic Snippet:**

```python
# In next():
bear_signal = self.get_bear_rollover_quality()
bull_signal = self.get_bull_rejection_quality()

# Threshold to filter noise (e.g., 0.4)
if bear_signal > 0.4:
    self.score -= self.params.weight_bear_rollover * bear_signal

if bull_signal > 0.4:
    self.score += self.params.weight_bull_rejection * bull_signal
```

### Next Steps for You

Would you like me to construct the full **`bt.Strategy` class** incorporating these snippets, specifically setting up the `bt.optimize` ranges for the `support_lookback` and `sensitivity_factor`?

https://github.com/zeta-zetra/code