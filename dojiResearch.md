This is an advanced optimization problem. Moving from a binary check (`is_doji = True/False`) to a continuous **Quality Score** will significantly smooth your backtest parameter surface and likely improve robustness.

Here is the research and implementation plan for your **Quality-Based Doji Scoring** and **Dynamic Thresholds**.

### 1\. The Strategy: Quantifying "Doji Quality"

Instead of a boolean flag, we calculate a `quality` score (0.0 to 1.0) for each candle. The final signal weight becomes:
$$\text{Weight} = \text{Base Weight} \times \text{Quality Score}$$

The Quality Score is composed of three factors:

1.  **Body Factor ($Q_{body}$):** The smaller the body relative to volatility (ATR), the higher the score.
2.  **Shadow Factor ($Q_{shadow}$):** Long shadows indicate rejection/indecision. Longer is better.
3.  **Type Factor ($Q_{type}$):** Specific shapes (Dragonfly, Gravestone, Long-Legged) get multipliers.

### 2\. Implementation logic

#### A. Dynamic Body Threshold (ATR-Based)

Hardcoding `doji_body_frac` fails when volatility changes. We use ATR to normalize.

  * **Logic:** A body of 5 ticks is a Doji in high volatility (ATR=100) but a trend bar in low volatility (ATR=6).
  * **Formula:** `max_body_size = doji_body_atr_frac * atr_cur`

#### B. The "Hammer" & "Doji" Types

In Heikin Ashi (HA), specific Dojis mean different things:

  * **Classic Doji:** Small body, shadows on both sides. *Sign: Indecision.*
  * **Dragonfly/Hammer-like:** Small body, long lower shadow, no upper shadow. *Sign: Bullish Reversal.*
  * **Gravestone/Star-like:** Small body, long upper shadow, no lower shadow. *Sign: Bearish Reversal.*

### 3\. Improved Code Structure (Backtrader Compatible)

This implementation replaces your boolean logic with a continuous scoring system.

```python
import backtest as bt # Assuming standard Backtrader or compatible framework

class HeikinAshiStrategy(bt.Strategy):
    params = (
        ('atr_period', 14),
        ('doji_atr_frac', 0.1),   # Dynamic threshold: Body must be < 10% of ATR
        ('weight_doji_base', 10), # Base score added when Doji is found
        ('prior_idx', -6),        # Lookback index (subject to optimize)
    )

    def __init__(self):
        # ... existing HA calculation code ...
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)

    def calculate_doji_score(self, idx):
        # 1. Fetch HA values
        ha_o = self.ha_open[idx]
        ha_h = self.ha_high[idx]
        ha_l = self.ha_low[idx]
        ha_c = self.ha_close[idx]
        
        # 2. Basic Dimensions
        body_size = abs(ha_c - ha_o)
        upper_shadow = ha_h - max(ha_o, ha_c)
        lower_shadow = min(ha_o, ha_c) - ha_l
        total_range = ha_h - ha_l
        
        # Avoid division by zero
        if total_range == 0: return 0 

        # 3. Dynamic Threshold Check (Gatekeeper)
        # If body is too big relative to ATR, it's not a Doji at all.
        # current ATR at that specific index
        current_atr = self.atr[idx] 
        if body_size > (self.params.doji_atr_frac * current_atr):
            return 0.0

        # --- QUALITY SCORING START ---

        # A. Body Quality (0.0 to 1.0)
        # The closer the body is to 0, the closer this is to 1.0
        q_body = 1.0 - (body_size / (self.params.doji_atr_frac * current_atr))

        # B. Shadow Intensity (0.0 to 1.0)
        # We want long shadows relative to the body or range.
        # Using Shadow / ATR ratio allows us to reward volatile rejection.
        shadow_dominance = (upper_shadow + lower_shadow) / total_range
        
        # C. Type Classification & Multipliers
        # Heikin Ashi Hammer (Bullish Reversal Sign): Long Lower, No Upper
        is_hammer_bull = (lower_shadow > body_size * 2) and (upper_shadow < body_size * 0.5)
        
        # Heikin Ashi Star (Bearish Reversal Sign): Long Upper, No Lower
        is_star_bear = (upper_shadow > body_size * 2) and (lower_shadow < body_size * 0.5)
        
        # Long Legged Doji: Huge shadows on both sides
        is_long_legged = (lower_shadow > body_size) and (upper_shadow > body_size)

        type_multiplier = 1.0
        if is_hammer_bull or is_star_bear:
            type_multiplier = 1.5  # Boost score for directional rejection
        elif is_long_legged:
            type_multiplier = 1.2  # Boost for extreme indecision

        # Final Quality Calculation
        # We weight shadow dominance slightly less than body purity
        final_quality = (q_body * 0.7 + shadow_dominance * 0.3) * type_multiplier

        return final_quality

    def next(self):
        # ... existing logic ...
        
        # Calculate score using the helper function
        doji_quality = self.calculate_doji_score(self.params.prior_idx)
        
        if doji_quality > 0:
            # Apply the weighted score
            # If is_entry logic matches your reversal direction:
            
            score_change = self.params.weight_doji_base * doji_quality
            
            # Example directional logic:
            prior_idx = self.params.prior_idx
            ha_c = self.ha_close[prior_idx]
            ha_o = self.ha_open[prior_idx]
            
            # Bullish Doji Signal (Green Doji or Hammer)
            if ha_c > ha_o: 
                 self.score += score_change
            # Bearish Doji Signal (Red Doji or Star)
            else:
                 self.score -= score_change # or whatever your scoring logic is
```

### 4\. Key Improvements Explained

1.  **`q_body` (Linear Decay):**

      * Previously: If `body` was 0.09 and limit was 0.1, weight was 100%. If `body` was 0.11, weight was 0%.
      * Now: A perfect flat doji gets 100% weight. A "fat" doji near the limit gets nearly 0% weight. This removes "cliff-edge" optimization artifacts.

2.  **`shadow_dominance`:**

      * This rewards "high-energy" indecision. A tiny Doji in a flat market is less significant than a massive "Long-Legged Doji" that signifies a violent battle between bulls and bears.

3.  **Hammer Detection (`is_hammer_bull`):**

      * You mentioned finding hammer-like indicators useful. In Heikin Ashi, a **Green candle with no lower shadow** is a trend signal, but a **candle with a long lower shadow and small body** (Hammer) is a reversal warning. The code explicitly detects this geometry and boosts the weight (`type_multiplier = 1.5`).

### 5\. Optimization Tips

When running `bt.optimize`, you can now refine these ranges:

  * `doji_atr_frac`: Range `0.05` to `0.2` (Strict to Loose).
  * `weight_doji_base`: Range `5` to `50`.

This approach makes your algorithm "aware" of market context. A Doji during high volatility (News event) will now be treated differently than a Doji during a lunch-hour lull.

