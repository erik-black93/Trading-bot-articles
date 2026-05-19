# Article 5: Markov Chains for Regime-Based Trading

## What the article is discussing

- **Brief Overview**: The article presents Markov Chain modeling as a framework for identifying and trading market regimes (Bull, Bear, Sideways) based on conditional state dependencies rather than naive independence assumptions. It provides complete implementation code and explains why regime-switching models outperform indicator-based trading.

- **Key Focus Areas**:
  - Why independence assumptions fail in financial markets
  - State definition and transition matrix estimation from historical data
  - Multi-step regime forecasting using the Chapman-Kolmogorov equation
  - Converting regime probabilities into walk-forward trading signals
  - Hidden Markov Models (HMMs) for latent regime inference

- **Scope**: Quantitative research for systematic trading strategy development, applicable to equity indices, individual securities, and any asset with regime structure. Bridges gap between academic probability theory and production trading systems.

---

## Key learned material

### 1. The Markov Property and Local Conditional Dependence

- Markets violate independence assumptions: a loan cannot jump from "current" to "90+ days delinquent" in one period, and a market cannot transition through impossible regime sequences.
- The Markov property states: `P(Xₙ₊₁ = s | X₀, X₁, ..., Xₙ) = P(Xₙ₊₁ = s | Xₙ)` — next state depends only on current state, not entire history.
- This single assumption produces dramatically more accurate models than independence-based approaches by capturing state persistence and realistic transition constraints.
- For trading: market regime next month depends only on whether we're currently in Bull, Bear, or Sideways regime — not on price history before the regime began.

### 2. State Definition and Transition Matrix Estimation

- States must be **mutually exclusive and collectively exhaustive**: every observation maps to exactly one state with no gaps or overlaps.
- Common regime definitions:
  - **Trend regimes**: Bull (20-day return > +2%), Bear (< -2%), Sideways (between)
  - **Volatility regimes**: Low / Medium / High based on rolling realized volatility thresholds
  - **Liquidity regimes**: High / Low based on bid-ask spreads
  - **Credit regimes**: Risk-on / Risk-off based on spreads or cross-asset correlations

- **Transition matrix estimation** uses maximum likelihood: `P̂(i,j) = Count(i → j) / Count(i → any state)`
- Each row sums to 1.0 (system must transition somewhere, including self-loops).
- Implementation: count observed state transitions in historical data, normalize by row.

### 3. Multi-Step Forecasting via Chapman-Kolmogorov Equation

- Compute n-step transition probabilities by raising transition matrix to the nth power: `P^(n)(i,j) = [Pⁿ]ᵢⱼ`
- Extract any entry to get probability of moving from state i to state j after exactly n periods.
- **Stationary distribution** π: long-run regime mix solved via `π = π × P` or linear algebra (`(P.T - I)π = 0`).
  - Tells you baseline regime probabilities as n → ∞
  - Reveals rare regimes; betting heavily on rare regimes = uncompensated tail risk
  
- **Actionable signals**: If currently Bull, use P^5 to quantify 5-step-ahead probability of returning to Bull, shifting to Bear, or entering Sideways — informs position horizon and exit rules.

### 4. Trade Signal Construction and Walk-Forward Validation

- **Signal design**: Convert regime probability vector into position size
  - Simple: Long in Bull, Short in Bear, Flat in Sideways
  - Sophisticated: `signal = P(Bull) - P(Bear)` with threshold bands (e.g., |signal| < 0.1 = neutral, > 0.3 = full position)
  
- **Walk-forward critical requirement**: Re-estimate transition matrix only from historical data available at each decision point — never use future data to estimate transition probabilities
  - Prevents lookahead bias and overfitting
  - Simulates production conditions (decisions made with information known at time t)
  
- **Backtest structure**:
  1. Estimate P from lookback window (e.g., 252 trading days)
  2. Generate signal from current regime using estimated P
  3. Execute position on next bar
  4. Roll forward one period; repeat
  
- Output metrics: Annualized Sharpe ratio, Maximum Drawdown, Annual Return, Regime Distribution (% time in each state)

### 5. Critical Model Assumptions and Limitations

- **Markov property violation**: Markets exhibit longer-range dependencies (multi-month trends, seasonal patterns); next regime may depend on history beyond last state.
  - **Mitigation**: Use higher-order models or add external features, but complexity-variance tradeoff applies

- **Time homogeneity violation**: Transition probabilities are not stationary; Bull→Bear probability in 2008 ≠ 2023.
  - **Mitigation**: Re-fit transition matrix on rolling windows (e.g., re-estimate quarterly or monthly) rather than once over full history

- **Estimation risk**: Rare transitions require large sample sizes for reliable estimates. With few observed transitions between states, MLE estimate has high variance.
  - **Mitigation**: Merge rare states, use Bayesian priors, or increase lookback window; validate via bootstrap

- **State definition risk**: Manual labeling (e.g., "Bull = returns > 2%") can be arbitrary; regime boundaries may not align with true market structure.
  - **Mitigation**: Iterate regime thresholds, use Hidden Markov Models to infer regimes from data

---

## How this helps build a profitable trading bot / system

- **Problem Solved**: Traditional technical indicators (RSI, moving averages) treat each day independently and ignore the structured persistence of market regimes. Markov Chains solve this by explicitly modeling state transitions and quantifying multi-step regime probabilities—enabling regime-aware position sizing and exit rules.

- **Direct Applications**:
  1. **Regime-aware portfolio allocation**: Size positions based on confidence in regime persistence. During strong Bull with high self-loop probability (P(Bull→Bull) = 0.85), allocate full capital; if P(Bull→Bull) = 0.55, reduce exposure.
  2. **Multi-horizon trading signals**: Use Chapman-Kolmogorov equation to trade different timeframes differently. Day-traders focus on 1-step probabilities; swing traders use 5-10 step forecasts; position traders use stationary distribution as long-term bias.

- **Risk Mitigation**:
  - **Overfitting reduction**: Walk-forward framework uses only past data at each decision point; prohibits lookahead bias common in indicator-based strategies
  - **Curve-fitting avoidance**: Markov model parameters (state definitions, transition estimates) are estimated from observable market data via MLE, not optimized against returns
  - **Tail risk quantification**: Stationary distribution identifies rare regimes; prevents overweighting low-probability states

- **Scalability**: 
  - Regime framework scales cleanly across asset classes (equities, commodities, currencies, crypto); only require defining asset-specific state boundaries
  - Multi-step forecasting is O(3×3) matrix multiplication regardless of portfolio size
  - Rolling window re-estimation adds linear computational cost but prevents model drift

- **Operational Benefits**:
  - **Monitoring**: Regime state is directly observable; alerts trigger when market transitions (Bull→Bear) for risk management
  - **Safety**: Binary regime framework prevents arbitrary leverage; position size is simple function of regime probability
  - **Interpretability**: Traders can explain position logic ("We are 65% confident market remains Bull based on transition history; sizing position accordingly") vs. black-box indicators

---

## Practical takeaways for implementation

1. **Define states** for your asset: Use domain knowledge + rolling return statistics
   - Trend regimes: Calculate 20-day rolling return; set Bull threshold (+2%), Bear threshold (-2%), residual = Sideways
   - Volatility regimes: Calculate 20-day rolling realized volatility; set Low/Med/High via percentiles (e.g., 33rd/67th)
   - Ensure mutual exclusivity: no observation belongs to multiple states

2. **Estimate transition matrix from historical data**:
   - Gather 5-10 years of price data (minimum)
   - Label each bar with its regime state
   - Count transitions: loop through state sequence, increment `counts[current_state][next_state]`
   - Normalize: `P[i][j] = counts[i][j] / sum(counts[i][:])` for each row

3. **Compute stationary distribution and multi-step forecasts**:
   - Solve `(P.T - I) × π = 0` subject to `sum(π) = 1` to get baseline regime mix
   - Use `numpy.linalg.matrix_power(P, n)` to forecast n-step-ahead probabilities
   - Compare multi-step forecast vs. stationary distribution to identify trading opportunities (e.g., current Bear state but 3-step forecast shows mean-reversion to Bull)

4. **Design signal and backtest with walk-forward logic**:
   - Generate signal: `signal = P[current_state][0] - P[current_state][1]` (Bull prob - Bear prob)
   - Define position rule: if signal > 0.3, long 1.0; if signal < -0.3, short 1.0; else neutral
   - **Backtest loop**:
     ```
     for each bar t:
         estimate_P using historical_states[t-252:t]
         current_regime = state_at[t]
         signal = compute_signal(P, current_regime)
         position[t+1] = generate_position(signal)
     strategy_return = position * market_return
     ```
   - Calculate metrics: Sharpe ratio, max drawdown, regime distribution

5. **Validate robustness**:
   - Test different state definitions (vary Bull/Bear thresholds by ±50bps)
   - Test different lookback windows (200, 252, 300 days)
   - Verify no forward-looking data enters transition matrix estimation
   - Out-of-sample test: train on first 70% of data, backtest on final 30%

---

## Notes

- **Limitations of observable Markov Chains**: 
  - Regime labels are inferred from price post-hoc; they do not causally explain returns. True economic regime (credit conditions, volatility regime) may be latent and mismatched to rolling-return-based labels.
  - Solution: Use Hidden Markov Models (Baum-Welch algorithm) to infer latent regimes from returns + emission variables (realized volatility, spreads, macro data)

- **When Markov Chains may not apply**:
  - High-frequency trading (< 1-minute): regime changes too fast relative to data frequency; regime persistence assumption breaks down
  - Single-stock trading with low liquidity: state transitions driven by idiosyncratic shocks, not regime dynamics
  - Assets with structural breaks: regime transition probabilities change due to new regulatory environment, market microstructure changes, etc.; rolling re-estimation mandatory

- **Advanced directions**:
  - **Hidden Markov Models (HMM)**: Use Baum-Welch algorithm to learn hidden regime structure and emission probabilities from observable returns. Viterbi algorithm decodes most-likely hidden state path.
  - **Multi-step signals**: Combine Chapman-Kolmogorov forecasts across multiple timeframes (1-day, 5-day, 20-day forecasts) for multi-asset regime consensus
  - **Regime-conditioned models**: Fit separate asset-return models within each regime (e.g., mean-reversion in Sideways, momentum in Bull); use Markov framework to weight models by regime probability

- **Connection to other concepts**:
  - Complements mean-reversion strategies in Sideways regimes; enhances momentum strategies in Bull/Bear regimes
  - Extends beyond single-indicator trading (RSI, moving average) by explicitly modeling state persistence
  - Related to regime-switching GARCH models in academic literature; Markov Chains provide simpler, more interpretable alternative for practitioners

---

## Implementation Reference

**Python dependencies**: `numpy`, `pandas`, `yfinance`, `scikit-learn`

**Complete working example** provided in article includes:
- `define_market_states()`: Label price data with regime states
- `estimate_transition_matrix()`: MLE estimation of transition probabilities
- `multi_step_transition()`: Chapman-Kolmogorov matrix powers
- `compute_trading_signal()`: Signal generation from regime probabilities
- `backtest_markov_strategy()`: Walk-forward validation framework
- `MarkovChainTradingSystem` class: Production-ready implementation with Sharpe/Drawdown calculation

**Configuration recommendations**:
- **Lookback window**: 252 days (1 year) balances recency vs. stability; shorter = faster regime adaptation but noisier estimates
- **Regime window**: 20 days for rolling return calculation; adjust for asset volatility
- **Bull/Bear thresholds**: Start with ±2% for broad equities; tighten for high-vol assets, loosen for low-vol assets
- **Signal thresholds**: Use |0.3| as full position boundary; smooth interpolation between -0.3 and +0.3 for partial positions
