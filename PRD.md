# PRD: NQ Khushi-MSSR Trading Engine

## 1. Executive Summary

**Objective:** Build an autonomous trading system for E-mini Nasdaq 100 (NQ) futures.
**Core Methodology:** Implementation of Dr. Matloob Khushi’s GA-MSSR (Genetic Algorithm Maximizing Sharpe and Sterling Ratio) framework.
**Primary Strategy:** 16-rule signal generation engine optimized via a Genetic Algorithm (GA) with a Wavelet Denoising pre-processing layer.

---

## 2. Technical Stack

* **Language:** Python 3.10+
* **Execution Engine:** `NautilusTrader` (Event-driven backtester)
* **Signal Processing:** `PyWavelets` (Denoising)
* **Optimization:** `PyGAD` (Genetic Algorithm)
* **Reference Repository:** `https://github.com/zzzac/Rule-based-forex-trading-system` (Source for the 16 rules and GA structure).

---

## 3. Mathematical Foundations

### A. Discrete Wavelet Transform (DWT) Denoising

Before indicator calculation, the raw NQ price  must be decomposed and reconstructed using a "Daubechies" (db4) wavelet to filter high-frequency noise while preserving price jumps.

### B. The SS Ratio (Sharpe-Sterling)

The fitness function for the GA is the SS Ratio (). Unlike the standard Sharpe Ratio, it penalizes the "Sterling" drawdown specifically.


---

## 4. System Architecture

### Component 1: Data Pipeline (`/data`)

* **Input:** 1-minute NQ OHLCV data.
* **Processor:** `WaveletFilter` class to output .

### Component 2: Signal Engine (`/strategies/khushi_rules.py`)

Implement 16 discrete signal rules from the reference repo, including:

1. **MA x MA Crossovers**
2. **EMA x MA Crossovers**
3. **DEMA x MA**
4. **Stochastic x Stochastic**
5. **Vortex Indicator High x Low**
6. **Ichimoku x Price**
7. **Bollinger Band Mean Reversion**
8. *(...And other rules defined in `tradingrule.py` of the reference repo)*

### Component 3: The GA Optimizer (`/optimizers/ga_mssr.py`)

* **Chromosome:** Vector of parameters for the 16 rules (e.g., lookback periods for EMAs, RSI thresholds).
* **Fitness Function:** Backtest execution over a rolling window where .

---

## 5. Implementation Requirements for Claude Code

1. **Refactor zzzac Repo:** Extract the logic from `ga.py` and `tradingrule.py` from the referenced GitHub repo.
2. **NQ Calibration:**
* Set Tick Size to `0.25`.
* Set Tick Value to `$5.00` (or `$20.00` for Big NQ).
* Account for $2.40 round-turn commission.


3. **Risk Management:**
* Hard Stop: $400 daily loss per contract.
* Max Drawdown Constraint: GA must discard any chromosome resulting in a >15% drawdown.


4. **NautilusTrader Integration:**
* Define a `KhushiStrategy(Strategy)` class.
* Implement `on_bar` logic to process denoised signals.



---

## 6. Success Metrics

* **Primary:** SS Ratio > 1.5.
* **Secondary:** Profit Factor > 1.3.
* **Robustness:** Walk-forward analysis must show positive returns on OOS (Out-of-Sample) data.

---

## 7. Immediate Tasks for Agent

1. Initialize project structure: `/data`, `/strategies`, `/optimizers`, `/logs`.
2. Install dependencies: `nautilus_trader`, `PyWavelets`, `PyGAD`, `pandas`.
3. Implement `SSR.py` class to calculate the Sharpe-Sterling ratio from a PnL series.
4. Map the 16 rules from `tradingrule.py` into a vectorized format compatible with NQ 1m bars.

---

### Instructions for the User:

Once you have created this `PRD.md` file, open your terminal and run your agent (e.g., `claude-code`) and type:

> *"Read PRD.md. I have the reference repo files ready. Please begin by implementing Step 1: the Data Pipeline and the Wavelet Denoising filter."*

Would you like me to help you draft the specific **`on_bar`** logic that tells the agent exactly how to combine the 16 rules into a single trade decision?