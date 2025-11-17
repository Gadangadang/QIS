
Improvement             Why             Difficulty
* Transaction costs           Even 1–2 bps kills many strategies          Easy
*Slippage model         Especially on shorts or low-liquidity           Easy
*Position sizing (e.g. Kelly, volatility targeting)         Right now it’s always 100%          Medium
* Partial fills / order queue simulation            For very large AUM          Hard
* Margin & leverage         Right now unlimited shorting power          Medium
* Dividends & corporate actions         Minor for indices, big for stocks           Medium


1. In-sample excellence
2. In-sample Permutation test
3. Walk forward test
4. Walk forward permutation test?

GitHub Copilot: Short answer: partly implemented. Below I map each step to the current code, explain what's present vs missing, and give concise suggestions to fully implement each (including safe choices for time-series permutation).

**1. In-sample excellence**  
- **Status:** Partially implemented.  
- **Why:** `backtest/backtest_engine.py::run_train_test()` computes basic in-sample metrics (Sharpe, max drawdown) for the train slice and prints them. That gives a quick in-sample performance check.  
- **What’s missing for “excellence”:** automated parameter search / model tuning (grid/opt), cross-validated metric aggregation, walk-overfitting checks, transaction-cost sensitivity, and reporting (tables/plots).  
- **Next steps (small):** add a parameter sweep wrapper that runs `signal.generate` with different params, records train metrics, and selects top candidates. Save results to CSV for inspection.

**2. In-sample permutation test (significance test inside train)**  
- **Status:** Not implemented.  
- **Why:** There is no code that computes a null distribution (permuted signals/labels) to produce p-values.  
- **How to do it (recommended approach for time series):**
  - Metric: choose a scalar metric (e.g., annualized Sharpe or cumulative return).
  - Null model: permute the *signal* sequence or do block-bootstrap of returns — but plain random shuffling breaks time dependence and can inflate significance. Prefer block permutations or circular shift (phase randomization) to preserve autocorrelation structure.
  - Algorithm (pseudo):
    ```
    observed = metric(signal, prices)
    for i in range(N):
        permuted_signals = block_permute(signal, block_size)
        perm_metric[i] = metric(permuted_signals, prices)
    p_value = (sum(perm_metric >= observed) + 1) / (N+1)
    ```
  - Practical defaults: N=500–2000, block_size between 5–20 days depending on data frequency.
- **Where to add:** new helper `backtest/stat_tests.py` and a `run_in_sample_permutation(signal_func, df, metric, N, block_size)` wrapper.

**3. Walk-forward test (rolling out-of-sample)**  
- **Status:** Partially implemented.  
- **Why:** We implemented `split_train_test()` and `run_train_test()` which perform a single train / single test split and allow providing `lookback` history. That supports a single out-of-sample test, which is the simplest holdout.  
- **What’s missing for full walk-forward:** a rolling/expanding window loop that: retrains (or re-computes signals) on each training window, then tests on the immediately following test window, and aggregates performance across all folds. Also logging per-fold trades and combined equity curve.  
- **How to implement (high level):**
  - Choose `train_window`, `test_window`, `step` (e.g., train=60% of history or fixed N days, test=20 days, step= test_window or smaller for overlapping folds).
  - Loop:
    - train_slice = df[i : i+train_window]
    - test_slice = df[i+train_window - lookback : i+train_window + test_window]
    - compute signals on test_slice (so indicators can initialize using lookback)
    - run paper trader with `start_date = first date of test_only` and collect trader.summary() and trades
    - advance i by `step`
  - Aggregate folds: concatenate `PortfolioValue` slices (careful to rebase), compute overall metrics and per-fold stats.
- **Where to add:** extend `backtest/backtest_engine.py` with `run_walk_forward(...)` that returns per-fold summaries plus aggregated results.

**4. Walk-forward permutation test (significance per fold / overall)**  
- **Status:** Not implemented.  
- **Why:** No permutation logic exists at all, and no code orchestrates nested/conditional testing inside walk-forward folds.  
- **Options (two meaningful approaches):**
  - Per-fold permutation: inside each walk-forward fold, run the in-sample permutation test or permute test signals to get fold-level p-values — expensive but granular.  
  - Global permutation across all folds: regenerate full sequence of permuted signals (with block-preserving permutations) and run the entire walk-forward pipeline on the permuted sequence to compute a distribution of aggregate metrics. This controls the whole pipeline’s selection bias better but is much more computationally expensive.  
- **Practical suggestion:** start with per-fold permutation (smaller N per fold), then move to a smaller number of global permutations if you need stricter validation. Use parallelization (multiprocessing) for permutations.

Additional important points
- **Trading vs testing cadence:** You are correct — a backtest uses only historical data. For live trading, the live runner should compute signals on available data up to "today" and then place orders for today/next N days. The code already enforces non-lookahead by using `Position.shift(1)` in `PaperTrader`. When you build walk-forward or live pipelines, always ensure signal computation uses only past data up to each timestamp.
- **Transaction costs / realistic execution:** Add commission/slippage modeling to both backtest and paper trader to avoid optimistic results.
- **Multiple testing:** If you perform many parameter sweeps and many permutations, apply corrections (control family-wise error / false discovery). Permutation p-values mitigate multiple testing, but you should still be cautious.

Concrete minimal next steps I can implement for you (pick one)
- Add a basic rolling `run_walk_forward()` in `backtest/backtest_engine.py` that runs multiple folds and aggregates results. (Low effort, high value.)
- Add an in-sample permutation test helper `backtest/stat_tests.py` implementing block permutations. (Moderate effort.)
- Add a global walk-forward permutation runner that runs the whole pipeline on permuted signals and returns an empirical p-value (higher effort, compute-heavy).

Which of the three would you like me to implement first? If you want the quick win, I recommend adding `run_walk_forward()` (rolling test) and a small CLI in `live/run_daily.py` to run it and save per-fold CSVs.  - Practical defaults: N=500–2000, block_size between 5–20 days depending on data frequency.
- **Where to add:** new helper `backtest/stat_tests.py` and a `run_in_sample_permutation(signal_func, df, metric, N, block_size)` wrapper.

**3. Walk-forward test (rolling out-of-sample)**  
- **Status:** Partially implemented.  
- **Why:** We implemented `split_train_test()` and `run_train_test()` which perform a single train / single test split and allow providing `lookback` history. That supports a single out-of-sample test, which is the simplest holdout.  
- **What’s missing for full walk-forward:** a rolling/expanding window loop that: retrains (or re-computes signals) on each training window, then tests on the immediately following test window, and aggregates performance across all folds. Also logging per-fold trades and combined equity curve.  
- **How to implement (high level):**
  - Choose `train_window`, `test_window`, `step` (e.g., train=60% of history or fixed N days, test=20 days, step= test_window or smaller for overlapping folds).
  - Loop:
    - train_slice = df[i : i+train_window]
    - test_slice = df[i+train_window - lookback : i+train_window + test_window]
    - compute signals on test_slice (so indicators can initialize using lookback)
    - run paper trader with `start_date = first date of test_only` and collect trader.summary() and trades
    - advance i by `step`
  - Aggregate folds: concatenate `PortfolioValue` slices (careful to rebase), compute overall metrics and per-fold stats.
- **Where to add:** extend `backtest/backtest_engine.py` with `run_walk_forward(...)` that returns per-fold summaries plus aggregated results.

**4. Walk-forward permutation test (significance per fold / overall)**  
- **Status:** Not implemented.  
- **Why:** No permutation logic exists at all, and no code orchestrates nested/conditional testing inside walk-forward folds.  
- **Options (two meaningful approaches):**
  - Per-fold permutation: inside each walk-forward fold, run the in-sample permutation test or permute test signals to get fold-level p-values — expensive but granular.  
  - Global permutation across all folds: regenerate full sequence of permuted signals (with block-preserving permutations) and run the entire walk-forward pipeline on the permuted sequence to compute a distribution of aggregate metrics. This controls the whole pipeline’s selection bias better but is much more computationally expensive.  
- **Practical suggestion:** start with per-fold permutation (smaller N per fold), then move to a smaller number of global permutations if you need stricter validation. Use parallelization (multiprocessing) for permutations.

Additional important points
- **Trading vs testing cadence:** You are correct — a backtest uses only historical data. For live trading, the live runner should compute signals on available data up to "today" and then place orders for today/next N days. The code already enforces non-lookahead by using `Position.shift(1)` in `PaperTrader`. When you build walk-forward or live pipelines, always ensure signal computation uses only past data up to each timestamp.
- **Transaction costs / realistic execution:** Add commission/slippage modeling to both backtest and paper trader to avoid optimistic results.
- **Multiple testing:** If you perform many parameter sweeps and many permutations, apply corrections (control family-wise error / false discovery). Permutation p-values mitigate multiple testing, but you should still be cautious.

Concrete minimal next steps I can implement for you (pick one)
- Add a basic rolling `run_walk_forward()` in `backtest/backtest_engine.py` that runs multiple folds and aggregates results. (Low effort, high value.)
- Add an in-sample permutation test helper `backtest/stat_tests.py` implementing block permutations. (Moderate effort.)
- Add a global walk-forward permutation runner that runs the whole pipeline on permuted signals and returns an empirical p-value (higher effort, compute-heavy).

Which of the three would you like me to implement first? If you want the quick win, I recommend adding `run_walk_forward()` (rolling test) and a small CLI in `live/run_daily.py` to run it and save per-fold CSVs.