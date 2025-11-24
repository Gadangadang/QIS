## TODO

| Week | Dates         | Main Goal                                  | Key Deliverables                                                                                                         | Difficulty   | Status   |
|------|---------------|--------------------------------------------|------------------------------------------------------------------------------------------------------------------------|--------------|----------|
| 0    | Nov 11–17     | Core vectorized system                     | 6 institutional signals + fully vectorized PaperTrader + elite ensemble                                               | Medium       | Done     |
| 1    | Nov 18–24     | Walk-forward Truth Engine                  | • Anchored rolling walk-forward (10y train / 5y test) <br>• Full 1990–2025 OOS report + stitched equity curve <br>• Permutation test skeleton | Medium-Hard  | Done     |
| 2    | Nov 25–Dec 1  | In-sample Excellence + Optimization        | Parameter optimization (grid / Bayesian / random search) inside each training window → best params per regime        | Medium       | Pending  |
| 3    | Dec 2–8       | Risk & Position Sizing Revolution          | • Volatility targeting (10–12% annualized) <br>• Kelly / Fractional Kelly <br>• Dynamic bet sizing                    | Medium       | Pending  |
| 4    | Dec 9–15      | Realistic Execution Modeling              | • Advanced slippage model (especially shorts & low-liquidity) <br>• Partial fills / order queue simulation <br>• Margin & leverage constraints | Hard         | Pending  |
| 5    | Dec 16–22     | Statistical Rigor & Overfitting Proof      | • In-sample + walk-forward permutation tests <br>• Monte Carlo regime shuffling <br>• Full execution analysis framework | Hard         | Pending  |
| 6    | Dec 23–29     | Multi-Asset Futures Migration              | • Full switch to futures (ES, NQ, GC, CL, TY, etc.) <br>• Contract rollover logic <br>• Continuous adjusted prices      | Medium-Hard  | Pending  |
| 7    | Dec 30–Jan 5  | Signal Zoo + Regime-Aware Ensemble         | Add new signals: <br>• Volatility regime <br>• Seasonality (TOM, Friday) <br>• Volume/liquidity breakouts <br>• Smart ensemble weighting | Medium       | Pending  |
| 8    | Jan 6–12      | Final Report + Go / No-Go Decision         | • Master dashboard (all assets, all periods) <br>• Final walk-forward results across 5+ futures <br>• Statistical significance report <br>• “Deploy or Kill” decision framework | Easy         | Pending  |

**Next Action (Week 2):**  
Parameter optimization inside each training window → turn –0.04% CAGR into +12–18% CAGR with zero look-ahead bias.

When you're ready, just say:  
**“Week 2 starts now.”**

Copy → paste → own your future.


- [ ] Slippage model         Especially on shorts or low-liquidity           Easy
- [ ] Position sizing (e.g. Kelly, volatility targeting)         Right now it’s always 100%          Medium
- [ ] Partial fills / order queue simulation            For very large AUM          Hard
- [ ] Margin & leverage         Right now unlimited shorting power          Medium
- [ ] Dividends & corporate actions         Minor for indices, big for stocks           Medium

- [ ] In-sample excellence
- [ ] In-sample Permutation test
- [ ] Walk forward engine
- [ ] Walk forward permutation test?
- [ ] Execution analysis framework
- [x] Vectorize paper trader and signals?
- [x] Add multiple signals, momentum, mean reversion and ensemble;
- [ ] Extend to futures
- [ ] Extend to other asset classes
- [ ] Allow for intraday trading
- [ ] Add better risk analysis framework

Other proposed models
Signal Type,Example,Typical Hold Period,Correlation
- [ ] Trend / Momentum,"ROC(60), ROC(120), MACD histogram",1–6 months,High with each other
- [ ] Volatility,Go long when VIX > 80th percentile,Days to weeks,Low
- [ ] Seasonality,"Turn-of-month, Friday effect",Days,Very Low
- [ ] Volume / Liquidity,"Volume breakout, OBV divergence",Weeks,Low
- [ ] Cross-sectional (later),Rank stocks by momentum → long top 20%,Months,N/A