## TODO

| Week | Dates (2025)       | Main Goal                                | Key Deliverables by Sunday night                                                                                     | Status     |
|------|--------------------|------------------------------------------|----------------------------------------------------------------------------------------------------------------------|------------|
| 0    | Nov 11 – Nov 17    | Vectorized core system                   | • 6 institutional signals (momentum + mean-reversion)<br>• Fully vectorized PaperTrader<br>• Elite ensemble with thresholds | DONE       |
| 1    | Nov 18 – Nov 24    | Walk-forward & out-of-sample proof       | • Walk-forward engine (10–15 rolling windows)<br>• Full 1990–2025 performance report<br>• Equity curves per window<br>• “No curve-fitting” proof + permutation test | In Progress |
| 2    | Nov 25 – Dec 1     | Futures-ready + volatility targeting     | • Strategy running on ES, NQ, GC, CL, etc.<br>• Volatility targeting (10–12% annualized)<br>• Margin, leverage, contract rollover logic | Pending    |
| 3    | Dec 2 – Dec 8      | Live paper-trading infrastructure        | • Daily automated signal generator<br>• CSV → signal → plot → log pipeline<br>• Telegram / email alerts<br>• Daily performance dashboard | Pending    |
| 4    | Dec 9 – Dec 15     | First real-money deployment (optional)   | • $10k–$100k live account (IBKR / Rithmic / CQG)<br>• Fully automated execution<br>• Kill switches & monitoring<br>• First live P&L | Pending    |


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