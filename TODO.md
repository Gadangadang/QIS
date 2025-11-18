## TODO

- [ ] Slippage model         Especially on shorts or low-liquidity           Easy
- [ ] Position sizing (e.g. Kelly, volatility targeting)         Right now it’s always 100%          Medium
- [ ] Partial fills / order queue simulation            For very large AUM          Hard
- [ ]Margin & leverage         Right now unlimited shorting power          Medium
- [ ]Dividends & corporate actions         Minor for indices, big for stocks           Medium


- [ ] In-sample excellence
- [ ] In-sample Permutation test
- [ ] Walk forward engine
- [ ] Walk forward permutation test?
- [x] Vectorize paper trader and signals?
- [x] Add multiple signals, momentum, mean reversion and emsemble;
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



