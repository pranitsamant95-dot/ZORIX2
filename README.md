# Zorix 2.0

A multi-asset quantitative decision-support system: Equities, Commodities,
Forex, and Bonds/yields, with a data-provider interface ready for
Futures & Options once a real F&O data source is plugged in.

## Multi-asset upgrade (this iteration)

Scoped strictly to: Commodities/Forex/Bonds asset classes + a genuine
four-state signal system. Per instruction, this pass did **not** touch
F&O, portfolio optimization, LSTM/GRU, stacking ensembles, or the risk
engine — those already exist from a prior pass and were left as-is.

- **Forex** (`config/settings.py:ForexPairs`): USD/INR, EUR/USD, GBP/USD,
  USD/JPY. USD/INR shows a ₹ prefix (the quote *is* rupees); EUR/USD and
  GBP/USD show as plain ratios — never a blanket ₹ symbol.
- **Bonds** (`config/settings.py:BondInstruments`): US 10Y/5Y/30Y Treasury
  yield (`^TNX`/`^FVX`/`^TYX`) and 13-week T-bill (`^IRX`), all correctly
  labeled "Current Yield" (never "price") and charted as a line, not a
  candlestick. US 2Y and India 10Y are included as best-effort attempts —
  Yahoo Finance doesn't reliably publish clean tickers for either, so they
  may show "Data unavailable," which is the intended honest behavior, not
  a bug.
- **Yield-scale correction** (`data/market_data.py:apply_yield_scale`):
  `^TNX`/`^TYX`/`^FVX` report yield×10 (e.g. 42.1 for 4.21%); corrected
  once at load time so every downstream feature/model/chart sees the real
  yield.
- **Asset-aware display** (`utils/display.py`): centralizes which label
  ("Current Price" / "Exchange Rate" / "Current Yield"), currency
  formatting, and chart type (candlestick vs. line) go with which
  instrument — read from config, not hardcoded per asset class in the UI.
- **Four-state signals** (`signals/signal_engine.py` — unchanged logic,
  now exercised across all four asset classes): BUY/SELL/HOLD are genuine
  3-class model outputs; NO TRADE is a separate confidence/edge gate. HOLD
  and NO TRADE are verified as distinct states with controlled-probability
  tests in `tests/test_multi_asset.py`.

## What changed from the original `app.py` / notebook

- **Removed a leakage bug.** The old app fabricated historical sentiment
  as `0.45 * return * 10 + noise` — i.e., a "sentiment" feature derived
  from the very return it was used to predict. It's gone. Sentiment is
  now live-only (today's headlines), clearly labeled, and not used as a
  historical training feature until a real historical news archive is
  connected.
- **Target redefined** (Section 4): instead of `next close > today close`,
  the target is now a volatility-adjusted BUY/HOLD/SELL label over
  configurable horizons (1D/5D/20D), so the model isn't forced to call a
  coin-flip-sized move a "prediction."
- **Multi-model, probabilistic, 3-class** (Sections 5/6): Logistic
  Regression, Random Forest, XGBoost, LightGBM all output
  SELL/HOLD/BUY probabilities. Each model that isn't installed is
  reported as skipped — never silently omitted or faked as having run.
- **NO TRADE gating** (Section 5): if the top class doesn't clear both a
  confidence floor and a margin over the runner-up class, the signal
  engine returns `NO TRADE` with a stated reason instead of forcing a call.
- **Walk-forward validation** (Section 8): expanding-window evaluation
  (`models/walkforward.py`) reporting mean/median/std across folds
  instead of one lucky 80/20 split.
- **Real regime detection** (Section 18): `hmmlearn` HMM if installed,
  else `sklearn` GaussianMixture, else a quantile rule — always reports
  which engine actually ran.
- **Cross-asset & relative-strength features** (Sections 11/12): NIFTY/
  sector indices, India VIX/VIX, gold/silver/crude/copper, USD/INR/DXY,
  fetched via yfinance, never fabricated if a symbol fails to fetch.
- **Data-provider abstraction** (Section 39): `MarketDataProvider`,
  `OptionsDataProvider`, `FuturesDataProvider` are all interfaces you can
  implement against a different backend without touching the rest of the
  codebase.

## Architecture

```
zorix/
├── app.py                     Streamlit dashboard
├── config/settings.py         all thresholds, symbols, hyperparameters
├── data/
│   ├── market_data.py         MarketDataProvider (YahooFinanceProvider)
│   ├── options_data.py        OptionsDataProvider (Null + your adapter)
│   ├── futures_data.py        FuturesDataProvider (Null + your adapter)
│   └── news_data.py           live headlines + VADER/FinBERT sentiment
├── features/
│   ├── technical.py           returns, volatility, RSI/MACD/BB
│   ├── cross_asset.py         relative strength, cross-asset block
│   ├── regime.py              HMM / GMM / quantile regime detection
│   └── target.py              volatility-adjusted BUY/HOLD/SELL labels
├── models/
│   ├── trainer.py             LogReg / RF / XGBoost / LightGBM / LSTM
│   ├── walkforward.py         expanding-window validation
│   ├── calibration.py         Platt/isotonic calibration, Brier score
│   └── ensemble.py            OOF stacking meta-model (not yet wired into app.py)
├── signals/signal_engine.py   NO TRADE gating, structured signal object
├── utils/helpers.py           Sharpe/Sortino/max drawdown/profit factor
└── tests/test_core.py         unit tests — no network required
```

## Installation

```bash
pip install -r zorix/requirements.txt
```

Optional packages (xgboost, lightgbm, tensorflow, hmmlearn, vaderSentiment,
transformers+torch, shap) are not required — the app disables the
corresponding feature and says so in the UI/logs if they're missing.

## Running locally

```bash
streamlit run zorix/app.py
```

## Running tests

```bash
pytest zorix/tests/test_core.py -v
```
14 tests covering target construction (including a lookahead-leakage
regression test), technical/return features, walk-forward split
causality, NO TRADE gating, and the Sharpe/Sortino/max-drawdown/profit-
factor utilities. All were run and verified passing during development.

## Known limitations (be honest about these — Section 55.6)

- **No genuine historical options/futures data.** `NullOptionsProvider`
  and `NullFuturesProvider` are wired in by default and always report
  "unavailable." You said you have an F&O API — implement
  `CustomOptionsProvider` / `CustomFuturesProvider` in
  `data/options_data.py` / `data/futures_data.py` (TODOs are marked)
  and swap them in in `app.py`.
- **Stacking ensemble exists but isn't wired into `app.py` yet**
  (`models/ensemble.py`); the dashboard currently shows a simple average
  of available models' probabilities as "final," not the trained
  meta-model. Wiring it in is the natural next step.
- **No risk engine / position sizing / stop-loss** (Section 20) — Phase 5,
  not built in this pass.
- **No transaction-cost-aware backtester or portfolio module**
  (Phases 6–7) — the existing notebook's `Backtester` class was not yet
  ported/upgraded.
- **Historical sentiment is not a training feature** — only live
  same-day sentiment is shown, since no historical news archive is
  connected. This is a deliberate accuracy/leakage tradeoff, not an
  oversight.
- **This environment had no network access** while building this, so
  `yfinance` calls could not be tested end-to-end here. Every module
  that doesn't require live data (target construction, technical
  features, regime detection on synthetic OHLCV, walk-forward split,
  the full train→consensus→signal pipeline, performance metrics) **was**
  executed and verified. Run `streamlit run zorix/app.py` locally to
  confirm live data fetching against your network.

## Disclaimer

Zorix is a research and educational quantitative decision-support tool.
It does not guarantee profits, accuracy, or returns. Model outputs are
probabilistic estimates; historical backtests do not guarantee future
performance. Nothing here is financial advice.
