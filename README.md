# Quanta Fellowship — Systematic QQQ Strategy

Submission for the Quanta Fellowship: a daily leverage-targeting strategy on QQQ,
developed under an explicitly AI-assisted research protocol.

Signal ideas and code were drafted in collaboration with Gemini 3.0. Every reported
number was produced by the backtest engine in `src/` running on data — see
[Methodology](#methodology--ai-assisted-research).

---

## Result

Final signal: `sig_igl_2plus_combo` — a volatility-targeted blend of three components
(trend/mean-reversion mix 46%, volume-compression 35%, turn-of-month 19%), scaled by a
Garman–Klass realized-volatility anchor and squashed through `tanh`. Leverage clipped to
[−1.0, 1.5]. Costs: 5 bps per unit of |Δleverage|. Benchmark is buy-and-hold at 1.0x,
charged zero cost (i.e. the benchmark is handicapped in its own favour).

### Blind holdout — 2022-01-01 to 2025-12-05 (985 trading days)

| | Strategy | Buy-and-hold |
|---|---:|---:|
| Sharpe | **0.95** | 0.62 |
| Calmar | **0.76** | 0.37 |
| Total return | **62.2%** | 59.5% |
| Max drawdown | **-17.4%** | -34.0% |
| Annualized vol | 14.1% | 23.8% |
| Avg daily turnover | 4.7% | 0% |

The strategy beat buy-and-hold on every metric on unseen data, delivering slightly higher
total return at roughly 60% of the volatility and half the drawdown. The edge is risk
control, not return generation: the volatility anchor de-levers into turbulence, which
costs participation in sharp rallies and avoids the deep drawdowns.

### All windows

| Window | Strategy Sharpe | Buy-hold Sharpe | Strategy MaxDD | Buy-hold MaxDD | n |
|---|---:|---:|---:|---:|---:|
| Train — 2000-01-01 to 2015-12-31 | 0.39 | 0.20 | -37.2% | -83.0% | 4025 |
| Validation — 2016-01-01 to 2021-12-31 | 1.57 | 1.15 | -12.0% | -28.6% | 1510 |
| Blind holdout — 2022-01-01 to 2025-12-05 | 0.95 | 0.62 | -17.4% | -34.0% | 985 |

Two things worth reading off this table honestly:

**Performance decayed out of sample.** Validation Sharpe 1.57 fell to 0.95 on the holdout.
That gap is the cost of having selected among 30+ candidate signals on the validation
window. The holdout number is the one to believe.

**The train window is the hard one.** 2000-2015 contains the dot-com collapse and the
financial crisis. Sharpe 0.39 there is unimpressive in absolute terms but doubles the
benchmark's 0.20, and cuts max drawdown from -83% to -37%. The strategy's relative
advantage is largest exactly where the benchmark suffers most.

---

## Data and splits

| Split | Range | Use |
|---|---|---|
| Train | 2000-01-03 - 2015-12-31 | Signal screening |
| Validation | 2016-01-01 - 2021-12-31 | Composite selection and weighting |
| Blind holdout | 2022-01-01 - 2025-12-05 | Evaluated once, after freezing |

Source files are `qqq_train_validation.csv` (2000-2021) and `qqq_blind_holdout.csv`. The
provided holdout file begins 2021-01-04, which overlaps the training data; the holdout is
therefore evaluated from 2022-01-01, where the training data ends, so the two are
disjoint. Leverage is computed over the full holdout history and the results are sliced
afterward, preserving rolling-window warmup.

---

## Method

**Signal library.** 30+ candidate signals in `src/signals.py` spanning trend/momentum,
mean reversion, volatility regime, volume/flow, and calendar-seasonal effects. Each is a
pure function `df -> leverage series`, so any signal can be swapped into the backtester
without touching it.

**Selection.** Candidates were screened on train (2000–2015), then survivors evaluated on
validation (2016–2020). The final composite was chosen on validation performance and
frozen before the holdout was evaluated.

**Composite.** Three alpha components blended at fixed weights, then scaled by a
Garman–Klass volatility-target anchor (20-day window, 9.8% annualized target) and passed
through `1.5·tanh(·)` to bound leverage smoothly rather than by hard clipping.

**Backtest conventions.** Leverage at close *t* applies to the close(*t*) → close(*t+1*)
return, so the signal is causal by construction — no shift is applied post hoc. Costs are
charged on |Δleverage| daily. Final day is dropped (no forward return).

---

## Methodology — AI-assisted research

This project was developed as an explicitly AI-assisted exercise. Being specific, since
"used an LLM" spans a wide range:

**Gemini 3.0 was used for:** brainstorming candidate signal formulations, drafting and
refactoring implementation code, and explaining unfamiliar techniques.

**Gemini 3.0 was not used for:** producing any reported number, selecting which signals
entered the composite, or evaluating the holdout.

The distinction matters. A language model can propose that volume-compression might
combine well with a volatility target. It cannot tell you whether it did. Every figure in
this README traces to `src/backtest_engine.py` running on data, and `prompts/` is
published so the provenance of the ideas is auditable rather than implied.

Signals are named after a CS:GO-themed prompting persona (`sig_igl_*`, `sig_csgo_*`,
`sig_flow_*`), documented in `persona_CSGO.md`. The naming is cosmetic; the functions are
ordinary technical signals.

---

## Structure

```
data/raw/     Provided datasets
notebooks/    01 signal brainstorming · 02 validation · 03 portfolio assembly · 04 final backtest
src/          signals.py · backtest_engine.py · utils.py
prompts/      Record of Gemini 3.0 interactions
results/      Logs and plots
```

```bash
pip install -r requirements.txt
# place the two CSVs in data/raw/
jupyter lab notebooks/     # run 01 -> 04 in order
```

---

## Caveats

- **Single instrument, single regime path.** One asset over one 25-year sample. The
  holdout is one draw, not a distribution — it bounds overfitting, it does not establish
  a durable edge.
- **Selection pressure.** 30+ signals were screened against the same validation window.
  The validation-to-holdout decay (1.54 → 0.95) is the visible cost of that.
- **Cost model is a simplification.** Flat 5 bps on turnover ignores spread dynamics,
  market impact, financing cost on leverage, and borrow on short exposure — the latter
  matters since leverage ranges to −1.0.
- **Close-to-close execution.** Assumes fills at the close with no slippage or gap risk.
- **Benchmark is charged zero cost.** Conservative in the strategy's disfavour, but it
  means the comparison is not perfectly apples-to-apples.