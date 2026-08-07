import pandas as pd
import numpy as np
from utils import annualized_sharpe, calmar_ratio, max_drawdown

LEVERAGE_MIN = -1.0
LEVERAGE_MAX = 1.5

def backtest_leverage_signal(
    df: pd.DataFrame,
    lev: pd.Series,
    name: str = "strategy",
    transaction_cost: float = 0.0005,  # 5 bps per unit of turnover
) -> tuple[pd.DataFrame, dict]:
    """
    Backtest a daily leverage signal.
    Input lev is 'leverage to hold from close t to close t+1'.

    transaction_cost is the cost per unit of absolute change in leverage,
    e.g. 0.0005 = 5 bps per |ΔL_t|.
    """
    # 0. Prep
    df = df.sort_index().copy()
    close = df["close"].astype(float)

    # 1. Align and clip leverage
    lev = lev.reindex(df.index).fillna(0.0).clip(LEVERAGE_MIN, LEVERAGE_MAX)

    # 2. Next day's asset return: r_{t+1} at index t
    ret_next = close.pct_change().shift(-1)

    # 3. Turnover and transaction costs
    lev_diff = lev.diff().abs()
    lev_diff.iloc[0] = 0.0  # no cost on first day
    costs = lev_diff * transaction_cost

    # 4. Strategy return (net of costs)
    gross_ret = lev * ret_next
    strat_ret = gross_ret - costs

    # Drop last day (no next return)
    strat_ret = strat_ret.iloc[:-1]
    lev = lev.iloc[:-1]
    costs = costs.iloc[:-1]
    dates = strat_ret.index

    # 5. Stats
    equity = (1.0 + strat_ret).cumprod()

    if len(equity) > 0:
        total_ret = float(equity.iloc[-1] - 1.0)
        mdd = float(max_drawdown(equity))
    else:
        total_ret = 0.0
        mdd = 0.0

    stats = {
        "name": name,
        "sharpe": annualized_sharpe(strat_ret),
        "calmar": calmar_ratio(strat_ret),
        "total_return": total_ret,
        "max_drawdown": mdd,
        "n_days": int(len(strat_ret)),
        "avg_turnover": float(lev_diff.mean()),
    }

    result_df = pd.DataFrame(
        {
            "close": close.loc[dates],
            "lev": lev,
            "ret": strat_ret,
            "equity": equity,
            "turnover": lev_diff.loc[dates],
            "costs": costs,
        },
        index=dates,
    )

    return result_df, stats

