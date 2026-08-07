import numpy as np
import pandas as pd

def annualized_sharpe(daily_returns: pd.Series,
                      rf: float = 0.0,
                      periods_per_year: int = 252) -> float:
    excess = daily_returns - rf / periods_per_year
    mu = excess.mean()
    sigma = excess.std(ddof=1)
    if sigma == 0 or np.isnan(sigma):
        return 0.0
    return float(np.sqrt(periods_per_year) * mu / sigma)

def max_drawdown(equity_curve: pd.Series) -> float:
    roll_max = equity_curve.cummax()
    drawdown = equity_curve / roll_max - 1.0
    return float(drawdown.min())

def calmar_ratio(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    equity = (1.0 + daily_returns).cumprod()
    mdd = max_drawdown(equity)
    if len(equity) == 0 or mdd == 0:
        return 0.0
    cagr = equity.iloc[-1] ** (periods_per_year / len(equity)) - 1.0
    return float(cagr / abs(mdd))


# -------------------------------
# Helper: RSI (rolling, no lookahead)
# -------------------------------

def _calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Simple rolling RSI (no lookahead).
    Uses rolling mean of gains/losses (not Wilder EMA).
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

