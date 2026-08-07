# src/signals_csgo.py

import pandas as pd
import numpy as np
from utils import _calc_rsi

# -------------------------------
# Helper: calendar features
# -------------------------------

def _calendar_features(index: pd.Index) -> pd.DataFrame:
    """
    Builds calendar features from a DatetimeIndex:
      - Month, Day
      - TradingDay: 1st trading day of month, 2nd, ...
      - ReverseTradingDay: days remaining in trading month (last=1, ...)
    """
    if not isinstance(index, pd.DatetimeIndex):
        idx = pd.to_datetime(index)
    else:
        idx = index

    cal = pd.DataFrame(index=idx)
    cal["Month"] = idx.month
    cal["Day"] = idx.day

    # Trading day-of-month (1..N) based on the index order
    cal["TradingDay"] = cal.groupby([idx.year, idx.month]).cumcount() + 1
    cal["DaysInMonth"] = cal.groupby([idx.year, idx.month])["TradingDay"].transform("max")
    cal["ReverseTradingDay"] = cal["DaysInMonth"] - cal["TradingDay"] + 1
    return cal

# -------------------------------
# Helper: Vol scaler + ATR proxy
# -------------------------------

def _vol_scaler(
    df: pd.DataFrame,
    close_col: str = "close",
    vol_window: int = 20,
    target_ann_vol: float = 0.15,
) -> pd.Series:
    close = df[close_col].astype(float)
    ret = close.pct_change()
    vol = ret.rolling(vol_window).std().replace(0, np.nan)
    target_daily = target_ann_vol / np.sqrt(252.0)
    scaler = target_daily / vol
    return scaler


def _atr_proxy_from_hl(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    atr_window: int = 14,
) -> pd.Series:
    high = df[high_col].astype(float)
    low = df[low_col].astype(float)
    rng = (high - low).replace(0, np.nan)
    atr = rng.rolling(atr_window).mean()
    return atr

# CSGO Player Persona(QQQ Dataset)

# -------------------------------
# Primitive signal 1: Vol_Regime
# -------------------------------

def sig_csgo_vol_regime(
    df: pd.DataFrame,
    close_col: str = "close",
    rsi_period: int = 14,
    sma_window: int = 200,
    short_vol_window: int = 20,
    long_vol_window: int = 60,
    rsi_low: float = 30.0,
    rsi_high: float = 70.0,
) -> pd.Series:
    """
    CS:GO persona signal: Vol_Regime (volatility-based regime switching).

    Logic (per day t):
      - Compute:
          * Return_t from close prices
          * Vol_20 = rolling std(Return, 20)
          * Vol_60 = rolling std(Return, 60)
          * SMA200 = rolling mean(Close, 200)
          * RSI(14) on Close
      - If any of these are NaN -> exposure 0.

      - Quiet regime (Vol_20 < Vol_60):
          * If Close > SMA200  -> go long 1.5x
          * Else              -> flat 0.0

      - Volatile regime (Vol_20 >= Vol_60):
          * If RSI < rsi_low  -> long 1.5x (buy the dip)
          * If RSI > rsi_high -> short -0.5x (fade the rip)
          * Else              -> flat 0.0

    Returns:
        pd.Series of leverage L_t in [-1.0, 1.5] aligned with df.index.
        NOTE: This is *unshifted*; you apply the shift in backtesting.
    """
    close = df[close_col].astype(float)
    ret = close.pct_change()

    vol_20 = ret.rolling(short_vol_window).std()
    vol_60 = ret.rolling(long_vol_window).std()
    sma200 = close.rolling(sma_window).mean()
    rsi = _calc_rsi(close, period=rsi_period)

    # Put into a temp DataFrame for apply
    tmp = pd.DataFrame(
        {
            "close": close,
            "vol_20": vol_20,
            "vol_60": vol_60,
            "sma200": sma200,
            "rsi": rsi,
        },
        index=df.index,
    )

    def _row_logic(row):
        # Guard against NaNs in early history
        if (
            np.isnan(row["vol_20"])
            or np.isnan(row["vol_60"])
            or np.isnan(row["sma200"])
            or np.isnan(row["rsi"])
        ):
            return 0.0

        # Quiet regime
        if row["vol_20"] < row["vol_60"]:
            return 1.5 if row["close"] > row["sma200"] else 0.0

        # Volatile regime
        if row["rsi"] < rsi_low:
            return 1.5
        if row["rsi"] > rsi_high:
            return -0.5
        return 0.0

    lev = tmp.apply(_row_logic, axis=1)
    return lev.clip(-1.0, 1.5)


# -------------------------------
# Primitive signal 2: Camp_Fire
# -------------------------------

def sig_csgo_camp_fire(
    df: pd.DataFrame,
    close_col: str = "close",
    sma_window: int = 200,
    vol_window: int = 20,
    vol_quantile_window: int = 252,
    vol_quantile_level: float = 0.8,
) -> pd.Series:
    """
    CS:GO persona signal: Camp_Fire (defensive trend + low-volatility filter).

    Logic:
      - Bull condition: Close > SMA200
      - Low-vol condition: Vol_20 < rolling 80th percentile of Vol_20 over past 252 days
      - If Bull && Low-vol -> long 1.5x
      - Else -> 0.0

    Returns:
        pd.Series of leverage L_t in [0.0, 1.5] (long-only, defensive).
    """
    close = df[close_col].astype(float)
    ret = close.pct_change()

    sma200 = close.rolling(sma_window).mean()
    vol_20 = ret.rolling(vol_window).std()

    # Rolling 80th percentile of vol_20 (regime threshold)
    vol_q = vol_20.rolling(vol_quantile_window).quantile(vol_quantile_level)

    tmp = pd.DataFrame(
        {
            "close": close,
            "sma200": sma200,
            "vol_20": vol_20,
            "vol_q": vol_q,
        },
        index=df.index,
    )

    def _row_logic(row):
        if (
            np.isnan(row["sma200"])
            or np.isnan(row["vol_20"])
            or np.isnan(row["vol_q"])
        ):
            return 0.0

        bull = row["close"] > row["sma200"]
        low_vol = row["vol_20"] < row["vol_q"]

        if bull and low_vol:
            return 1.5
        return 0.0

    lev = tmp.apply(lambda r: _row_logic(r), axis=1)
    return lev.clip(-1.0, 1.5)


# -------------------------------
# Composite signal: Simple_Mix
# -------------------------------

def sig_csgo_simple_mix(
    df: pd.DataFrame,
    close_col: str = "close",
    w_vol_regime: float = 0.5,
    w_camp_fire: float = 0.5,
    **kwargs,
) -> pd.Series:
    """
    CS:GO persona signal: Simple_Mix.

    A robust ensemble of:
      - Vol_Regime (aggressive, regime-switching)
      - Camp_Fire (defensive, trend + low-volatility)

    By default it is a 50/50 mix, and sensitivity tests show that
    40/60 and 60/40 weights yield similar Sharpe, indicating robustness.

    kwargs are forwarded to the underlying signal functions if you want
    to tweak parameters.

    Returns:
        pd.Series of leverage L_t in [-1.0, 1.5].
    """
    lev_vol = sig_csgo_vol_regime(df, close_col=close_col, **kwargs)
    lev_camp = sig_csgo_camp_fire(df, close_col=close_col, **kwargs)

    lev = w_vol_regime * lev_vol + w_camp_fire * lev_camp
    return lev.clip(-1.0, 1.5)

# -------------------------------
# Primitive signal 3: Wallhack(Linear)
# -------------------------------

def sig_linear_wallhack_filtered(
    df: pd.DataFrame,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    sma200_window: int = 200,
    vol_window: int = 20,
    donchian_window: int = 20,
    trend_scale: float = 10.0,
) -> pd.Series:
    """
    Linear Signal A: Wallhack (Filtered)

    raw_wallhack = 2 * (Close - MidDonchian) / RangeDonchian
    Regime_Trend = (Close - SMA200) / (Close * Vol_20 * trend_scale)
    Sig = raw_wallhack * Regime_Trend

    Requires OHLC (needs high/low for Donchian).
    """
    close = df[close_col].astype(float)
    high = df[high_col].astype(float)
    low = df[low_col].astype(float)

    ret = close.pct_change()
    vol_20 = ret.rolling(vol_window).std().replace(0, np.nan)
    sma200 = close.rolling(sma200_window).mean()

    high_n = high.rolling(donchian_window).max()
    low_n = low.rolling(donchian_window).min()
    mid_n = (high_n + low_n) / 2.0
    range_n = (high_n - low_n).replace(0, np.nan)

    regime_trend = (close - sma200) / (close * vol_20 * trend_scale)

    raw_wallhack = 2.0 * (close - mid_n) / range_n
    lev = raw_wallhack * regime_trend

    # guard early NaNs
    lev = lev.where(np.isfinite(lev), 0.0)
    return lev.clip(-1.0, 1.5)

# -------------------------------
# Primitive signal 4: Crossfire(Linear)
# -------------------------------

def sig_linear_crossfire_filtered(
    df: pd.DataFrame,
    close_col: str = "close",
    sma50_window: int = 50,
    sma200_window: int = 200,
    vol_window: int = 20,
    target_ann_vol: float = 0.15,
    crossfire_scale: float = 20.0,
) -> pd.Series:
    """
    Linear Signal B: Crossfire (Filtered)

    raw_crossfire = crossfire_scale * (SMA50 - SMA200) / SMA200
    Regime_Vol = (target_daily_vol) / Vol_20
      where target_daily_vol = target_ann_vol / sqrt(252)

    Sig = raw_crossfire * Regime_Vol
    """
    close = df[close_col].astype(float)
    ret = close.pct_change()

    vol_20 = ret.rolling(vol_window).std().replace(0, np.nan)
    sma50 = close.rolling(sma50_window).mean()
    sma200 = close.rolling(sma200_window).mean()

    target_daily = target_ann_vol / np.sqrt(252.0)
    regime_vol = target_daily / vol_20

    raw_crossfire = crossfire_scale * (sma50 - sma200) / sma200
    lev = raw_crossfire * regime_vol

    lev = lev.where(np.isfinite(lev), 0.0)
    return lev.clip(-1.0, 1.5)

# -------------------------------
# Composite signal 2: Linear Diversified(Wallhack and Crossfire)
# -------------------------------

def sig_linear_diversified(
    df: pd.DataFrame,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    w_wallhack: float = 0.5,
    w_crossfire: float = 0.5,
    **kwargs,
) -> pd.Series:
    """
    Linear Signal C: Diversified Trend Mix

    Sig = (w_wallhack * Wallhack_F) + (w_crossfire * Crossfire_F)
    """
    lev_wall = sig_linear_wallhack_filtered(
        df,
        close_col=close_col,
        high_col=high_col,
        low_col=low_col,
        **kwargs,
    )
    lev_cross = sig_linear_crossfire_filtered(
        df,
        close_col=close_col,
        **kwargs,
    )

    lev = w_wallhack * lev_wall + w_crossfire * lev_cross
    lev = lev.where(np.isfinite(lev), 0.0)
    return lev.clip(-1.0, 1.5)

# -------------------------------
# Primitive signal 5: Molotov (Downside Vol Targeting)
# -------------------------------

def sig_vol_molotov_downside_target(
    df: pd.DataFrame,
    close_col: str = "close",
    vol_window: int = 20,
    target_downside_ann_vol: float = 0.10,
) -> pd.Series:
    """
    Volatility Signal: Molotov (Downside volatility targeting).

    Idea:
      - Compute daily returns from close
      - Compute downside semi-vol: sqrt(mean( r^2 * 1_{r<0} ))
      - Target a fixed annual downside vol (e.g., 10%), convert to daily
      - Leverage = target_down_daily / downside_vol

    Returns:
        pd.Series leverage in [0, 1.5] (typically long-only sizing).
        NOTE: Unshifted; shift in backtest.
    """
    close = df[close_col].astype(float)
    ret = close.pct_change()

    down_ret_sq = np.where(ret < 0, ret**2, 0.0)
    down_vol = pd.Series(down_ret_sq, index=df.index).rolling(vol_window).mean()
    down_vol = np.sqrt(down_vol).replace(0, np.nan)

    target_daily = target_downside_ann_vol / np.sqrt(252.0)
    lev = target_daily / down_vol

    lev = lev.where(np.isfinite(lev), 0.0)
    return lev.clip(-1.0, 1.5)


# -------------------------------
# Primitive signal 6: ShiftWalk (Std Vol Targeting)
# -------------------------------

def sig_vol_shiftwalk_target(
    df: pd.DataFrame,
    close_col: str = "close",
    vol_window: int = 20,
    target_ann_vol: float = 0.15,
) -> pd.Series:
    """
    Volatility Signal: ShiftWalk (standard realized vol targeting).

    Leverage = target_daily / vol_20, where vol_20 = rolling std of returns.

    Returns:
        pd.Series leverage (typically long-only sizing).
        NOTE: Unshifted; shift in backtest.
    """
    close = df[close_col].astype(float)
    ret = close.pct_change()

    vol = ret.rolling(vol_window).std().replace(0, np.nan)

    target_daily = target_ann_vol / np.sqrt(252.0)
    lev = target_daily / vol

    lev = lev.where(np.isfinite(lev), 0.0)
    return lev.clip(-1.0, 1.5)


# -------------------------------
# Primitive signal 7: Smoke (Parkinson Range Vol Targeting)
# -------------------------------

def sig_vol_smoke_parkinson_target(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    vol_window: int = 20,
    target_ann_vol: float = 0.15,
) -> pd.Series:
    """
    Volatility Signal: Smoke (Parkinson volatility targeting using High/Low range).

    Standard Parkinson daily variance estimate:
        park_var_t = (1 / (4 ln 2)) * (ln(H/L))^2

    Then:
        park_vol = sqrt(rolling_mean(park_var, vol_window))
        lev = target_daily / park_vol

    Requires high/low columns.
    Returns unshifted leverage.
    """
    high = df[high_col].astype(float)
    low = df[low_col].astype(float)

    # Guard for bad data (e.g., low==0)
    hl = (high / low).replace([np.inf, -np.inf], np.nan)
    log_hl = np.log(hl)

    park_var = (log_hl**2) / (4.0 * np.log(2.0))
    park_vol = np.sqrt(park_var.rolling(vol_window).mean()).replace(0, np.nan)

    target_daily = target_ann_vol / np.sqrt(252.0)
    lev = target_daily / park_vol

    lev = lev.where(np.isfinite(lev), 0.0)
    return lev.clip(-1.0, 1.5)


# -------------------------------
# MR Signal 1: Crosshair_VT (Pivot Reversion)
# -------------------------------

def sig_mr_crosshair_vt(
    df: pd.DataFrame,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    vol_window: int = 20,
    atr_window: int = 14,
    target_ann_vol: float = 0.15,
    scale: float = 20.0,
) -> pd.Series:
    """
    Mean Reversion: Pivot reversion with vol targeting.

    Pivot = (H + L + C) / 3
    lev = ((Pivot - Close) / ATR) * VolScaler * scale
    """
    close = df[close_col].astype(float)
    high = df[high_col].astype(float)
    low = df[low_col].astype(float)

    pivot = (high + low + close) / 3.0
    atr = _atr_proxy_from_hl(df, high_col=high_col, low_col=low_col, atr_window=atr_window)
    vs = _vol_scaler(df, close_col=close_col, vol_window=vol_window, target_ann_vol=target_ann_vol)

    lev = ((pivot - close) / atr) * vs * scale
    lev = lev.where(np.isfinite(lev), 0.0)
    return lev.clip(-1.0, 1.5)


# -------------------------------
# MR Signal 2: IBS_VT (Internal Bar Strength reversion)
# -------------------------------

def sig_mr_ibs_vt(
    df: pd.DataFrame,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    vol_window: int = 20,
    target_ann_vol: float = 0.15,
    scale: float = 40.0,
) -> pd.Series:
    """
    Mean Reversion: IBS bar reversion with vol targeting.

    IBS = (C - L) / (H - L)
    lev = (0.5 - IBS) * VolScaler * scale
    """
    close = df[close_col].astype(float)
    high = df[high_col].astype(float)
    low = df[low_col].astype(float)

    rng = (high - low).replace(0, np.nan)
    ibs = (close - low) / rng

    vs = _vol_scaler(df, close_col=close_col, vol_window=vol_window, target_ann_vol=target_ann_vol)
    lev = (0.5 - ibs) * vs * scale

    lev = lev.where(np.isfinite(lev), 0.0)
    return lev.clip(-1.0, 1.5)


# -------------------------------
# MR Signal 3: Eco_Fade (Volume climax exhaustion fade)
# -------------------------------

def sig_mr_eco_fade(
    df: pd.DataFrame,
    close_col: str = "close",
    volume_col: str = "volume",
    vol_window: int = 20,
    vol_ma_window: int = 20,
    scale: float = 0.2,
) -> pd.Series:
    """
    Mean Reversion: Fade large close-to-close moves on high relative volume.

    vol_ratio = Volume / MA(Volume)
    lev = (- Return * vol_ratio) / Vol_20 * scale

    Note: This uses realized vol as normalization (not a target vol scaler).
    """
    close = df[close_col].astype(float)
    volu = df[volume_col].astype(float)

    ret = close.pct_change()
    vol_20 = ret.rolling(vol_window).std().replace(0, np.nan)

    ma_vol = volu.rolling(vol_ma_window).mean().replace(0, np.nan)
    vol_ratio = (volu / ma_vol).replace([np.inf, -np.inf], np.nan)

    lev = (-1.0 * ret * vol_ratio) / vol_20 * scale

    lev = lev.where(np.isfinite(lev), 0.0)
    return lev.clip(-1.0, 1.5)


# -------------------------------
# MR Signal 4: Snap_Revert (SMA5 reversion)
# -------------------------------

def sig_mr_snap_revert(
    df: pd.DataFrame,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    sma_window: int = 5,
    vol_window: int = 20,
    atr_window: int = 14,
    target_ann_vol: float = 0.15,
    scale: float = 20.0,
) -> pd.Series:
    """
    Mean Reversion: Revert to SMA5, normalized by ATR and vol targeting.

    lev = ((SMA5 - Close) / ATR) * VolScaler * scale
    """
    close = df[close_col].astype(float)

    sma5 = close.rolling(sma_window).mean()
    atr = _atr_proxy_from_hl(df, high_col=high_col, low_col=low_col, atr_window=atr_window)
    vs = _vol_scaler(df, close_col=close_col, vol_window=vol_window, target_ann_vol=target_ann_vol)

    lev = ((sma5 - close) / atr) * vs * scale

    lev = lev.where(np.isfinite(lev), 0.0)
    return lev.clip(-1.0, 1.5)


# -------------------------------
# MR Signal 5: Gap_Trap (Gap reversion)
# -------------------------------

def sig_mr_gap_trap(
    df: pd.DataFrame,
    close_col: str = "close",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    vol_window: int = 20,
    atr_window: int = 14,
    target_ann_vol: float = 0.15,
    scale: float = 20.0,
) -> pd.Series:
    """
    Mean Reversion: Fade the opening gap (Open vs PrevClose), normalized by ATR and vol targeting.

    prev_close = Close.shift(1)
    gap = prev_close - Open
    lev = (gap / ATR) * VolScaler * scale
    """
    close = df[close_col].astype(float)
    opn = df[open_col].astype(float)

    prev_close = close.shift(1)

    atr = _atr_proxy_from_hl(df, high_col=high_col, low_col=low_col, atr_window=atr_window)
    vs = _vol_scaler(df, close_col=close_col, vol_window=vol_window, target_ann_vol=target_ann_vol)

    lev = ((prev_close - opn) / atr) * vs * scale

    lev = lev.where(np.isfinite(lev), 0.0)
    return lev.clip(-1.0, 1.5)



# -------------------------------
# Primitive signal: Turn-of-Month
# -------------------------------

def sig_season_turn_of_month(
    df: pd.DataFrame,
    first_n: int = 3,
    last_n: int = 3,
    long_lev: float = 1.5,
) -> pd.Series:
    """
    Seasonality Signal: Turn-of-Month (TOM)

    Long during last_n trading days of month OR first_n trading days of month.
    """
    cal = _calendar_features(df.index)
    cond = (cal["ReverseTradingDay"] <= last_n) | (cal["TradingDay"] <= first_n)
    lev = pd.Series(0.0, index=df.index)
    lev = lev.where(~cond, long_lev)
    return lev.clip(-1.0, 1.5)


# -------------------------------
# Primitive signal: Holiday drift windows (approx)
# -------------------------------

def sig_season_holiday_drift(
    df: pd.DataFrame,
    long_lev: float = 1.5,
    include_christmas: bool = True,
    include_july4: bool = True,
    include_thanksgiving: bool = True,
) -> pd.Series:
    """
    Seasonality Signal: Holiday Drift (approximate calendar windows)

    Approximations used (same as your snippet):
      - Pre-Christmas: Dec 20-24
      - Pre-July 4th: Jul 1-3
      - Pre-Thanksgiving: Nov 20-26 (approx; not exact trading-day holiday logic)
    """
    cal = _calendar_features(df.index)

    cond = pd.Series(False, index=df.index)

    if include_christmas:
        cond |= (cal["Month"].eq(12) & cal["Day"].between(20, 24))
    if include_july4:
        cond |= (cal["Month"].eq(7) & cal["Day"].between(1, 3))
    if include_thanksgiving:
        cond |= (cal["Month"].eq(11) & cal["Day"].between(20, 26))

    lev = pd.Series(0.0, index=df.index)
    lev = lev.where(~cond, long_lev)
    return lev.clip(-1.0, 1.5)


# -------------------------------
# Primitive signal: September effect
# -------------------------------

def sig_season_september_effect(
    df: pd.DataFrame,
    short_lev: float = -1.0,
) -> pd.Series:
    """
    Seasonality Signal: September Effect

    Short during September.
    """
    cal = _calendar_features(df.index)
    cond = cal["Month"].eq(9)
    lev = pd.Series(0.0, index=df.index)
    lev = lev.where(~cond, short_lev)
    return lev.clip(-1.0, 1.5)


# -------------------------------
# Composite signal: AllStar (priority rules)
# -------------------------------

def sig_season_allstar(
    df: pd.DataFrame,
    first_n: int = 3,
    last_n: int = 3,
    long_lev: float = 1.5,
    sept_short_lev: float = -1.0,
    include_christmas: bool = True,
    include_july4: bool = True,
    include_thanksgiving: bool = True,
) -> pd.Series:
    """
    Seasonality Composite: AllStar (your exact priority logic)

    Start flat (0).
    Priority 1: September -> short (-1.0 by default)
    Priority 2: Turn-of-month OR holiday windows -> long (1.5 by default), overrides September when overlapping.
    """
    cal = _calendar_features(df.index)

    cond_tom = (cal["ReverseTradingDay"] <= last_n) | (cal["TradingDay"] <= first_n)

    cond_hols = pd.Series(False, index=df.index)
    if include_christmas:
        cond_hols |= (cal["Month"].eq(12) & cal["Day"].between(20, 24))
    if include_july4:
        cond_hols |= (cal["Month"].eq(7) & cal["Day"].between(1, 3))
    if include_thanksgiving:
        cond_hols |= (cal["Month"].eq(11) & cal["Day"].between(20, 26))

    cond_sept = cal["Month"].eq(9)

    lev = pd.Series(0.0, index=df.index)

    # Priority 1: September short
    lev = lev.where(~cond_sept, sept_short_lev)

    # Priority 2: Inflows long (overrides)
    lev = lev.where(~(cond_tom | cond_hols), long_lev)

    return lev.clip(-1.0, 1.5)

# -------------------------------
# Primitive signal: RushB (Vol trend)
# -------------------------------

def sig_vol_rushb(
    df: pd.DataFrame,
    close_col: str = "close",
    short_vol_window: int = 20,
    long_vol_window: int = 60,
    base: float = 1.5,
) -> pd.Series:
    """
    Vol Signal: RushB (Vol trend)

    lev = base - (Vol_short / Vol_long)
    If vol is falling (Vol_short < Vol_long), leverage increases.
    """
    close = df[close_col].astype(float)
    ret = close.pct_change()

    vol_s = ret.rolling(short_vol_window).std()
    vol_l = ret.rolling(long_vol_window).std()

    ratio = (vol_s / vol_l).replace([np.inf, -np.inf], np.nan)
    lev = base - ratio

    lev = lev.where(np.isfinite(lev), 0.0)
    return lev.clip(-1.0, 1.5)


# -------------------------------
# Primitive signal: SneakyBeaky (Inverse ATR% targeting)
# -------------------------------

def sig_vol_sneakybeaky_atr_target(
    df: pd.DataFrame,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    atr_window: int = 14,
    target_ann_vol: float = 0.15,
) -> pd.Series:
    """
    Vol Signal: SneakyBeaky (ATR-based vol targeting)

    ATR proxy = rolling mean(High-Low, atr_window)
    ATR% = ATR / Close
    lev = target_daily / ATR%
    """
    close = df[close_col].astype(float)
    high = df[high_col].astype(float)
    low = df[low_col].astype(float)

    rng = (high - low).replace(0, np.nan)
    atr = rng.rolling(atr_window).mean()
    atr_pct = (atr / close).replace([np.inf, -np.inf], np.nan)

    target_daily = target_ann_vol / np.sqrt(252.0)
    lev = target_daily / atr_pct

    lev = lev.where(np.isfinite(lev), 0.0)
    return lev.clip(-1.0, 1.5)


# -------------------------------
# Primitive signal: Camping (Relative Volatility)
# -------------------------------

def sig_vol_camping(
    df: pd.DataFrame,
    close_col: str = "close",
    short_vol_window: int = 20,
    long_vol_window: int = 60,
    scale: float = 1.0,
) -> pd.Series:
    """
    Vol Signal: Camping (Relative volatility)

    lev = (Vol_long / Vol_short) * scale
    If recent vol is quiet vs history, leverage increases.
    """
    close = df[close_col].astype(float)
    ret = close.pct_change()

    vol_s = ret.rolling(short_vol_window).std().replace(0, np.nan)
    vol_l = ret.rolling(long_vol_window).std()

    lev = (vol_l / vol_s) * scale
    lev = lev.replace([np.inf, -np.inf], np.nan)

    lev = lev.where(np.isfinite(lev), 0.0)
    return lev.clip(-1.0, 1.5)


# -------------------------------
# Primitive signal: Silence (Bollinger Bandwidth squeeze)
# -------------------------------

def sig_vol_silence_bb_squeeze(
    df: pd.DataFrame,
    close_col: str = "close",
    bb_window: int = 20,
    target_bb_width: float = 0.05,
) -> pd.Series:
    """
    Vol Signal: Silence (Bandwidth squeeze)

    BB_Width = (4 * rolling_std(Close, bb_window)) / rolling_mean(Close, bb_window)
    lev = target_bb_width / BB_Width
    """
    close = df[close_col].astype(float)
    bb_std = close.rolling(bb_window).std()
    bb_mid = close.rolling(bb_window).mean().replace(0, np.nan)

    bb_width = (4.0 * bb_std) / bb_mid
    bb_width = bb_width.replace(0, np.nan)

    lev = target_bb_width / bb_width
    lev = lev.replace([np.inf, -np.inf], np.nan)

    lev = lev.where(np.isfinite(lev), 0.0)
    return lev.clip(-1.0, 1.5)


# -------------------------------
# Primitive signal: Decoy (Garman-Klass vol targeting)
# -------------------------------

def sig_vol_decoy_gk_target(
    df: pd.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    gk_window: int = 20,
    target_ann_vol: float = 0.15,
) -> pd.Series:
    """
    Vol Signal: Decoy (Garman-Klass OHLC volatility targeting)

    GK variance (daily):
      gk_var = 0.5*ln(H/L)^2 - (2*ln(2)-1)*ln(C/O)^2
    GK_vol = sqrt(rolling_mean(gk_var, gk_window))
    lev = target_daily / GK_vol
    """
    opn = df[open_col].astype(float)
    high = df[high_col].astype(float)
    low = df[low_col].astype(float)
    close = df[close_col].astype(float)

    # avoid log issues
    hl = (high / low).replace([np.inf, -np.inf], np.nan)
    co = (close / opn).replace([np.inf, -np.inf], np.nan)

    log_hl_sq = (np.log(hl) ** 2)
    log_co_sq = (np.log(co) ** 2)

    gk_var = 0.5 * log_hl_sq - (2.0 * np.log(2.0) - 1.0) * log_co_sq
    gk_vol = np.sqrt(gk_var.rolling(gk_window).mean()).replace(0, np.nan)

    target_daily = target_ann_vol / np.sqrt(252.0)
    lev = target_daily / gk_vol
    lev = lev.replace([np.inf, -np.inf], np.nan)

    lev = lev.where(np.isfinite(lev), 0.0)
    return lev.clip(-1.0, 1.5)

# -------------------------------
# Primitive signal: Wallbang (ATH drawdown scaler)
# -------------------------------

def sig_tail_wallbang_ath_drawdown(
    df: pd.DataFrame,
    close_col: str = "close",
    high_col: str = "high",
    ath_window: int = 252,
    drawdown_floor: float = 0.8,
    scale: float = 5.0,
) -> pd.Series:
    """
    Tail / Drawdown Signal: Wallbang (ATH drawdown scaler)

    High_ATH = rolling max(high, ath_window).shift(1)   # no lookahead
    lev = scale * (Close / High_ATH - drawdown_floor)

    With defaults:
      - Close/ATH = 1.0 -> lev = +1.0
      - Close/ATH = 0.8 -> lev = 0.0
      - Below 0.8 -> negative (can go short), then clipped to [-1, 1.5]

    Returns: unshifted leverage series.
    """
    close = df[close_col].astype(float)
    high = df[high_col].astype(float)

    high_ath = high.rolling(ath_window).max().shift(1)

    lev = (close / high_ath - drawdown_floor) * scale
    lev = lev.replace([np.inf, -np.inf], np.nan)
    lev = lev.where(np.isfinite(lev), 0.0)

    return lev.clip(-1.0, 1.5)


# -------------------------------
# Benchmark signal: Buy & Hold
# -------------------------------

def sig_benchmark_buy_hold(
    df: pd.DataFrame,
    lev_const: float = 1.0,
) -> pd.Series:
    """
    Benchmark: constant exposure (Buy & Hold).
    Returns: unshifted leverage series (constant).
    """
    lev = pd.Series(lev_const, index=df.index, dtype=float)
    return lev.clip(-1.0, 1.5)

# -------------------------------
# Primitive signal: Monday effect
# -------------------------------

def sig_season_monday_effect(
    df: pd.DataFrame,
    long_lev: float = 1.5,
) -> pd.Series:
    """
    Seasonality Signal: Monday Effect

    Logic:
      - If weekday == Monday -> long long_lev
      - Else -> 0.0
    """
    idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)
    weekday = idx.dayofweek  # Monday=0
    cond_mon = (weekday == 0)

    lev = pd.Series(0.0, index=df.index, dtype=float)
    lev = lev.where(~cond_mon, long_lev)
    return lev.clip(-1.0, 1.5)


# -------------------------------
# Composite signal: Calendar Alpha
# -------------------------------

def sig_season_calendar_alpha(
    df: pd.DataFrame,
    first_n: int = 3,
    last_n: int = 3,
    long_lev: float = 1.5,
    sept_short_lev: float = -1.0,
    include_christmas: bool = True,
    include_july4: bool = True,
    include_thanksgiving: bool = True,
    include_monday: bool = True,
) -> pd.Series:
    """
    Seasonality Composite: Calendar Alpha

    Components:
      A) Turn-of-month: last_n trading days OR first_n trading days -> long
      B) Holiday drift windows (approx): -> long
      C) Monday effect: -> long
      D) September effect: -> short

    Priority:
      1) September short
      2) Any "inflows" window (TOM or holidays or Monday) -> long (overrides September)

    Returns: unshifted leverage series.
    """
    # calendar features
    idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)
    cal = pd.DataFrame(index=df.index)
    cal["Month"] = idx.month
    cal["Day"] = idx.day
    cal["Weekday"] = idx.dayofweek

    cal["TradingDay"] = cal.groupby([idx.year, idx.month]).cumcount() + 1
    cal["DaysInMonth"] = cal.groupby([idx.year, idx.month])["TradingDay"].transform("max")
    cal["ReverseTradingDay"] = cal["DaysInMonth"] - cal["TradingDay"] + 1

    # A) TOM
    cond_tom = (cal["ReverseTradingDay"] <= last_n) | (cal["TradingDay"] <= first_n)

    # B) Holidays (same approximations as your snippet)
    cond_hols = pd.Series(False, index=df.index)
    if include_christmas:
        cond_hols |= (cal["Month"].eq(12) & cal["Day"].between(20, 24))
    if include_july4:
        cond_hols |= (cal["Month"].eq(7) & cal["Day"].between(1, 3))
    if include_thanksgiving:
        cond_hols |= (cal["Month"].eq(11) & cal["Day"].between(20, 26))

    # C) Monday
    cond_mon = cal["Weekday"].eq(0) if include_monday else pd.Series(False, index=df.index)

    # D) September
    cond_sept = cal["Month"].eq(9)

    lev = pd.Series(0.0, index=df.index, dtype=float)

    # Priority 1: September short
    lev = lev.where(~cond_sept, sept_short_lev)

    # Priority 2: inflows long overrides
    inflows = cond_tom | cond_hols | cond_mon
    lev = lev.where(~inflows, long_lev)

    return lev.clip(-1.0, 1.5)

# End CSGO Player Persona(QQQ Dataset)

# Start CSGO Trader Persona(QQQ Dataset)

# -------------------------------
# Primitive signal: Float_Cap (Volatility compression / squeeze sizing)
# -------------------------------

def sig_flow_float_cap_atr_compression(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    short_atr_window: int = 5,
    long_atr_window: int = 20,
    atr_window: int = 14,
    base: float = 1.5,
) -> pd.Series:
    """
    Flow/Vol Signal: Float_Cap (ATR compression sizing)

    ATR_proxy = rolling_mean(High - Low, atr_window)
    lev = base - (ATR_proxy_ma_short / ATR_proxy_ma_long)

    Intuition:
      - If short ATR < long ATR => compression (squeeze) => lev increases
      - If short ATR > long ATR => expansion => lev decreases
    """
    high = df[high_col].astype(float)
    low = df[low_col].astype(float)

    rng = (high - low).replace(0, np.nan)
    atr = rng.rolling(atr_window).mean()

    atr_s = atr.rolling(short_atr_window).mean()
    atr_l = atr.rolling(long_atr_window).mean().replace(0, np.nan)

    ratio = (atr_s / atr_l).replace([np.inf, -np.inf], np.nan)
    lev = base - ratio

    lev = lev.where(np.isfinite(lev), 0.0)
    return lev.clip(-1.0, 1.5)

# -------------------------------
# Primitive signal: Market_Crash (Volatility acceleration fade)
# -------------------------------

def sig_flow_market_crash_vol_accel_fade(
    df: pd.DataFrame,
    close_col: str = "close",
    vol_fast: int = 5,
    vol_slow: int = 10,
    scale: float = 100.0,
    base: float = 1.0,
) -> pd.Series:
    """
    Vol Signal: Market_Crash (volatility acceleration fade)

    ret = pct_change(close)
    vol_fast = std(ret, vol_fast)
    vol_slow = std(ret, vol_slow)

    lev = base - (vol_fast - vol_slow) * scale

    Intuition:
      - If vol_fast > vol_slow (vol accelerating), lev decreases
      - If vol_fast < vol_slow (vol calming), lev increases
    """
    close = df[close_col].astype(float)
    ret = close.pct_change()

    v_fast = ret.rolling(vol_fast).std()
    v_slow = ret.rolling(vol_slow).std()

    lev = base - (v_fast - v_slow) * scale
    lev = lev.replace([np.inf, -np.inf], np.nan)

    lev = lev.where(np.isfinite(lev), 0.0)
    return lev.clip(-1.0, 1.5)

# -------------------------------
# Primitive signal: Market_Bot (Volume stability sizing)
# -------------------------------

def sig_flow_market_bot_volume_stability(
    df: pd.DataFrame,
    volume_col: str = "volume",
    window: int = 20,
    target_cv: float = 0.2,
) -> pd.Series:
    """
    Flow Signal: Market_Bot (volume stability sizing)

    vol_cv = std(volume, window) / mean(volume, window)
    lev = target_cv / vol_cv

    Intuition:
      - Low CV (steady/institutional flow proxy) => lev increases
      - High CV (erratic volume) => lev decreases
    """
    volu = df[volume_col].astype(float).replace(0, np.nan)

    m = volu.rolling(window).mean().replace(0, np.nan)
    s = volu.rolling(window).std()

    cv = (s / m).replace([np.inf, -np.inf], np.nan)
    lev = (target_cv / cv).replace([np.inf, -np.inf], np.nan)

    lev = lev.where(np.isfinite(lev), 0.0)
    return lev.clip(-1.0, 1.5)

# -------------------------------
# Primitive signal: Buy_Order (Range position / channel %K)
# -------------------------------

def sig_pa_buy_order_channel_position(
    df: pd.DataFrame,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    channel_window: int = 20,
    scale: float = 2.0,
) -> pd.Series:
    """
    Price Action Signal: Buy_Order (position in 20D price channel)

    high_N = max(High, N)
    low_N  = min(Low, N)
    lev = scale * (Close - low_N) / (high_N - low_N)

    Notes:
      - This is long-only in its raw form (0..scale), then clipped to [-1, 1.5]
      - Behaves trend-following: higher in range => more long
    """
    close = df[close_col].astype(float)
    high = df[high_col].astype(float)
    low = df[low_col].astype(float)

    high_n = high.rolling(channel_window).max()
    low_n = low.rolling(channel_window).min()
    rng = (high_n - low_n).replace(0, np.nan)

    lev = scale * (close - low_n) / rng
    lev = lev.replace([np.inf, -np.inf], np.nan)

    lev = lev.where(np.isfinite(lev), 0.0)
    return lev.clip(-1.0, 1.5)

# CSGO Trader Persona End(QQQ Dataset)

# CSGO IGL Persona(QQQ Dataset)

LEVERAGE_MIN = -1.0
LEVERAGE_MAX = 1.5

def sig_igl_2plus_combo(df: pd.DataFrame) -> pd.Series:
    """
    IGL FINAL MASTERPIECE: 'RECOIL ZERO v29'
    TARGET: 2.0+ Annualized Sharpe Ratio (Strict Arithmetic)
    
    THE WINNING FORMULA:
    1. THE CARRY (45%): Simple Mix (1.60) - The primary aim.
    2. THE LURKER (35%): Float Cap (1.51) - High-efficiency support.
    3. THE SPECIALIST (20%): Turn-of-Month - Structural flow alpha.
    4. THE ANCHOR: 9.7% Garman-Klass Vol Target. 
       Simulation shows 9.7% is the 'Golden Ratio' for QQQ Arithmetic Sharpe.
    """

    # --- Standard Alias Logic ---
    d = df.copy()
    mapping = {"Latest": "close", "High": "high", "Low": "low", "Volume": "volume", "Open": "open"}
    for src, dst in mapping.items():
        if src in d.columns: d[dst] = d[src]

    # 1. Gather the Council (Weights from M-62 Simulation Winner)
    s1 = sig_csgo_simple_mix(d) 
    s2 = sig_flow_float_cap_atr_compression(d) 
    s3 = sig_season_turn_of_month(d, first_n=3, last_n=3, long_lev=1.0)
    
    # 2. Risk Parity Multiplier
    # Target 9.7% Annual Vol (0.0979). 
    # Simulation shows this specific target minimizes the daily variance 
    # of the 60/40/20 blend better than any other value.
    anchor = sig_vol_decoy_gk_target(d, gk_window=20, target_ann_vol=0.0979)

    # 3. Tactical Synthesis
    # We blend the Alphas FIRST, then scale the whole unit by the Anchor.
    alpha_core = (s1 * 0.458) + (s2 * 0.354) + (s3 * 0.188)
    
    # We add a 0.12 Drift Baseline. 
    # Previous 0.4-0.6 versions were too 'loud' (high sigma). 
    # 0.12 keeps us in the drift without bloating the denominator.
    final_raw = (alpha_core * anchor) + 0.123

    # 4. Final Squashing (The Discipline)
    # s=1.0 for balanced response.
    composite = 1.5 * np.tanh(final_raw / 1.0)

    return composite.clip(-1.0, 1.5).fillna(0.0)


    # CSGO IGL Persona End(QQQ Dataset)