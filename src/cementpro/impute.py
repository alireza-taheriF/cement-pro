"""Missing-data compensation.

Policy by gap length (all thresholds configurable):

* short gap   -> linear interpolation in time
* medium gap  -> cross-tag regression (least squares on correlated tags) plus a
                 linear correction of the model residual across the gap edges
* long gap    -> *not* filled; left NaN and reported as unrecoverable, because
                 invented data over long outages would silently bias later models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .quality import gap_runs


@dataclass(frozen=True)
class ImputeConfig:
    short_gap_min: float = 5.0
    medium_gap_min: float = 60.0
    max_predictors: int = 5
    min_fit_rows: int = 200


def _linear_fill(s: pd.Series, mask: pd.Series) -> pd.Series:
    filled = s.interpolate(method="time", limit_area="inside")
    return s.where(~mask, filled)


def _fit_predictors(
    target: pd.Series, others: pd.DataFrame, cfg: ImputeConfig
) -> tuple[np.ndarray, list[str], float, float] | None:
    """Fit a standardized least-squares model target ~ others on rows where all are observed."""
    joined = pd.concat([target.rename("_y"), others], axis=1).dropna()
    if len(joined) < cfg.min_fit_rows:
        return None
    corr = joined.corr()["_y"].drop("_y").abs().sort_values(ascending=False)
    cols = [c for c in corr.index[: cfg.max_predictors] if corr[c] > 0.1]
    if not cols:
        return None
    x = joined[cols]
    mu, sd = x.mean(), x.std().replace(0, 1.0)
    xs = np.column_stack([np.ones(len(x)), ((x - mu) / sd).to_numpy()])
    y = joined["_y"].to_numpy()
    beta, *_ = np.linalg.lstsq(xs, y, rcond=None)
    return beta, cols, mu, sd  # type: ignore[return-value]


def _predict(beta: np.ndarray, cols: list[str], mu: pd.Series, sd: pd.Series, others: pd.DataFrame) -> pd.Series:
    x = others[cols]
    xs = np.column_stack([np.ones(len(x)), ((x - mu) / sd).to_numpy()])
    pred = xs @ beta
    pred[x.isna().any(axis=1).to_numpy()] = np.nan
    return pd.Series(pred, index=x.index)


def impute_tag(
    native: pd.Series,
    period_s: int,
    predictors: pd.DataFrame | None,
    cfg: ImputeConfig = ImputeConfig(),
    method: str = "hybrid",
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Fill gaps of one native-period series.

    ``method``: ``hybrid`` (policy above), ``linear`` (linear up to the medium
    threshold) or ``ffill`` (hold last value up to the medium threshold).

    Returns ``(filled, imputed_mask, unrecoverable_mask)``.
    """
    period_min = period_s / 60.0
    out = native.copy()
    imputed = pd.Series(False, index=native.index)
    gaps = gap_runs(native.isna())
    if gaps.empty:
        return out, imputed, imputed.copy()

    model = None
    if method == "hybrid" and predictors is not None and not predictors.empty:
        pr = predictors.reindex(native.index)
        model = _fit_predictors(native, pr, cfg)
        pred = _predict(*model, pr) if model else None
    else:
        pred = None

    for g in gaps.itertuples():
        minutes = g.n_samples * period_min
        sl = native.index[(native.index >= g.start) & (native.index <= g.end)]
        if minutes > cfg.medium_gap_min:
            continue  # unrecoverable: left NaN
        before = native.loc[:g.start].iloc[:-1].dropna()
        after = native.loc[g.end:].iloc[1:].dropna()
        if method == "ffill":
            if before.empty:
                continue
            out.loc[sl] = before.iloc[-1]
        elif minutes <= cfg.short_gap_min or pred is None:
            if before.empty or after.empty:
                continue  # cannot interpolate at series edges
            seg = native.loc[before.index[-1]: after.index[0]].interpolate(method="time")
            out.loc[sl] = seg.loc[sl]
        else:
            base = pred.loc[sl]
            r0 = (before.iloc[-1] - pred.get(before.index[-1], np.nan)) if not before.empty else np.nan
            r1 = (after.iloc[0] - pred.get(after.index[0], np.nan)) if not after.empty else np.nan
            if np.isnan(r0) and np.isnan(r1):
                r0 = r1 = 0.0
            r0 = r1 if np.isnan(r0) else r0
            r1 = r0 if np.isnan(r1) else r1
            w = np.linspace(0, 1, len(sl) + 2)[1:-1]
            out.loc[sl] = base.to_numpy() + (1 - w) * r0 + w * r1
            if out.loc[sl].isna().any() and not before.empty and not after.empty:
                # predictors were themselves missing: fall back to linear for those samples
                seg = native.loc[before.index[-1]: after.index[0]].interpolate(method="time")
                out.loc[sl] = out.loc[sl].fillna(seg.loc[sl])
        imputed.loc[sl] = out.loc[sl].notna().to_numpy()

    return out, imputed, native.isna() & out.isna()
