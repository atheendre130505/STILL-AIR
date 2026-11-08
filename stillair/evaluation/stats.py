"""Confidence intervals and number formatting."""

from __future__ import annotations

import math
import statistics

_T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23, 14: 2.14, 19: 2.09, 29: 2.05}


def mean_ci(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return (None, None)
    m = statistics.fmean(xs)
    if len(xs) < 2:
        return (m, 0.0)
    df = len(xs) - 1
    t = _T95.get(df) or _T95[max(k for k in _T95 if k <= df)] if df < 30 else 1.96
    return (m, t * statistics.stdev(xs) / math.sqrt(len(xs)))


def fmt(x, pct=False, digits=2):
    if x is None or x[0] is None:
        return "n/a"
    m, ci = x
    if pct:
        return f"{100*m:.0f}% ±{100*ci:.0f}"
    return f"{m:.{digits}f} ±{ci:.{digits}f}"
