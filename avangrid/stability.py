"""Pattern comparison between windows (notebook 50) and change-point scans (notebook 60)."""

import numpy as np


def row_corr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson correlation between matching rows of A and B, ignoring missing values."""
    A = A - np.nanmean(A, axis=1, keepdims=True)
    B = B - np.nanmean(B, axis=1, keepdims=True)
    return np.nansum(A * B, axis=1) / np.sqrt(np.nansum(A * A, axis=1) * np.nansum(B * B, axis=1))


def window_r(E: np.ndarray, L: int, anchors, kind: str = 'adjacent', day_mask: np.ndarray | None = None) -> np.ndarray:
    """Per-customer R between 24-hour patterns of two L-day windows, averaged over start days.

    E: customers x days x 24 (e.g. peer-relative residuals).
    kind: 'adjacent' (days [a, a+L) vs [a+L, a+2L)) or 'interleaved' (odd vs even days of [a, a+2L)).
    day_mask: optional boolean per day; only those days are used (e.g. weekdays only).
    """
    rs = []
    for a in anchors:
        block = E[:, a:a + 2 * L]
        keep = np.ones(2 * L, bool) if day_mask is None else day_mask[a:a + 2 * L]
        if kind == 'adjacent':
            m1, m2 = keep.copy(), keep.copy()
            m1[L:], m2[:L] = False, False
        elif kind == 'interleaved':
            m1, m2 = keep.copy(), keep.copy()
            m1[1::2], m2[0::2] = False, False
        else:
            raise ValueError(kind)
        if m1.any() and m2.any():
            rs.append(row_corr(np.nanmean(block[:, m1], 1), np.nanmean(block[:, m2], 1)))
    return np.nanmean(np.array(rs), axis=0)


def change_scan(W: np.ndarray, min_weeks: int) -> tuple[np.ndarray, np.ndarray]:
    """Step-change statistic for every split of a weekly multi-hour pattern.

    W: customers x weeks x 24. For each split t (the "after" period starts at week t, with at
    least ``min_weeks`` on each side), computes a diagonal Hotelling statistic: the squared
    difference of before/after means per hour, scaled by the pooled within-period variance and
    summed over hours. Returns (split weeks, statistic of shape customers x splits).
    """
    c, nw, h = W.shape
    W = np.where(np.isnan(W), np.nanmean(W, 1, keepdims=True), W).astype(np.float64)
    S1 = np.concatenate([np.zeros((c, 1, h)), np.cumsum(W, 1)], 1)
    S2 = np.concatenate([np.zeros((c, 1, h)), np.cumsum(W * W, 1)], 1)
    ts = np.arange(min_weeks, nw - min_weeks + 1)
    n1 = ts[None, :, None]
    n2 = nw - n1
    s1, q1 = S1[:, ts], S2[:, ts]
    s2, q2 = S1[:, [nw]] - s1, S2[:, [nw]] - q1
    m1, m2 = s1 / n1, s2 / n2
    var = np.maximum(((q1 - n1 * m1 ** 2) + (q2 - n2 * m2 ** 2)) / (nw - 2), 1e-4)
    T = (n1[:, :, 0] * n2[:, :, 0] / nw) * ((m1 - m2) ** 2 / var).sum(2)
    return ts, T


def step_scan(x: np.ndarray, weeks: np.ndarray, min_weeks: int) -> tuple[np.ndarray, np.ndarray]:
    """One-signal step scan (customers x weeks) restricted to ``weeks``.

    Returns (split week index, z) where z is the two-sample t-like statistic (after minus before)
    for every split, so positive z means the signal went up.
    """
    x = x[:, weeks]
    x = np.where(np.isnan(x), np.nanmean(x, 1, keepdims=True), x)
    n = x.shape[1]
    S1 = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(x, 1)], 1)
    S2 = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(x * x, 1)], 1)
    ts = np.arange(min_weeks, n - min_weeks + 1)
    n1 = ts[None, :]
    n2 = n - n1
    m1 = S1[:, ts] / n1
    m2 = (S1[:, [n]] - S1[:, ts]) / n2
    ss = (S2[:, ts] - n1 * m1 ** 2) + ((S2[:, [n]] - S2[:, ts]) - n2 * m2 ** 2)
    sp = np.sqrt(np.maximum(ss / (n - 2), 1e-8))
    z = (m2 - m1) / (sp * np.sqrt(1 / n1 + 1 / n2))
    return weeks[ts], z


EVENING_NIGHT = np.r_[0:6, 18:24]   # hour_ending 1-6 and 19-24, when most home charging happens
HP_SCAN_MAX_TEMP_F = 45             # HP changes are only scanned in weeks colder than this
FLAG_QUANTILE = 0.95                # flag z above this quantile of customers on no list (~5% false flags)


def ev_weekly_signal(kwh: np.ndarray, temp: np.ndarray, n_weeks: int) -> np.ndarray:
    """Weekly evening/overnight kWh above the customer's usual use, adjusted for weekly weather.

    Counts only hours at least 2.5 kWh above the customer's median for that hour. In simulation this
    caught ~35% of injected EV adopters at a 5% false-flag rate, against ~6% for daily session counts
    (which also rise with electric heating in winter).
    """
    from .features import weekly
    K = np.asarray(kwh)
    excess = K - np.nanmedian(K, axis=1)[:, None, :]
    evening_night = excess[:, :, EVENING_NIGHT]
    daily = np.nansum(np.where(evening_night >= 2.5, evening_night, 0), axis=2)
    Y = weekly(daily, n_weeks)
    tw = weekly(np.asarray(temp)[None, :], n_weeks)[0]
    X = np.column_stack([np.ones(n_weeks), np.maximum(55 - tw, 0), np.maximum(tw - 65, 0)])
    beta = np.linalg.pinv(X) @ np.nan_to_num(Y).T
    return Y - (X @ beta).T


def tech_weekly_signals(D: dict, kwh: np.ndarray, temp: np.ndarray, n_weeks: int) -> dict[str, np.ndarray]:
    """One weekly series per technology, oriented so that adoption makes it go up (as in notebook 60)."""
    from .features import weekly
    return {'EV': ev_weekly_signal(kwh, temp, n_weeks),
            'PV': -weekly(np.asarray(D['e_midday']) - np.asarray(D['e_night']), n_weeks),  # deeper midday dip -> higher
            'HP': weekly(np.asarray(D['e_level']), n_weeks)}


def tech_scan_weeks(temp: np.ndarray, n_weeks: int) -> dict[str, np.ndarray]:
    """Weeks each technology's step scan covers: all weeks, except HP (cold weeks only)."""
    from .features import weekly
    tw = weekly(np.asarray(temp)[None, :], n_weeks)[0]
    return {'EV': np.arange(n_weeks), 'PV': np.arange(n_weeks), 'HP': np.flatnonzero(tw < HP_SCAN_MAX_TEMP_F)}


def tech_change_points(D: dict, kwh: np.ndarray, temp: np.ndarray, min_weeks: int,
                       not_listed: np.ndarray, thresholds: dict | None = None) -> dict[str, dict]:
    """Technology-specific change scan for every customer (notebook 60, section 3b).

    Returns {tech: {'z', 'week', 'threshold', 'flag'}}. Thresholds default to the FLAG_QUANTILE of
    customers on no list; pass ``thresholds`` to score new rows (e.g. simulated customers) on the same scale.
    """
    n_weeks = np.asarray(kwh).shape[1] // 7
    signals = tech_weekly_signals(D, kwh, temp, n_weeks)
    weeks = tech_scan_weeks(temp, n_weeks)
    out = {}
    for tech, x in signals.items():
        split_week, z = step_scan(x, weeks[tech], min_weeks)
        zmax, best = z.max(axis=1), split_week[z.argmax(axis=1)]
        thr = thresholds[tech] if thresholds else np.quantile(zmax[not_listed], FLAG_QUANTILE)
        out[tech] = {'z': zmax, 'week': best, 'threshold': thr, 'flag': zmax >= thr}
    return out
