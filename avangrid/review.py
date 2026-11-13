"""Visual review sheets for hand-labelling homes.

Each home gets two panels:
  - a day x hour heat map of hourly kWh for the year (x: Sep -> Aug, y: hour of day, midnight at top),
    on a fixed 0-5 kWh colour scale so large loads (e.g. 240 V EV charging at 3-7 kWh) stand out;
  - daily kWh against daily average temperature, which shows heating (left) and cooling (right) response.

Homes are labelled only with an anonymous code, so the reviewer doesn't know whether a home is a model
candidate, on a list, or a random draw (blind review).
"""

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .grids import HourlyData  # noqa: E402

MONTH_TICKS = [(0, 'Sep'), (30, 'Oct'), (61, 'Nov'), (91, 'Dec'), (122, 'Jan'), (153, 'Feb'), (181, 'Mar'),
               (212, 'Apr'), (242, 'May'), (273, 'Jun'), (303, 'Jul'), (334, 'Aug')]


def review_sheet(hd: HourlyData, rows: np.ndarray, codes: list[str], path, vmax: float = 5.0) -> None:
    """Save one sheet with a heat map + temperature scatter for each home in ``rows`` (array row indices)."""
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(14, 2.3 * n), gridspec_kw={'width_ratios': [4, 1]})
    axes = np.atleast_2d(axes)
    for (ax_h, ax_s), r, code in zip(axes, rows, codes):
        k = np.asarray(hd.kwh[r])                     # days x 24
        ax_h.imshow(k.T, aspect='auto', cmap='magma', vmin=0, vmax=vmax, interpolation='nearest')
        ax_h.set_yticks([0, 6, 12, 18, 23], ['1', '7', '13', '19', '24'], fontsize=7)
        ax_h.set_xticks([t for t, _ in MONTH_TICKS], [m for _, m in MONTH_TICKS], fontsize=7)
        ax_h.set_ylabel('hour', fontsize=7)
        ax_h.set_title(f'Home {code}', fontsize=9, loc='left')
        daily = np.nanmean(k, 1) * 24
        ax_s.scatter(hd.temp, daily, s=4, color='#0072B2', alpha=0.6)
        ax_s.set_xlabel('daily temp (F)', fontsize=7)
        ax_s.set_ylabel('kWh/day', fontsize=7)
        ax_s.tick_params(labelsize=7)
        ax_s.set_ylim(bottom=0)
        ax_s.grid(color='#e5e5e5', linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
