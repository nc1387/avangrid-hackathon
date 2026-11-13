"""The EV, heat pump (HP) and solar PV customer lists from Avangrid's workbook.

From the workbook's Description tab: the EV and HP lists are random samples of about 25% of
the customers Avangrid believes have them; the PV list is described as the full list of
solar customers.
"""

import numpy as np
import polars as pl

from . import config


def load_customer_groups() -> pl.DataFrame:
    """Long table: one row per (CUSTOMER_NUMBER, customer_group), groups 'EV', 'HP', 'PV'.

    A customer on several lists appears once per list.
    """
    sheets = pl.read_excel(config.CUSTOMER_GROUPS_PATH, sheet_id=0)
    sheets.pop('Description')
    return pl.concat([
        customers.select(
            pl.col('CUSTOMER_NUMBER').cast(pl.Int32),
            pl.lit(tab.split()[0]).alias('customer_group'),
        )
        for tab, customers in sheets.items()
    ])


def description() -> list[str]:
    """The text of the workbook's Description tab."""
    d = pl.read_excel(config.CUSTOMER_GROUPS_PATH, sheet_name='Description')
    return [line.strip() for line in [d.columns[0], *d.to_series()]]


def label_arrays(customer_numbers, groups: pl.DataFrame | None = None) -> dict[str, np.ndarray]:
    """Boolean array per technology, aligned to ``customer_numbers``."""
    groups = load_customer_groups() if groups is None else groups
    customer_numbers = np.asarray(customer_numbers)
    return {
        tech: np.isin(customer_numbers, groups.filter(pl.col('customer_group') == tech)['CUSTOMER_NUMBER'].to_numpy())
        for tech in config.TECHS
    }
