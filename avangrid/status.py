"""The ``customer_status`` table: one row per customer number, saying which step (if any) excluded them.

Built from tables made by notebooks 10-50, so rebuild it after re-running any of those:

    python -m avangrid status-build

Customers are excluded at these stages, in order:

| status                     | meaning                                                                          |
|----------------------------|----------------------------------------------------------------------------------|
| no meter data              | absent from the meter data entirely                                              |
| dropped: incomplete        | in ``excluded_customers`` (joined late, left early, or a gap longer than 3 days) |
| excluded: low usage        | in ``ami_analysis`` but averages under 0.05 kWh/hour, so left out of peer groups |
| included                   | in ``customer_peer_groups``: used by notebooks 50 and 60 and the cached arrays   |

``in_ami_analysis`` is True for both "excluded: low usage" and "included": notebooks 20-40 use
every ``ami_analysis`` customer.
"""

import polars as pl

from . import db
from .labels import load_customer_groups

TOTAL_CUSTOMERS = 35_725  # customer numbers 1-35,725, per the data dictionary

STATUS_SQL = f'''
CREATE OR REPLACE TABLE customer_status AS
WITH all_customers AS (
    SELECT range::INTEGER AS CUSTOMER_NUMBER FROM range(1, {TOTAL_CUSTOMERS + 1})
),
in_ami AS (SELECT DISTINCT CUSTOMER_NUMBER FROM ami),
in_analysis AS (SELECT DISTINCT CUSTOMER_NUMBER FROM ami_analysis),
imputed AS (SELECT CUSTOMER_NUMBER, COUNT(*) AS imputed_hours FROM ami_imputed_rows GROUP BY 1),
lists AS (
    SELECT CUSTOMER_NUMBER,
           BOOL_OR(customer_group = 'EV') AS on_ev_list,
           BOOL_OR(customer_group = 'HP') AS on_hp_list,
           BOOL_OR(customer_group = 'PV') AS on_pv_list
    FROM customer_groups_df GROUP BY 1
)
SELECT
    c.CUSTOMER_NUMBER,
    CASE
        WHEN a.CUSTOMER_NUMBER IS NULL THEN 'no meter data'
        WHEN e.CUSTOMER_NUMBER IS NOT NULL THEN 'dropped: incomplete'
        WHEN g.CUSTOMER_NUMBER IS NULL THEN 'excluded: low usage'
        ELSE 'included'
    END AS status,
    a.CUSTOMER_NUMBER IS NOT NULL AS in_ami,
    e.CUSTOMER_NUMBER IS NOT NULL AS dropped_incomplete,
    e.reason AS drop_reason,
    e.missing_hours AS missing_hours_after_imputation,
    COALESCE(i.imputed_hours, 0) AS imputed_hours,
    s.CUSTOMER_NUMBER IS NOT NULL AS in_ami_analysis,
    s.CUSTOMER_NUMBER IS NOT NULL AND g.CUSTOMER_NUMBER IS NULL AS low_usage,
    g.CUSTOMER_NUMBER IS NOT NULL AS in_peer_groups,
    g.peer_group,
    COALESCE(l.on_ev_list, FALSE) AS on_ev_list,
    COALESCE(l.on_hp_list, FALSE) AS on_hp_list,
    COALESCE(l.on_pv_list, FALSE) AS on_pv_list
FROM all_customers c
LEFT JOIN in_ami a USING (CUSTOMER_NUMBER)
LEFT JOIN excluded_customers e USING (CUSTOMER_NUMBER)
LEFT JOIN in_analysis s USING (CUSTOMER_NUMBER)
LEFT JOIN customer_peer_groups g USING (CUSTOMER_NUMBER)
LEFT JOIN imputed i USING (CUSTOMER_NUMBER)
LEFT JOIN lists l USING (CUSTOMER_NUMBER)
ORDER BY c.CUSTOMER_NUMBER
'''


def build_customer_status() -> pl.DataFrame:
    """Create or replace ``customer_status`` in avangrid.db and return a summary by status."""
    groups = load_customer_groups()
    with db.connect(read_only=False) as con:
        con.register('customer_groups_df', groups)
        con.execute(STATUS_SQL)
        return con.sql('''
            SELECT status, COUNT(*) AS customers,
                   SUM(on_ev_list::INT) AS ev_list, SUM(on_hp_list::INT) AS hp_list, SUM(on_pv_list::INT) AS pv_list
            FROM customer_status GROUP BY 1
            ORDER BY CASE status WHEN 'included' THEN 1 WHEN 'excluded: low usage' THEN 2
                     WHEN 'dropped: incomplete' THEN 3 WHEN 'no meter data' THEN 4 ELSE 5 END
        ''').pl()
