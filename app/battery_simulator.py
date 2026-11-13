"""Behind-the-meter battery simulator (Streamlit).

Run from the project folder:

    streamlit run app/battery_simulator.py

Simulates a battery at every customer (or a filtered group) using their actual hourly load, and shows
peak demand reduction, bill savings, and the shift of kWh from on-peak to off-peak hours, for the whole
dataset, by customer segment, and customer by customer (ranked by savings).

All rates and the demand charge are PLACEHOLDERS - replace them with the actual tariff.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import avangrid  # noqa: E402,F401  (thread limit before numerical libraries)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import polars as pl  # noqa: E402
import streamlit as st  # noqa: E402
from avangrid import battery as bt, db, grids  # noqa: E402

st.set_page_config(page_title='Battery simulator', layout='wide')
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
FLAGS = {'peak_contributor': 'Peak contributors', 'high_electrification': 'High electrification',
         'flexible_load': 'Flexible load', 'hp_retrofit_candidate': 'Heat-pump retrofit candidates',
         'ev_adoption_candidate': 'EV adoption candidates', 'solar': 'Solar', 'seasonal': 'Seasonal'}
COLORS = {'before': '#8c8c8c', 'after': '#0072B2', 'accent': '#E69F00'}


# ------------------------------------------------------------------------------------------- data
@st.cache_resource(show_spinner='Loading hourly data (first run only)...')
def load_data():
    hd = grids.load_hourly(verbose=False)
    L = bt.fill_missing(hd.kwh, hd.dates).reshape(hd.shape[0], -1)
    seg = (pl.DataFrame({'CUSTOMER_NUMBER': hd.customer_numbers})
           .join(db.sql('SELECT * FROM customer_segments'), on='CUSTOMER_NUMBER', how='left'))
    return hd.dates, hd.customer_numbers, L, seg


def battery_and_tariff(p: dict):
    battery = bt.Battery(p['capacity'], p['power'], p['rte'], max_charge_kw=p['charge_limit'], reserve_fraction=p['reserve'])
    tariff = bt.Tariff(summer_months=p['summer_months'], on_peak_start_hour_ending=p['on_start'],
                       on_peak_end_hour_ending=p['on_end'], on_peak_weekdays_only=p['weekdays'],
                       summer_on_peak=p['s_on'], summer_off_peak=p['s_off'], winter_on_peak=p['w_on'],
                       winter_off_peak=p['w_off'], demand_charge_per_kw_month=p['demand'],
                       demand_basis=p['demand_basis'])
    return battery, tariff


def segment_labels(rows: tuple) -> np.ndarray:
    return load_data()[3]['segment'].fill_null('Unassigned').to_numpy()[np.array(rows)]


@st.cache_data(show_spinner='Simulating every charging window for every customer (10-20 s)...', max_entries=4)
def compare(core: tuple, rows: tuple) -> dict:
    """Each window applied to everyone: per-customer savings and system load by group (for the planner)."""
    p = dict(core)
    dates, _, L, _ = load_data()
    battery, tariff = battery_and_tariff(p)
    return bt.compare_windows(L[np.array(rows)], dates, segments=segment_labels(rows), battery=battery, tariff=tariff,
                              strategy=p['strategy'], threshold_quantile=p['threshold'], system_peak_hours=p['peak_hours'])


@st.cache_data(show_spinner='Tracing the savings vs system-peak tradeoff...', max_entries=4)
def frontier(core: tuple, rows: tuple) -> pl.DataFrame:
    return bt.peak_savings_frontier(compare(core, rows), points=7)


@st.cache_data(show_spinner='Simulating batteries...', max_entries=8)
def run(core: tuple, plan: tuple, rows: tuple):
    p, plan = dict(core), dict(plan)
    dates, _, L, _ = load_data()
    battery, tariff = battery_and_tariff(p)
    idx = np.array(rows)
    info = {}
    if plan['mode'] == 'same':
        names, windows, window_of = [plan['window']], [bt.CHARGE_WINDOWS[plan['window']]], np.zeros(len(idx), dtype=int)
    else:
        cmp = compare(core, rows)
        names, windows = cmp['names'], cmp['windows']
        if plan['mode'] == 'customer':
            window_of = bt.assign_windows(cmp, 'customer')
        elif plan['mode'] == 'segment':
            chosen = dict(plan['segment_windows'])
            window_of = np.array([names.index(chosen[s]) for s in segment_labels(rows)])
        else:
            co = bt.coordinate_windows(cmp, cmp['system_before'].max() * (1 - plan['peak_cut']))
            window_of = co['window_of']
            info = {'min_peak_kw': co['min_peak_kw'], 'peak_cap_kw': co['peak_cap_kw']}
    res = bt.simulate(L[idx], dates, battery, tariff, strategy=p['strategy'], threshold_quantile=p['threshold'],
                      system_peak_hours=p['peak_hours'], windows=windows, window_of=window_of)
    h = tariff.hourly(dates)
    return {k: res[k] for k in ['customers', 'system_before', 'system_after', 'system_peak_hours']} | {
        'on_peak': h['on_peak'], 'month': h['month'], 'window_name': np.array(names, dtype=object)[window_of],
        'plan_info': info}


dates, customer_numbers, L, segments = load_data()

# ------------------------------------------------------------------------------------------- inputs
st.sidebar.header('Battery')
capacity = st.sidebar.number_input('Usable capacity (kWh)', 1.0, 100.0, 20.16, 0.5)
power = st.sidebar.number_input('Power (kW)', 0.5, 30.0, 7.68, 0.1)
rte = st.sidebar.slider('Round-trip efficiency', 0.70, 1.00, 0.894, 0.001)
charge_limit = st.sidebar.number_input('Max charging power (kW)', 0.5, 30.0, 3.0, 0.5,
                                       help='Challenge Statement: a 200-amp service should not take more than ~3 kVA '
                                            'of additional load without further evaluation. Battery charging is added '
                                            'load, so charging is capped at 3 kW by default. Discharge uses the full power.')
reserve = st.sidebar.slider('Backup reserve (share of capacity not used)', 0.0, 0.5, 0.0, 0.05)
st.sidebar.caption('Defaults: Enphase IQ Battery 10C x 2 from the Challenge Statement. The 11.52 kW 10-second '
                   'rating is not modelled (hourly data).')

st.sidebar.header('Tariff (placeholders)')
summer_months = st.sidebar.multiselect('Summer months', MONTHS, ['Jun', 'Jul', 'Aug', 'Sep'])
c1, c2 = st.sidebar.columns(2)
on_start = c1.number_input('On-peak starts (hour, 24h)', 0, 23, 16, help='16 = 4 pm')
on_end = c2.number_input('On-peak ends (hour, 24h)', 1, 24, 21, help='21 = 9 pm')
weekdays = st.sidebar.checkbox('On-peak on weekdays only', True)
c1, c2 = st.sidebar.columns(2)
s_on = c1.number_input('Summer on-peak $/kWh', 0.0, 2.0, 0.30, 0.01)
s_off = c2.number_input('Summer off-peak $/kWh', 0.0, 2.0, 0.12, 0.01)
w_on = c1.number_input('Winter on-peak $/kWh', 0.0, 2.0, 0.22, 0.01)
w_off = c2.number_input('Winter off-peak $/kWh', 0.0, 2.0, 0.11, 0.01)
demand = st.sidebar.number_input('Demand charge ($/kW-month)', 0.0, 50.0, 8.0, 0.5)
demand_basis = st.sidebar.radio('Demand charge applies to', ['on_peak', 'any'],
                                format_func=lambda x: 'Monthly max during on-peak hours' if x == 'on_peak' else 'Monthly max, any hour')

st.sidebar.header('Dispatch')
strategy = st.sidebar.selectbox('Strategy', ['hybrid', 'tou', 'peak_shave'], format_func=lambda s: {
    'hybrid': 'Hybrid: cover on-peak + shave other spikes', 'tou': 'Time-of-use: cover on-peak load',
    'peak_shave': 'Peak shaving only'}[s])
threshold = st.sidebar.slider('Peak threshold (percentile of own monthly load)', 0.80, 0.99, 0.95, 0.01,
                              help='Battery discharges above this off-peak (hybrid / peak shaving), and never charges above it.')
peak_hours = st.sidebar.number_input('Grid peak hours to report on', 10, 500, 100, 10)

st.sidebar.header('Customers')
seg_names = sorted(segments['segment'].drop_nulls().unique().to_list())
chosen_segments = st.sidebar.multiselect('Segments', seg_names, seg_names)
chosen_flags = st.sidebar.multiselect('Only customers in these program segments', list(FLAGS), format_func=FLAGS.get)

mask = segments['segment'].is_in(chosen_segments).to_numpy()
for f in chosen_flags:
    mask &= segments[f].fill_null(False).to_numpy()
rows = tuple(np.flatnonzero(mask).tolist())
if not rows:
    st.warning('No customers match the filters.')
    st.stop()

core = tuple(sorted({
    'capacity': capacity, 'power': power, 'rte': rte, 'reserve': reserve, 'charge_limit': charge_limit,
    'summer_months': tuple(MONTHS.index(m) + 1 for m in summer_months),
    'on_start': int(on_start) + 1, 'on_end': int(on_end), 'weekdays': weekdays,
    's_on': s_on, 's_off': s_off, 'w_on': w_on, 'w_off': w_off, 'demand': demand, 'demand_basis': demand_basis,
    'strategy': strategy, 'threshold': threshold, 'peak_hours': int(peak_hours)}.items()))

st.sidebar.header('Charging windows')
MODES = {'coordinated': 'Coordinated: most savings under a system-peak target',
         'customer': 'Best window for each customer',
         'segment': 'Chosen per segment',
         'same': 'Same window for everyone'}
mode = st.sidebar.radio('How each battery\'s charging window is set', list(MODES), format_func=MODES.get,
                        help='Each battery spreads its recharge evenly over the remaining hours of its window. If every '
                             'battery uses the same window they all charge together and can create a new system peak; '
                             'giving customers different windows spreads that load out.')
plan = {'mode': mode}
if mode == 'same':
    plan['window'] = st.sidebar.selectbox('Window', list(bt.CHARGE_WINDOWS))
elif mode == 'coordinated':
    plan['peak_cut'] = st.sidebar.slider('System peak cut target (%)', 0.0, 15.0, 7.5, 0.5,
                                         help='The planner gives each segment a mix of windows that maximizes total '
                                              'savings while keeping the system peak (these customers\' combined load) '
                                              'at least this far below today\'s. If the target is not achievable, the '
                                              'lowest achievable peak is used.') / 100
elif mode == 'segment':
    cmp = compare(core, rows)
    seg_rows = segment_labels(rows)
    with st.sidebar.expander('Window for each segment (defaults: best for the segment)', expanded=True):
        plan['segment_windows'] = tuple(
            (sg, st.selectbox(sg, cmp['names'], index=int(cmp['savings'][seg_rows == sg].sum(0).argmax()), key=f'w_{sg}'))
            for sg in sorted(set(seg_rows)))
plan = tuple(sorted(plan.items()))

res = run(core, plan, rows)
cust = res['customers'].with_columns(
    pl.Series('CUSTOMER_NUMBER', customer_numbers[list(rows)]),
    pl.Series('segment', segments['segment'].to_numpy()[list(rows)]),
    *[pl.Series(f, segments[f].fill_null(False).to_numpy()[list(rows)]) for f in FLAGS],
    pl.Series('charge_window', res['window_name'].astype(str)))

# ------------------------------------------------------------------------------------------- header
st.title('Behind-the-meter battery simulator')
st.caption(f'{len(rows):,} customers simulated with their actual hourly load (Sep 2025 - Aug 2026). '
           'Rates and demand charge are placeholders. Bills cover energy and demand charges only.')
tab_all, tab_plan, tab_seg, tab_rank, tab_one = st.tabs(['Whole dataset', 'Charging plan', 'By segment',
                                                         'Customer ranking', 'Customer detail'])
dict_plan = dict(plan)
if dict_plan['mode'] == 'coordinated' and res['plan_info']['peak_cap_kw'] > res['system_before'].max() * (1 - dict_plan['peak_cut']) + 1:
    st.info(f"A {dict_plan['peak_cut']:.1%} peak cut isn't achievable with these settings; using the lowest achievable "
            f"system peak, {res['plan_info']['peak_cap_kw'] / 1000:,.1f} MW.")

sb, sa = res['system_before'], res['system_after']
on, month = res['on_peak'], res['month']

with tab_all:
    tot = lambda c: float(cust[c].sum())
    k = st.columns(4)
    k[0].metric('Annual bill savings', f"${tot('bill_savings') / 1e6:,.2f}M",
                f"{tot('bill_savings') / tot('bill_before'):.1%} of bills")
    k[1].metric('Avg savings / customer', f"${cust['bill_savings'].mean():,.0f}",
                f"median ${cust['bill_savings'].median():,.0f} ({cust['bill_savings_pct'].median():.0%} of bill)",
                delta_color='off')
    k[2].metric('On-peak kWh removed', f"{tot('on_peak_kwh_decrease') / 1e6:,.1f} GWh",
                f"-{tot('on_peak_kwh_decrease') / tot('on_peak_kwh_before'):.1%}", delta_color='inverse')
    k[3].metric('Off-peak kWh added', f"{tot('off_peak_kwh_increase') / 1e6:,.1f} GWh",
                f"+{tot('off_peak_kwh_increase') / tot('off_peak_kwh_before'):.1%}", delta_color='off')
    k = st.columns(4)
    k[0].metric('Avg monthly peak cut', f"{cust['avg_monthly_peak_kw_reduction'].mean():.2f} kW",
                f"{cust['avg_monthly_peak_kw_reduction'].sum() / cust['avg_monthly_peak_kw_before'].sum():.0%} of their peaks",
                delta_color='off')
    k[1].metric(f'Reduction in the {peak_hours} highest grid hours', f"{tot('grid_peak_kw_reduction') / 1000:,.1f} MW",
                f"{tot('grid_peak_kw_reduction') / tot('grid_peak_kw_before'):.0%} of these customers' load then",
                delta_color='off')
    k[2].metric('System peak (these customers)', f'{sa.max() / 1000:,.1f} MW', f'{(sa.max() - sb.max()) / 1000:+,.2f} MW',
                delta_color='inverse', help='Highest single hour after the batteries, wherever it now falls.')
    k[3].metric('Battery losses', f"{tot('losses_kwh') / 1e6:,.1f} GWh",
                f"{cust['equivalent_full_cycles'].median():.0f} full cycles/yr (median)", delta_color='off')

    c1, c2 = st.columns(2)
    day = int(np.argmax(sb)) // 24
    hrs = np.arange(1, 25)
    fig = go.Figure()
    fig.add_scatter(x=hrs, y=sb[day * 24:(day + 1) * 24] / 1000, name='Before', line=dict(color=COLORS['before'], width=3))
    fig.add_scatter(x=hrs, y=sa[day * 24:(day + 1) * 24] / 1000, name='After', line=dict(color=COLORS['after'], width=3))
    fig.update_layout(title=f'System load on the peak day ({dates[day]})', xaxis_title='Hour ending',
                      yaxis_title='MW', height=380, legend=dict(orientation='h', y=1.12), margin=dict(t=70))
    c1.plotly_chart(fig, width='stretch')

    mdf = pd.DataFrame({'month': month, 'on': on, 'before': sb, 'after': sa})
    agg = mdf.groupby(['month', 'on'])[['before', 'after']].sum().reset_index()
    order = [9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8]
    fig = go.Figure()
    for is_on, name, col in [(True, 'On-peak', COLORS['accent']), (False, 'Off-peak', COLORS['after'])]:
        a = agg[agg['on'] == is_on].set_index('month').reindex(order)
        fig.add_bar(x=[MONTHS[m - 1] for m in order], y=(a['after'] - a['before']) / 1000, name=f'{name} change', marker_color=col)
    fig.update_layout(title='Monthly kWh shift (MWh)', yaxis_title='MWh',
                      barmode='relative', height=380, legend=dict(orientation='h', y=1.12), margin=dict(t=70))
    c2.plotly_chart(fig, width='stretch')

    fig = go.Figure(go.Histogram(x=cust['bill_savings'].to_numpy(), nbinsx=60, marker_color=COLORS['after']))
    fig.update_layout(title='Distribution of annual savings per customer', xaxis_title='$ per year',
                      yaxis_title='Customers', height=320, margin=dict(t=50))
    st.plotly_chart(fig, width='stretch')

with tab_plan:
    cmp = compare(core, rows)
    fr = frontier(core, rows)
    st.markdown('**Savings vs the system peak.** If every battery charges in the same window, they all charge '
                'together and can set a new system peak. Giving customers different windows spreads that load out. '
                'The line is the most savings achievable for each system peak (each segment gets a mix of windows; '
                'from a linear program over the simulated loads, which matched full simulations within 0.1%).')
    fig = go.Figure()
    fig.add_scatter(x=fr['system_peak_kw'] / 1000, y=fr['total_savings'] / 1e6, mode='lines+markers',
                    name='Coordinated windows', line=dict(color=COLORS['after'], width=3))
    sm = cmp['summary']
    label_pos = {'Any off-peak hour': 'top left', '9 pm - noon': 'bottom right', 'Midnight - 9 am': 'bottom center'}
    fig.add_scatter(x=sm['system_peak_mw_after'], y=sm['total_savings'] / 1e6, mode='markers+text', text=sm['window'],
                    textposition=[label_pos.get(w, 'top center') for w in sm['window']], name='Same window for everyone',
                    marker=dict(color=COLORS['before'], size=10, symbol='square'))
    fig.add_scatter(x=[sa.max() / 1000], y=[cust['bill_savings'].sum() / 1e6], mode='markers', name='Current plan',
                    marker=dict(color=COLORS['accent'], size=16, symbol='star'))
    fig.add_vline(x=sb.max() / 1000, line_dash='dash', line_color=COLORS['before'],
                  annotation_text='System peak today', annotation_position='bottom right')
    fig.update_layout(xaxis_title='System peak after batteries (MW)', yaxis_title='Total annual savings ($M)', height=480,
                      legend=dict(orientation='h', y=1.1), margin=dict(t=60))
    st.plotly_chart(fig, width='stretch')

    mix = (cust.group_by('segment', 'charge_window').len()
           .with_columns((pl.col('len') / pl.col('len').sum().over('segment')).alias('share')))
    c1, c2 = st.columns(2)
    fig = go.Figure()
    palette = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00', '#8c8c8c']
    for i, w in enumerate(bt.CHARGE_WINDOWS):
        m = mix.filter(pl.col('charge_window') == w)
        if m.height:
            fig.add_bar(y=m['segment'], x=m['share'] * 100, name=w, orientation='h', marker_color=palette[i % len(palette)])
    fig.update_layout(title='Charging window mix by segment (current plan)', barmode='stack', xaxis_title='% of customers',
                      height=420, legend=dict(orientation='h', y=-0.2), margin=dict(t=50, l=10),
                      yaxis=dict(categoryorder='category descending'))
    c1.plotly_chart(fig, width='stretch')
    c2.markdown('**Customers per window (current plan)**')
    c2.dataframe(cust.group_by('charge_window').agg(pl.len().alias('Customers'),
                                                     pl.col('bill_savings').mean().alias('Avg savings ($)'))
                 .sort('Customers', descending=True), hide_index=True, width='stretch',
                 column_config={'charge_window': 'Window', 'Avg savings ($)': st.column_config.NumberColumn(format='$%.0f')})
    c2.markdown('**Same window for everyone**')
    c2.dataframe(sm.with_columns(pl.col('total_savings') / 1e6), hide_index=True, width='stretch', column_config={
        'window': 'Window',
        'total_savings': st.column_config.NumberColumn('Savings ($M)', format='%.2f'),
        'savings_share_of_bill': st.column_config.NumberColumn('Savings (% of bills)', format='percent'),
        'grid_peak_hour_cut_mw': st.column_config.NumberColumn(f'Top-{peak_hours}-hour cut (MW)', format='%.1f'),
        'system_peak_mw_after': st.column_config.NumberColumn('System peak (MW)', format='%.2f')})

with tab_seg:
    by = (cust.group_by('segment').agg(
        pl.len().alias('Customers'),
        pl.col('bill_savings').sum().round(0).alias('Total savings ($)'),
        pl.col('bill_savings').mean().alias('Avg savings ($)'),
        pl.col('bill_savings').median().alias('Median savings ($)'),
        pl.col('bill_savings_pct').median().alias('Median savings (% of own bill)'),
        (pl.col('bill_savings').sum() / pl.col('bill_before').sum()).alias('Savings (% of bill)'),
        pl.col('avg_monthly_peak_kw_reduction').mean().alias('Avg monthly peak cut (kW)'),
        pl.col('grid_peak_kw_reduction').mean().alias('Grid-peak-hour cut per customer (kW)'),
        (pl.col('grid_peak_kw_reduction').sum() / 1000).alias('Grid-peak-hour cut total (MW)'),
        (pl.col('on_peak_kwh_decrease').sum() / 1000).round(0).alias('On-peak removed (MWh)'),
        (pl.col('off_peak_kwh_increase').sum() / 1000).round(0).alias('Off-peak added (MWh)'),
    ).sort('Avg savings ($)', descending=True))
    st.dataframe(by, hide_index=True, width='stretch', column_config={
        'Total savings ($)': st.column_config.NumberColumn(format='localized'),
        'Avg savings ($)': st.column_config.NumberColumn(format='$%.0f'),
        'Median savings ($)': st.column_config.NumberColumn(format='$%.0f'),
        'Savings (% of bill)': st.column_config.NumberColumn(format='percent'),
        'Median savings (% of own bill)': st.column_config.NumberColumn(format='percent'),
        'Avg monthly peak cut (kW)': st.column_config.NumberColumn(format='%.2f'),
        'Grid-peak-hour cut per customer (kW)': st.column_config.NumberColumn(format='%.2f'),
        'Grid-peak-hour cut total (MW)': st.column_config.NumberColumn(format='%.2f'),
        'On-peak removed (MWh)': st.column_config.NumberColumn(format='localized'),
        'Off-peak added (MWh)': st.column_config.NumberColumn(format='localized')})
    by_pd = by.with_columns((pl.col('Savings (% of bill)') * 100)).to_pandas()
    c1, c2 = st.columns(2)
    fig = go.Figure(go.Bar(y=by_pd['segment'], x=by_pd['Avg savings ($)'], orientation='h', marker_color=COLORS['after'],
                           text=[f'${v:,.0f}' for v in by_pd['Avg savings ($)']], textposition='outside'))
    fig.update_layout(title='Avg annual savings per customer', xaxis_title='$ per year',
                      xaxis=dict(range=[0, by_pd['Avg savings ($)'].max() * 1.25]),
                      yaxis=dict(autorange='reversed'), height=420, margin=dict(t=50, l=10))
    c1.plotly_chart(fig, width='stretch')
    fig = go.Figure(go.Bar(y=by_pd['segment'], x=by_pd['Grid-peak-hour cut per customer (kW)'], orientation='h',
                           marker_color=COLORS['accent'],
                           text=[f'{v:.2f} kW' for v in by_pd['Grid-peak-hour cut per customer (kW)']], textposition='outside'))
    fig.update_layout(title=f'Avg cut in top {peak_hours} grid hours (kW)', xaxis_title='kW per customer',
                      xaxis=dict(range=[0, by_pd['Grid-peak-hour cut per customer (kW)'].max() * 1.35]),
                      yaxis=dict(autorange='reversed'), height=420, margin=dict(t=50, l=10))
    c2.plotly_chart(fig, width='stretch')

with tab_rank:
    rank_by = st.radio('Rank customers by', ['bill_savings', 'bill_savings_pct', 'grid_peak_kw_reduction',
                                             'avg_monthly_peak_kw_reduction'],
                       horizontal=True, format_func={'bill_savings': 'Bill savings ($)', 'bill_savings_pct': 'Bill savings (%)',
                                                     'grid_peak_kw_reduction': 'Grid-peak-hour cut',
                                                     'avg_monthly_peak_kw_reduction': 'Monthly peak cut'}.get)
    ranked = (cust.sort(rank_by, descending=True).with_row_index('rank', offset=1)
              .with_columns(pl.col('bill_before', 'on_peak_kwh_decrease', 'off_peak_kwh_increase').round(0))
              .select('rank', 'CUSTOMER_NUMBER', 'segment', 'charge_window', 'bill_savings', 'bill_savings_pct', 'energy_savings', 'demand_savings',
                      'bill_before', 'avg_monthly_peak_kw_before', 'avg_monthly_peak_kw_reduction', 'grid_peak_kw_reduction',
                      'on_peak_kwh_decrease', 'off_peak_kwh_increase', 'equivalent_full_cycles', *FLAGS))
    st.dataframe(ranked.head(1000), hide_index=True, width='stretch', height=520, column_config={
        'bill_savings': st.column_config.NumberColumn('Savings ($/yr)', format='$%.0f'),
        'charge_window': st.column_config.TextColumn('Charging window'),
        'bill_savings_pct': st.column_config.NumberColumn('Savings (% of bill)', format='percent'),
        'energy_savings': st.column_config.NumberColumn('Energy savings', format='$%.0f'),
        'demand_savings': st.column_config.NumberColumn('Demand savings', format='$%.0f'),
        'bill_before': st.column_config.NumberColumn('Bill before ($)', format='localized'),
        'avg_monthly_peak_kw_before': st.column_config.NumberColumn('Monthly peak (kW)', format='%.2f'),
        'avg_monthly_peak_kw_reduction': st.column_config.NumberColumn('Monthly peak cut (kW)', format='%.2f'),
        'grid_peak_kw_reduction': st.column_config.NumberColumn('Grid-peak-hour cut (kW)', format='%.2f'),
        'on_peak_kwh_decrease': st.column_config.NumberColumn('On-peak kWh removed', format='localized'),
        'off_peak_kwh_increase': st.column_config.NumberColumn('Off-peak kWh added', format='localized'),
        'equivalent_full_cycles': st.column_config.NumberColumn('Cycles/yr', format='%.0f'),
        **{f: st.column_config.CheckboxColumn(FLAGS[f]) for f in FLAGS}})
    st.caption('Showing the top 1,000. Download the full ranking below.')
    st.download_button('Download full ranking (CSV)', ranked.write_csv(), 'battery_ranking.csv', 'text/csv')

with tab_one:
    top = cust.sort('bill_savings', descending=True)['CUSTOMER_NUMBER'].head(500).to_list()
    choice = st.selectbox('Customer (top 500 by savings, or type a number)', top)
    week = st.slider('Week of the year (starting Monday 1 Sep 2025)', 1, 52, 45)
    r = int(np.flatnonzero(customer_numbers == choice)[0])
    w_name = str(res['window_name'][rows.index(r)]) if r in rows else 'Midnight - 9 am'
    battery, tariff = battery_and_tariff(dict(core))
    one = bt.simulate(L[[r]], dates, battery, tariff, strategy=dict(core)['strategy'],
                      threshold_quantile=dict(core)['threshold'], charge_window=bt.CHARGE_WINDOWS[w_name],
                      keep_net_for=np.array([0]))
    st.caption(f'Charging window for this customer: {w_name}')
    s0 = one['customers'].row(0, named=True)
    k = st.columns(4)
    k[0].metric('Annual savings', f"${s0['bill_savings']:,.0f}", f"{s0['bill_savings'] / s0['bill_before']:.0%} of ${s0['bill_before']:,.0f} bill", delta_color='off')
    k[1].metric('Avg monthly peak', f"{s0['avg_monthly_peak_kw_after']:.2f} kW", f"{-s0['avg_monthly_peak_kw_reduction']:+.2f} kW", delta_color='inverse')
    k[2].metric('On-peak kWh', f"{s0['on_peak_kwh_after']:,.0f}", f"{-s0['on_peak_kwh_decrease']:+,.0f}", delta_color='inverse')
    k[3].metric('Off-peak kWh', f"{s0['off_peak_kwh_after']:,.0f}", f"{s0['off_peak_kwh_increase']:+,.0f}", delta_color='off')
    a, b = (week - 1) * 168, week * 168
    x = pd.date_range(str(dates[0]), periods=L.shape[1], freq='h')[a:b]
    fig = go.Figure()
    shade = on[a:b]
    for i in np.flatnonzero(shade & ~np.r_[False, shade[:-1]]):
        j = i
        while j < len(shade) and shade[j]:
            j += 1
        fig.add_vrect(x0=x[i], x1=x[j - 1] + pd.Timedelta(hours=1), fillcolor=COLORS['accent'], opacity=0.12, line_width=0)
    fig.add_scatter(x=x, y=L[r, a:b], name='Before', line=dict(color=COLORS['before'], width=2))
    fig.add_scatter(x=x, y=one['net'][0, a:b], name='After battery', line=dict(color=COLORS['after'], width=2))
    fig.update_layout(title=f'Customer {choice}: hourly load, week {week} (shaded = on-peak)', yaxis_title='kWh per hour',
                      height=420, legend=dict(orientation='h', y=1.1), margin=dict(t=70))
    st.plotly_chart(fig, width='stretch')
