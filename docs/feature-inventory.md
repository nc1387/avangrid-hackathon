# AMI Feature Inventory

Every per-customer feature built from the hourly meter and weather data (24,968 customers, Sep 2025 – Aug 2026), how each is derived, and which analyses use it. Times are clock times; the data's hour_ending 1 is 12–1 am.

**Legend:** ● used in the final version · ○ tested in a model comparison, not kept · — not used

## At a glance

| Feature family | Features | EV | HP | PV | Appliance rules | Clustering | Source |
|---|---|---|---|---|---|---|---|
| Whole-year usage profile | 23 | ● | ● | ● | — | — | `notebook 40 · customer_features.py` |
| Appliance evidence | 21 | ● | ● | — | ● (14) | ● (3) | `appliances.py` |
| Heat-pump physics | 9 | — | ● | — | — | — | `physics.py` |
| EV night model (per night) | 14 | ● | — | — | — | — | `ev_nights.py` |
| Solar index signature | 3 | ○ | ○ | ● | — | — | `signatures.py` |
| Charging-block signature | 6 | ○ | ○ | ● | — | — | `signatures.py` |
| Cold-response signature | 6 | ○ | ○ | ● | — | — | `signatures.py` |
| Peak load | 15 | ○ | ● | ● | — | ● (2) | `signatures.py` |
| Peer-relative, full year | 15 | ○ | ○ | ○ | — | — | `notebook 50/60 · features.py` |
| Mid-year change | 9 | ○ | ○ | ○ | — | — | `notebook 60 · stability.py` |
| Segmentation | 11 | — | — | — | — | ● | `segmentation.py` |
| **All families** | **132** | | | | | | |

A number in brackets means only that many of the family's features are used there.

## All features

### Whole-year usage profile (23)

Size, timing, solar and high-power counts, and simple weather slopes, over the full year. Source: `notebook 40 · customer_features.py`.

| Feature | How it is derived | EV | HP | PV | Appliance rules | Clustering |
|---|---|---|---|---|---|---|
| `mean_kwh` | Average hourly kWh over the year. | ● | ● | ● | — | — |
| `p95_kwh` | 95th percentile of hourly kWh. | ● | ● | ● | — | — |
| `max_kwh` | Highest single hour of the year. | ● | ● | ● | — | — |
| `winter_summer_ratio` | Average use in Dec–Feb divided by average use in Jun–Aug. | ● | ● | ● | — | — |
| `cold_night_ratio` | Night (12–6 am) use in the coldest temperature quartile divided by night use in the third quartile. | ● | ● | ● | — | — |
| `summer_midday_zero_share` | Share of Apr–Sep hours from 10 am to 3 pm with use under 0.05 kWh (solar covering the home). | ● | ● | ● | — | — |
| `summer_midday_kwh` | Average kWh from 10 am to 3 pm in Apr–Sep. | ● | ● | ● | — | — |
| `summer_night_kwh` | Average kWh from 12 to 6 am in Apr–Sep. | ● | ● | ● | — | — |
| `winter_midday_kwh` | Average kWh from 10 am to 3 pm in Nov–Jan. | ● | ● | ● | — | — |
| `night_share`<br>`morning_share`<br>`afternoon_share`<br>`evening_share` | Average use in each 6-hour block (from midnight) divided by the overall average. | ● | ● | ● | — | — |
| `summer_midday_night_ratio` | summer_midday_kwh divided by summer_night_kwh. | ● | ● | ● | — | — |
| `midday_summer_winter_ratio` | summer_midday_kwh divided by winter_midday_kwh. | ● | ● | ● | — | — |
| `high_sessions` | Number of runs of hours at least 3 kWh above the customer’s median for that hour of day. | ● | ● | ● | — | — |
| `share_high_hours` | Share of hours at least 3 kWh above that hour’s median. | ● | ● | ● | — | — |
| `sharp_rises` | Number of hour-to-hour increases of 3 kWh or more. | ● | ● | ● | — | — |
| `heating_slope` | Slope of daily kWh on heating degrees (55 °F base) over days below 55 °F. | ● | ● | ● | — | — |
| `cooling_slope` | Slope of daily kWh on cooling degrees (65 °F base) over days above 65 °F. | ● | ● | ● | — | — |
| `mild_day_kwh` | Average daily kWh on 55–65 °F days. | ● | ● | ● | — | — |
| `heating_slope_rel`<br>`cooling_slope_rel` | heating_slope and cooling_slope divided by mild_day_kwh. | ● | ● | ● | — | — |

### Appliance evidence (21)

Evidence features behind the appliance rules; all of them also feed the EV and HP models. Source: `appliances.py`.

| Feature | How it is derived | EV | HP | PV | Appliance rules | Clustering |
|---|---|---|---|---|---|---|
| `ac_slope` | Daily kWh per cooling degree above 65 °F, fitted on days averaging 55 °F or more with a weekend term. | ● | ● | — | — | — |
| `ac_t` | t-statistic of ac_slope. | ● | ● | — | ● | — |
| `ac_share` | Share of Jun–Aug use explained by the cooling term. | ● | ● | — | ● | — |
| `ac_added_kw` | Average kW from 2 to 7 pm on days of 80 °F or more, minus the same on 60–68 °F days; also splits central from room AC. | ● | ● | — | ● | ● |
| `heat_slope` | Nov–Mar slope of daily-use anomalies (each day minus its 15-day centered mean) on heating-degree anomalies. | ● | ● | — | — | — |
| `heat_t` | t-statistic of heat_slope. | ● | ● | — | ● | — |
| `heat_kw_at_20f` | heat_slope × 35 ÷ 24: the average extra kW on a 20 °F day. | ● | ● | — | ● | ● |
| `heat_share` | Share of Dec–Feb use explained by heating. | ● | ● | — | — | — |
| `heat_added_kw` | Average kW on days below 25 °F minus on 55–65 °F days. | ● | ● | — | — | — |
| `wh_morning_share` | On 55–70 °F days, share of days with an hour from 4 to 10 am at least 2 kWh above that night’s 12–5 am minimum. | ● | ● | — | ● | — |
| `wh_evening_share` | The same as wh_morning_share for 4 to 10 pm. | ● | ● | — | — | — |
| `wh_morning_kwh` | Median size of those morning spikes. | ● | ● | — | — | — |
| `pool_day_floor_jump` | Minimum use from 9 am to noon on cool summer days, minus the same on mild spring and autumn days. | ● | ● | — | ● | — |
| `pool_day_night_jump` | The same comparison for the gap between the mid-morning and overnight minimums. | ● | ● | — | ● | — |
| `ev_l1_night_share` | Share of mild nights (42–65 °F) with 5+ consecutive hours 0.8–2.5 kWh above the customer’s quiet-night level. | ● | ● | — | ● | — |
| `ev_l1_excess_kwh` | Median size of that Level-1-like excess. | ● | ● | — | — | — |
| `ev_l2_night_share` | Share of mild nights with 2+ consecutive hours at least 3 kWh above the quiet-night level. | ● | ● | — | ● | — |
| `hot_tub_night_share` | Share of mild nights with a 1–2 hour spike at least 1.5 kWh above that night’s minimum between midnight and 6 am. | ● | ● | — | ● | — |
| `solar_midday_zero_share` | Share of Apr–Sep hours from 10 am to 3 pm under 0.05 kWh (same definition as summer_midday_zero_share). | ● | ● | — | ● | ● |
| `battery_evening_zero_share` | Share of Apr–Sep days with at least 3 of the hours from 5 to 10 pm under 0.1 kWh. | ● | ● | — | ● | — |
| `summer_morning_kwh` | Average kWh from 5 to 9 am in Apr–Sep, to rule out homes that are empty all summer. | ● | ● | — | ● | — |

### Heat-pump physics (9)

Responses heat pumps have and resistance or gas heat do not, scaled by the customer’s average daily kWh. Source: `physics.py`.

| Feature | How it is derived | EV | HP | PV | Appliance rules | Clustering |
|---|---|---|---|---|---|---|
| `hp_frost_coef`<br>`hp_frost_t` | Nov–Mar use anomaly per unit of humidity during 25–40 °F hours (defrost cycles), with heating degrees and wind held fixed, and its t-statistic. | — | ● | — | — | — |
| `hp_wind_coef`<br>`hp_wind_t` | Use anomaly per unit of wind × heating degrees (air leakage), and its t-statistic. | — | ● | — | — | — |
| `hp_hdd_partial` | Heating response per degree with frost and wind held fixed. | — | ● | — | — | — |
| `hp_mild_band_slope` | Heating response on 40–55 °F days only, when heat pumps run and gas homes barely change. | — | ● | — | — | — |
| `hp_backup_step` | Use in hours below 12 °F above what the 12–44 °F trend predicts (backup resistance strips). | — | ● | — | — | — |
| `hp_heat_cool_log_ratio` | Log of heating kW at 20 °F divided by AC kW on hot afternoons (one system doing both). | — | ● | — | — | — |
| `hp_heat_cool_min` | The smaller of heating kW at 20 °F and AC kW on hot afternoons. | — | ● | — | — | — |

### EV night model (per night) (14)

Scored for every 7 pm–6 am night against the customer’s quiet-night level; a customer’s score is the average of their 20 highest-scoring nights. Source: `ev_nights.py`.

| Feature | How it is derived | EV | HP | PV | Appliance rules | Clustering |
|---|---|---|---|---|---|---|
| `max_excess`<br>`energy_excess` | Largest hourly and total positive use above the quiet-night level. | ● | — | — | — | — |
| `hours_ge3`<br>`hours_ge1` | Hours at least 3 kWh and at least 1 kWh above that level. | ● | — | — | — | — |
| `run_ge3` | Longest run of hours at least 3 kWh above. | ● | — | — | — | — |
| `run_l1` | Longest run at 0.8–2.5 kWh above (Level 1 range). | ● | — | — | — | — |
| `start_ge25` | Position in the night of the first hour at least 2.5 kWh above (−1 if none). | ● | — | — | — | — |
| `sd` | Standard deviation of use within the night. | ● | — | — | — | — |
| `step_up`<br>`step_down` | Largest hour-to-hour rise and fall. | ● | — | — | — | — |
| `night_kwh` | Total kWh over the night. | ● | — | — | — | — |
| `temp`<br>`weekday`<br>`month` | The night’s average temperature, whether it starts on a weekday, and the month (context). | ● | — | — | — | — |

### Solar index signature (3)

Does the day-to-day midday dip rise and fall with sunshine? Separates solar homes from homes empty at midday. Source: `signatures.py`.

| Feature | How it is derived | EV | HP | PV | Appliance rules | Clustering |
|---|---|---|---|---|---|---|
| `solar_index_corr` | Correlation of the customer’s daily midday-dip anomaly with a sunshine index (average anomaly of the 1% of customers with the deepest Apr–Sep dip, leaving the customer out). | ○ | ○ | ● | — | — |
| `solar_index_slope` | Slope of that same relationship. | ○ | ○ | ● | — | — |
| `dip_vs_temp_range_corr` | Correlation of the dip anomaly with the daily temperature-range anomaly (clear days swing more). | ○ | ○ | ● | — | — |

### Charging-block signature (6)

A block is 2–8 consecutive hours at least 2.5 kWh above that hour’s median, at a steady level, with jumps of 2+ kWh at both ends. Source: `signatures.py`.

| Feature | How it is derived | EV | HP | PV | Appliance rules | Clustering |
|---|---|---|---|---|---|---|
| `ev_blocks_per_week` | Number of blocks per week. | ○ | ○ | ● | — | — |
| `ev_block_kwh` | Average level of the blocks above the customer’s usual use. | ○ | ○ | ● | — | — |
| `ev_block_hours` | Average block length in hours. | ○ | ○ | ● | — | — |
| `ev_block_start_concentration` | Largest share of block starts in any 3-hour window (scheduled charging). | ○ | ○ | ● | — | — |
| `ev_block_mild_day_excess` | Share of blocks on 50–70 °F days minus the share of such days overall (rules out heating and cooling). | ○ | ○ | ● | — | — |
| `ev_block_weekday_excess` | Share of blocks on weekdays minus the weekday share of days. | ○ | ○ | ● | — | — |

### Cold-response signature (6)

One fit per customer of daily use (divided by their own average) on heating degrees below 55, 40 and 20 °F, plus cooling and weekend terms. Source: `signatures.py`.

| Feature | How it is derived | EV | HP | PV | Appliance rules | Clustering |
|---|---|---|---|---|---|---|
| `hp_slope_40_55`<br>`hp_slope_20_40`<br>`hp_slope_below_20` | Heating response per degree in each temperature band (convex when efficiency drops or backup heat starts). | ○ | ○ | ● | — | — |
| `hp_deep_cold_extra` | The extra response per degree below 20 °F on top of the warmer bands. | ○ | ○ | ● | — | — |
| `hp_cold_load_factor_change` | Daily load factor (average ÷ peak) on days below 35 °F minus on 50–65 °F days. | ○ | ○ | ● | — | — |
| `hp_cold_morning_ramp_change` | Morning ramp (5–9 am ÷ 12–5 am use) on cold days minus on mild days. | ○ | ○ | ● | — | — |

### Peak load (15)

How high, how often, when, and how coincident with the system peak each customer’s peaks are. Source: `signatures.py`.

| Feature | How it is derived | EV | HP | PV | Appliance rules | Clustering |
|---|---|---|---|---|---|---|
| `peak_daily_rel` | Median of each day’s highest hour divided by the customer’s average hourly use. | ○ | ● | ● | — | — |
| `peak_daily_rel_vs_peers` | peak_daily_rel minus the peer group’s median. | ○ | ● | ● | — | — |
| `peak_daily_cv` | Day-to-day variation of the daily peak. | ○ | ● | ● | — | — |
| `peak_winter_summer_ratio` | Median daily peak in Dec–Feb divided by Jun–Aug. | ○ | ● | ● | — | — |
| `load_factor_annual` | Average hourly use divided by the highest hour of the year. | ○ | ● | ● | — | ● |
| `load_factor_daily` | Median over days of daily average ÷ daily peak. | ○ | ● | ● | — | — |
| `peak_hours_near_max` | Median hours per day within 80% of that day’s peak (broad vs spiky). | ○ | ● | ● | — | — |
| `peak_share_night`<br>`peak_share_morning`<br>`peak_share_evening` | Share of daily peaks falling 12–6 am, 6–10 am and 4–10 pm. | ○ | ● | ● | — | — |
| `peak_hour_concentration` | Largest share of daily peaks in any 3-hour window. | ○ | ● | ● | — | — |
| `grid_peak_ratio` | Average use in the 100 highest system-load hours divided by own average use. | ○ | ● | ● | — | ● |
| `grid_peak_ratio_winter`<br>`grid_peak_ratio_summer` | The same for the 50 highest system hours in Dec–Feb and in Jun–Aug. | ○ | ● | ● | — | — |
| `grid_peak_kwh` | Average kWh in the 100 highest system-load hours (also defines the peak-contributor segment flag). | ○ | ● | ● | — | — |

### Peer-relative, full year (15)

Built on residuals: each hour’s use relative to the customer’s peer group (16 groups) after weather is removed. Source: `notebook 50/60 · features.py`.

| Feature | How it is derived | EV | HP | PV | Appliance rules | Clustering |
|---|---|---|---|---|---|---|
| `e_night`<br>`e_morning`<br>`e_afternoon`<br>`e_evening`<br>`e_midday`<br>`e_level` | Average residual in each time block, at midday (10 am–3 pm) and overall. | ○ | ○ | ○ | — | — |
| `zero_mid_rel` | Daily share of midday hours under 0.05 kWh, minus the peer group’s daily average. | ○ | ○ | ○ | — | — |
| `high_hours_rel`<br>`sessions_rel`<br>`sharp_rises_rel` | Daily counts of high hours (3+ kWh above that hour’s median), high-use session starts and 3+ kWh rises, minus peers. | ○ | ○ | ○ | — | — |
| `dip_rel` | Midday ÷ night use ratio minus the peer group’s (more negative = deeper solar-like dip). | ○ | ○ | ○ | — | — |
| `heat_slope_rel`<br>`cool_slope_rel` | Residual level per heating degree and per cooling degree. | ○ | ○ | ○ | — | — |
| `cold_level_rel` | Average residual on days below 35 °F. | ○ | ○ | ○ | — | — |
| `peer_log_mean_kwh` | Log of average hourly kWh, added to this set as a size term. | ○ | ○ | ○ | — | — |

### Mid-year change (9)

Looks for a lasting step in a technology-specific weekly signal after the customer’s pattern has settled. Source: `notebook 60 · stability.py`.

| Feature | How it is derived | EV | HP | PV | Appliance rules | Clustering |
|---|---|---|---|---|---|---|
| `ev_change_z`<br>`hp_change_z`<br>`pv_change_z` | Largest step-change z-score in a weekly signal: weather-adjusted evening and overnight excess (EV), midday dip (PV), overall residual level in cold weeks (HP). | ○ | ○ | ○ | — | — |
| `ev_change_flag`<br>`hp_change_flag`<br>`pv_change_flag` | z-score above the 95th percentile of customers on no list. | ○ | ○ | ○ | — | — |
| `current_<feature>` | Each peer-relative feature recomputed only for the weeks since the flagged change. | ○ | ○ | ○ | — | — |
| `change_in_<feature>` | Current-period value minus the before-change value. | ○ | ○ | ○ | — | — |
| `current_period_weeks` | Weeks since the flagged change. | ○ | ○ | ○ | — | — |

### Segmentation (11)

Clustering also uses ac_added_kw, heat_kw_at_20f, solar_midday_zero_share, grid_peak_ratio and load_factor_annual (marked in their rows). Source: `segmentation.py`.

| Feature | How it is derived | EV | HP | PV | Appliance rules | Clustering |
|---|---|---|---|---|---|---|
| `log_mean_kwh` | Log of average hourly kWh (size). | — | — | — | — | ● |
| `night_share`<br>`morning_share`<br>`afternoon_share`<br>`evening_share` | Each 6-hour block’s share of the average daily profile. | — | — | — | — | ● |
| `evening_peak_ratio` | Average use from 5 to 9 pm divided by the overall average. | — | — | — | — | ● |
| `heat_rel` | heat_kw_at_20f divided by average kW. | — | — | — | — | ● |
| `cool_rel` | ac_added_kw divided by average kW. | — | — | — | — | ● |
| `log_winter_summer` | Log of average daily use in Dec–Feb divided by Jun–Aug. | — | — | — | — | ● |
| `empty_months` | Number of months averaging under 30% of the customer’s yearly average (seasonal or vacant). | — | — | — | — | ● |
| `daily_cv` | Day-to-day variation of daily kWh. | — | — | — | — | ● |

## What each column means

- **EV.** Bagged positive-unlabeled gradient boosting on the whole-year profile plus appliance evidence, averaged (by percentile rank) with the night-by-night model.
- **HP.** Bagged positive-unlabeled gradient boosting on the whole-year profile, peak load, appliance evidence and heat-pump physics.
- **PV.** Bagged positive-unlabeled gradient boosting on the whole-year profile, all three signature groups and peak load. It uses notebook 40's SQL table of the whole-year features; EV and HP recompute the same definitions in NumPy.
- **Appliance rules.** Physics-based rules for AC, electric heating, water heater, pool, hot tub, EV Level 1 and 2, and solar + battery. Thresholds were set by adding simulated appliances to real homes and measuring how often each rule catches them. "Heat pump likely" also uses the HP model score.
- **Clustering.** k-means with k = 8 on 14 robust-standardized features, which produce the named customer segments.
- **Not covered.** Peer groups, time-of-day blocks and weather quartiles (notebooks 30 and 50) are building blocks for the features above, not features of their own.
