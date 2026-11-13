"""Battery simulator web API + the HTML front end (web/index.html).

Run from the project folder:

    uvicorn app.server:app --port 8000

then open http://localhost:8000. Data come from the local data folder (avangrid.config.DATA_DIR, or set
AVANGRID_DATA_DIR). The front end can also be opened straight from disk (web/index.html); it then calls
http://localhost:8000, or the URL given as ?api=... .

Endpoints (JSON unless noted):
  GET  /api/health                              status, and whether the data are loaded
  GET  /api/meta                                customers, segments, flags, windows, default settings
  POST /api/simulate                            run a scenario (see avangrid.battery_service) -> results + run_id
  GET  /api/runs/{run_id}/plan                  single windows vs coordinated frontier, best window per segment
  GET  /api/runs/{run_id}/ranking               customers ranked (rank_by, segment, customer, limit, offset)
  GET  /api/runs/{run_id}/ranking.csv           the full ranking as CSV
  GET  /api/runs/{run_id}/customers/{number}    one customer: results and an hourly week before/after
"""

import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import avangrid  # noqa: E402,F401  (thread limits before numerical libraries)
from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, Response  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from avangrid import battery_service as svc  # noqa: E402

WEB_DIR = Path(__file__).resolve().parents[1] / 'web'


class BatteryIn(BaseModel):
    capacity_kwh: float = Field(20.16, gt=0, le=200)
    power_kw: float = Field(7.68, gt=0, le=50)
    round_trip_efficiency: float = Field(0.894, gt=0.3, le=1)
    max_charge_kw: float = Field(3.0, gt=0, le=50)
    reserve_fraction: float = Field(0.0, ge=0, lt=1)


class TariffIn(BaseModel):
    summer_months: list[int] = [6, 7, 8, 9]
    on_peak_start: int = Field(16, ge=0, le=23)
    on_peak_end: int = Field(21, ge=1, le=24)
    weekdays_only: bool = True
    summer_on_peak: float = Field(0.30, ge=0, le=5)
    summer_off_peak: float = Field(0.12, ge=0, le=5)
    winter_on_peak: float = Field(0.22, ge=0, le=5)
    winter_off_peak: float = Field(0.11, ge=0, le=5)
    demand_charge: float = Field(8.0, ge=0, le=200)
    demand_basis: str = Field('on_peak', pattern='^(on_peak|any)$')


class DispatchIn(BaseModel):
    strategy: str = Field('hybrid', pattern='^(hybrid|tou|peak_shave)$')
    threshold_quantile: float = Field(0.95, ge=0.5, le=0.999)
    peak_hours: int = Field(100, ge=1, le=2000)


class PlanIn(BaseModel):
    mode: str = Field('coordinated', pattern='^(coordinated|customer|segment|same)$')
    peak_cut: float = Field(0.075, ge=0, le=0.5)
    window: str = 'Midnight - 9 am'
    segment_windows: dict[str, str] = {}


class FiltersIn(BaseModel):
    segments: list[str] | None = None
    flags: list[str] = []


class Scenario(BaseModel):
    battery: BatteryIn = BatteryIn()
    tariff: TariffIn = TariffIn()
    dispatch: DispatchIn = DispatchIn()
    plan: PlanIn = PlanIn()
    filters: FiltersIn = FiltersIn()


@asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=svc.load, daemon=True).start()   # warm the data while the server starts
    yield


app = FastAPI(title='Avangrid BTM battery simulator', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])


def _run_or_404(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except KeyError:
        raise HTTPException(404, 'Run not found (it may have expired): run the simulation again.')
    except (ValueError, LookupError) as e:
        raise HTTPException(400, str(e))


@app.get('/api/health')
def health():
    return {'status': 'ok', 'data_loaded': svc._data is not None}


@app.get('/api/meta')
def meta():
    return svc.meta()


@app.post('/api/simulate')
def simulate(scenario: Scenario):
    try:
        return svc.simulate(scenario.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get('/api/runs/{run_id}/plan')
def plan(run_id: str):
    return _run_or_404(svc.plan, run_id)


@app.get('/api/runs/{run_id}/ranking')
def ranking(run_id: str, rank_by: str = 'bill_savings', segment: str | None = None, customer: int | None = None,
            limit: int = Query(1000, ge=1, le=50000), offset: int = Query(0, ge=0)):
    df = _run_or_404(svc.ranking, run_id, rank_by, segment, customer)
    return {'total': df.height, 'rows': df.slice(offset, limit).to_dicts()}


@app.get('/api/runs/{run_id}/ranking.csv')
def ranking_csv(run_id: str, rank_by: str = 'bill_savings', segment: str | None = None):
    df = _run_or_404(svc.ranking, run_id, rank_by, segment)
    return Response(df.write_csv(), media_type='text/csv',
                    headers={'Content-Disposition': f'attachment; filename="battery_ranking_{rank_by}.csv"'})


@app.get('/api/runs/{run_id}/customers/{number}')
def customer(run_id: str, number: int, week: int = 45):
    return _run_or_404(svc.customer, run_id, number, week)


@app.get('/')
def index():
    return FileResponse(WEB_DIR / 'index.html')
