"""Tests on a synthetic network where the truth is known, plus checks on the real extract when present."""
import numpy as np
import pandas as pd
import pytest

from tlpunct import data, kpi, predict, runtime


def synthetic(days=10, trips_per_day=120, stops=12, slow_segment=(5, 6), slow_s=60, seed=0):
    """Line A: every trip loses slow_s seconds between stops 5 and 6 at peak hours, plus noise elsewhere."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(days):
        day = pd.Timestamp("2026-09-01") + pd.Timedelta(days=d)
        for t in range(trips_per_day):
            start = 6 * 3600 + t * 120
            delay = rng.normal(0, 20)
            for s in range(stops):
                sched = start + s * 120
                if s > 0:
                    peak = 16 * 3600 <= sched < 19 * 3600 or 7 * 3600 <= sched < 9 * 3600
                    gain = rng.normal(5, 15)
                    if (s - 1, s) == slow_segment and peak:
                        gain += slow_s
                    delay += gain
                rows.append([day, t, s, "A", "B", 0, 0, f"S{s}", sched, delay, "REAL", sched, delay, "REAL"])
    df = pd.DataFrame(rows, columns=data.COLUMNS)
    df["trip_key"] = df["day"].dt.strftime("%m%d") + "_" + df["trip"].astype(str)
    return df


@pytest.fixture(scope="module")
def syn():
    return synthetic()


def test_stop_events_filters(syn):
    df = syn.copy()
    df.loc[0, "dep_status"] = "PROG"
    df.loc[1, "dep_delay"] = 9999
    df.loc[2, "cancelled"] = 1
    ev = data.stop_events(df)
    assert len(ev) == len(df) - 3
    assert {"hour", "weekday", "daytype"} <= set(ev.columns)


def test_kpis(syn):
    ev = data.stop_events(syn)
    net = kpi.network(ev)
    assert net["on_time_share"] + net["late_share"] + net["early_share"] == pytest.approx(1.0)
    g = kpi.delay_growth(ev)
    assert g["delay"].iloc[-1] > g["delay"].iloc[0]          # delay builds up along the trip


def test_segments_find_the_slow_one(syn):
    seg = runtime.segments(syn)
    peak = seg[seg.band.isin(["am_peak", "pm_peak"])]
    med = peak.groupby(["from_stop", "to_stop"])["gain_s"].median()
    assert med.idxmax() == ("S5", "S6")
    assert med.max() == pytest.approx(65, abs=8)


def test_proposals_and_evaluation(syn):
    train = syn[syn.day < "2026-09-08"]
    test = syn[syn.day >= "2026-09-08"]
    changes = runtime.propose(runtime.segments(train), min_trips=10)
    slow = changes[(changes.from_stop == "S5") & (changes.to_stop == "S6")]
    assert set(slow.band) == {"am_peak"} and (slow.add_s > 20).all()   # trips run 6:00 to 10:30
    res = runtime.evaluate(test, changes)
    assert res["after"]["late"] < res["before"]["late"]


def test_prediction_beats_persistence(syn):
    ev = data.stop_events(syn)
    tr, te = ev[ev.day < "2026-09-08"], ev[ev.day >= "2026-09-08"]
    Ptr, Pte = predict.pairs(tr, horizons=(1, 5)), predict.pairs(te, horizons=(1, 5))
    model, cats = predict.fit(Ptr)
    sc = predict.scores(Pte, predict.predict(model, cats, Pte)).set_index("k")
    # delays grow by a known amount, which persistence ignores
    assert (sc["mae_reduction"] > 0.1).all()


def test_pairs_are_on_the_same_trip(syn):
    ev = data.stop_events(syn)
    P = predict.pairs(ev, horizons=(3,))
    assert np.allclose(P["sched_gap_min"], 6.0)


def test_real_extract_if_present():
    paths = sorted((data.RAW / "sample").glob("*.csv.gz"))
    if not paths:
        pytest.skip("no extract")
    df = data.read_extract(paths)
    assert set(data.COLUMNS) <= set(df.columns)
    ev = data.stop_events(df)
    assert len(ev) > 10_000 and ev["line"].nunique() > 10
    assert ev["delay"].between(-600, 3600).all()


def test_dashboard_runs():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    root = Path(__file__).resolve().parents[1]
    if not any((root / "results" / d / "summary.json").exists() for d in ("", "sample")):
        pytest.skip("run experiments/run_all.py first")
    at = AppTest.from_file(str(root / "app" / "dashboard.py"), default_timeout=120).run()
    assert not at.exception and len(at.tabs) == 7


def regular(syn):
    df = syn.copy()
    df["arr_delay"] = df["dep_delay"] = 30.0               # every bus 30 s late: perfectly regular
    return df


def test_regularity_regular_service_has_no_excess_wait(syn):
    from tlpunct import regularity
    h = regularity.headways(regular(syn))
    s = regularity.summary(h, ["line"]).iloc[0]
    assert s["sched_headway_min"] == pytest.approx(2.0)
    assert s["bunched_share"] < 0.02 and abs(s["ewt_min"]) < 0.1


def test_regularity_detects_bunching(syn):
    from tlpunct import regularity
    df = regular(syn)
    odd = df["trip"] % 2 == 1                   # every other bus 90 s late from stop 4: pairs of buses
    late = odd & (df["seq"] >= 4)
    for c in ("arr_delay", "dep_delay"):
        df.loc[late, c] += 90
    h = regularity.headways(df)
    along = regularity.along(h, [f"S{i}" for i in range(12)])
    assert along.loc[along.position < 4, "bunched_share"].max() < 0.05
    assert along.loc[along.position >= 4, "bunched_share"].min() > 0.4
    assert regularity.summary(h, ["line"]).iloc[0]["ewt_min"] > 0.2


def hub_network(late_s):
    """Line A arrives at hub H at :00 every 10 min, line B leaves H at :04; A runs late_s late at H."""
    rows = []
    for d in range(3):
        day = pd.Timestamp("2026-09-01") + pd.Timedelta(days=d)
        for t in range(60):
            base = 7 * 3600 + t * 600
            rows += [[day, t, 0, "A", "B", 0, 0, "X1", base - 300, 0.0, "REAL", base - 300, 0.0, "REAL"],
                     [day, t, 1, "A", "B", 0, 0, "H", base, late_s, "REAL", base + 20, late_s, "REAL"],
                     [day, t, 2, "A", "B", 0, 0, "X2", base + 300, late_s, "REAL", base + 300, late_s, "REAL"]]
            rows += [[day, 1000 + t, 0, "B", "B", 0, 0, "H", np.nan, np.nan, "", base + 240, 0.0, "REAL"],
                     [day, 1000 + t, 1, "B", "B", 0, 0, "Y1", base + 500, 0.0, "REAL", base + 500, 0.0, "REAL"]]
    df = pd.DataFrame(rows, columns=data.COLUMNS)
    df["trip_key"] = df["day"].dt.strftime("%m%d") + "_" + df["trip"].astype(str)
    return df


def test_connections_kept_and_missed():
    from tlpunct import connections
    ok = connections.connections(hub_network(60), ["H"])          # 4 min planned, 1 min late: kept
    assert len(ok) > 0 and not ok["missed"].any()
    bad = connections.connections(hub_network(150), ["H"])        # 2.5 min late + 2 min walk: missed by 30 s
    assert bad["missed"].all()
    assert bad["extra_wait_s"].median() == pytest.approx(600)     # next B ten minutes later
    assert connections.hold(bad, 60)["missed_share_after"] < 0.05
    assert connections.hold(bad, 20)["recovered"] == 0


def test_delay_created_ranks_the_slow_segment(syn):
    from tlpunct import geo
    t = geo.delay_created(geo.legs(data.stop_events(syn)))
    assert tuple(t.iloc[0][["from_station", "to_station"]]) == ("S5", "S6")


def test_prediction_interval_covers(syn):
    ev = data.stop_events(syn)
    tr, te = ev[ev.day < "2026-09-08"], ev[ev.day >= "2026-09-08"]
    Ptr, Pte = predict.pairs(tr, horizons=(1, 3)), predict.pairs(te, horizons=(1, 3))
    qs = predict.fit_quantiles(Ptr)
    model, cats = predict.fit(Ptr)
    X, _ = predict.encode(Pte, cats)
    lo = Pte["delay_now"].to_numpy() + qs[0.1].predict(X)
    hi = Pte["delay_now"].to_numpy() + qs[0.9].predict(X)
    cov = predict.interval_scores(Pte, lo, hi)["coverage"]
    assert cov.between(0.65, 0.92).all()
