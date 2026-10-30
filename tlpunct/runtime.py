"""Running-time analysis: where the timetable gives less time than buses actually need, and what to change.

For two consecutive stops of a trip, the scheduled running time is sched_arr(next) - sched_dep(this) and the
actual one adds the change in delay: arr_delay(next) - dep_delay(this). A positive gain means the bus lost time
on that segment. Segments that lose time on most trips, at the same hours, point at a timetable that is too
tight there, not at bad luck.

`propose` learns, on training days, the extra running time to give each segment and hour band (a share of the
median gain, only where the gain is systematic), and `evaluate` applies those changes to unseen days: shifted
scheduled times, recomputed delays, recomputed punctuality.
"""
import numpy as np
import pandas as pd

from .kpi import EARLY_S, LATE_S

BANDS = [(0, 7, "early"), (7, 9, "am_peak"), (9, 16, "day"), (16, 19, "pm_peak"), (19, 30, "evening")]


def band_of(hour):
    h = np.asarray(hour)
    out = np.full(h.shape, "evening", dtype=object)
    for lo, hi, name in BANDS:
        out[(h >= lo) & (h < hi)] = name
    return out


def segments(df):
    """Consecutive stop pairs of each trip with scheduled and actual running times (measured data only)."""
    d = df[(df.cancelled == 0)].sort_values(["trip_key", "seq"])
    nxt = d.groupby("trip_key").shift(-1)
    seg = pd.DataFrame({
        "trip_key": d["trip_key"], "day": d["day"], "line": d["line"], "seq": d["seq"],
        "from_stop": d["stop"], "to_stop": nxt["stop"],
        "sched_dep": d["sched_dep"], "sched_arr_next": nxt["sched_arr"],
        "dep_delay": d["dep_delay"], "arr_delay_next": nxt["arr_delay"],
        "ok": (d["dep_status"].astype("string") == "REAL") & (nxt["arr_status"].astype("string") == "REAL"),
    })
    seg = seg.dropna(subset=["to_stop", "sched_dep", "sched_arr_next", "dep_delay", "arr_delay_next"])
    seg = seg[seg["ok"]].drop(columns="ok")
    seg["sched_run_s"] = seg["sched_arr_next"] - seg["sched_dep"]
    seg["gain_s"] = seg["arr_delay_next"] - seg["dep_delay"]
    seg = seg[(seg["sched_run_s"] > 0) & (seg["sched_run_s"] < 1800) & seg["gain_s"].between(-600, 900)]
    seg["band"] = band_of(seg["sched_dep"] // 3600)
    from .data import HOLIDAYS
    seg["weekday"] = (seg["day"].dt.weekday < 5) & ~seg["day"].dt.strftime("%Y-%m-%d").isin(HOLIDAYS)
    return seg.reset_index(drop=True)


def propose(seg, min_trips=20, share_positive=0.7, fraction=0.5, cap_s=90):
    """Extra seconds per (line, segment, band) on weekdays where buses lose time on most trips."""
    s = seg[seg["weekday"]]
    g = s.groupby(["line", "from_stop", "to_stop", "band"], observed=True)["gain_s"]
    t = pd.DataFrame({"trips": g.size(), "median_gain_s": g.median(),
                      "positive_share": g.apply(lambda x: float((x > 0).mean())),
                      "sched_run_s": s.groupby(["line", "from_stop", "to_stop", "band"], observed=True)["sched_run_s"].median()})
    t = t[(t.trips >= min_trips) & (t.positive_share >= share_positive) & (t.median_gain_s > 0)]
    t["add_s"] = np.minimum(np.round(fraction * t["median_gain_s"]), cap_s)
    return t[t.add_s > 0].reset_index()


def apply(df, changes):
    """Shift scheduled times of later stops by the running time added upstream on the same trip."""
    d = df[df.cancelled == 0].sort_values(["trip_key", "seq"]).copy()
    d["band"] = band_of(d["sched_dep"].fillna(d["sched_arr"]) // 3600)
    d["next_stop"] = d.groupby("trip_key")["stop"].shift(-1)
    key = changes.set_index(["line", "from_stop", "to_stop", "band"])["add_s"]
    idx = pd.MultiIndex.from_arrays([d["line"], d["stop"], d["next_stop"], d["band"]])
    add = pd.Series(key.reindex(idx).to_numpy(), index=d.index).fillna(0.0)
    add[d["day"].dt.weekday >= 5] = 0.0
    shift = add.groupby(d["trip_key"]).cumsum() - add          # time added on segments before this stop
    d["arr_delay_new"] = d["arr_delay"] - shift
    d["dep_delay_new"] = d["dep_delay"] - shift
    d["added_s"] = add
    return d


def _punct(delay):
    delay = delay.dropna()
    return {"on_time": float(((delay > EARLY_S) & (delay < LATE_S)).mean()),
            "late": float((delay >= LATE_S).mean()), "early": float((delay <= EARLY_S).mean())}


def evaluate(df_test, changes):
    """Punctuality of departures on unseen days, with today's timetable and with the proposed one.

    Buses are assumed to run as they did (same actual times); only the timetable they are judged against
    moves. Departures are held at the stop when early under the new timetable is not modelled, so 'early'
    after the change is an upper bound.
    """
    d = apply(df_test, changes)
    m = (d["dep_status"].astype("string") == "REAL") & d["dep_delay"].between(-600, 3600)
    wk = d["day"].dt.weekday < 5
    before = _punct(d.loc[m & wk, "dep_delay"])
    after = _punct(d.loc[m & wk, "dep_delay_new"])
    trips_changed = d.loc[wk].groupby("trip_key")["added_s"].sum()
    return {"before": before, "after": after,
            "segments_changed": int(len(changes)),
            "median_minutes_added_per_trip": float(trips_changed[trips_changed > 0].median() / 60) if (trips_changed > 0).any() else 0.0,
            "share_trips_changed": float((trips_changed > 0).mean())}
