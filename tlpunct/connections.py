"""Connection reliability at the main hubs of the network.

A planned connection: a bus of line A arrives at a hub, and the first departure of another line B (in one
direction) leaves at least `plan_min_s` later in the timetable, and at most `plan_max_s` later. The passenger
needs `walk_s` to change. With actual times:

* kept if the planned departure leaves at least `walk_s` after the actual arrival (a passenger who can
  catch an earlier, late-running bus of B is better off and counted as kept);
* missed otherwise; the passenger takes the next departure of B in that direction, and the extra wait is
  its actual departure minus the actual departure of the planned one.

Ridership is not public, so every planned connection counts the same. `hold` estimates what a short holding
rule would recover: missed connections where B would only have had to wait up to `hold_s` more.
"""
import numpy as np
import pandas as pd

WALK_S, PLAN_MIN_S, PLAN_MAX_S = 120, 180, 600


def hubs(ev, n=12):
    """Stations (first 7 digits of the platform id) served by the most lines on weekdays."""
    w = ev[ev["daytype"] == "weekday"]
    return w.assign(station=w["stop"].str[:7]).groupby("station")["line"].nunique().nlargest(n).index.tolist()


def connections(df, hub_ids, walk_s=WALK_S, plan_min_s=PLAN_MIN_S, plan_max_s=PLAN_MAX_S, hours=(6, 21)):
    from .data import HOLIDAYS
    d = df[df.cancelled == 0].sort_values(["trip_key", "seq"])
    g = d.groupby("trip_key")
    d = d.assign(station=d["stop"].str[:7], next_stop=g["stop"].shift(-1), prev_stop=g["stop"].shift(1))
    for c in ("sched_arr", "arr_delay", "sched_dep", "dep_delay"):
        d[c] = d[c].astype("float64")
    d = d[d["station"].isin(hub_ids) & (d["day"].dt.weekday < 5) & ~d["day"].dt.strftime("%Y-%m-%d").isin(HOLIDAYS)]
    A = d[(d["arr_status"].astype("string") == "REAL") & d["arr_delay"].between(-600, 3600) & d["sched_arr"].notna()]
    A = A[(A["sched_arr"] >= hours[0] * 3600) & (A["sched_arr"] < hours[1] * 3600)]
    A = A[["day", "station", "line", "prev_stop", "trip_key", "sched_arr", "arr_delay"]].copy()
    A["act_arr"] = A["sched_arr"] + A["arr_delay"]
    D = d[(d["dep_status"].astype("string") == "REAL") & d["dep_delay"].between(-600, 3600) & d["next_stop"].notna()]
    D = D[["day", "station", "line", "next_stop", "trip_key", "sched_dep", "dep_delay"]].copy()
    D["act_dep"] = D["sched_dep"] + D["dep_delay"]
    out = []
    for (station, line, nxt), dep in D.groupby(["station", "line", "next_stop"], observed=True):
        if len(dep) < 50:
            continue
        # arrivals of other lines, not coming from where this departure goes (no backtracking)
        arr = A[(A["station"] == station) & (A["line"] != line) & (A["prev_stop"] != nxt)]
        if arr.empty:
            continue
        plan = dep[["day", "sched_dep", "act_dep", "trip_key"]].rename(
            columns={"act_dep": "plan_act", "trip_key": "trip_plan"}).sort_values("sched_dep")
        x = arr.assign(key=arr["sched_arr"] + plan_min_s).sort_values("key")
        p = pd.merge_asof(x, plan, left_on="key", right_on="sched_dep", by="day", direction="forward")
        p = p.dropna(subset=["sched_dep"])
        p = p[p["sched_dep"] - p["sched_arr"] <= plan_max_s]
        if p.empty:
            continue
        caught = dep[["day", "act_dep", "trip_key"]].rename(columns={"act_dep": "caught_act", "trip_key": "trip_caught"})
        p = p.assign(key=p["act_arr"] + walk_s).sort_values("key")
        q = pd.merge_asof(p, caught.sort_values("caught_act"), left_on="key", right_on="caught_act", by="day",
                          direction="forward")
        q = q.dropna(subset=["caught_act"])
        q["to_line"], q["to_next"] = line, nxt
        out.append(q[["day", "station", "line", "to_line", "to_next", "sched_arr", "act_arr", "sched_dep",
                      "plan_act", "caught_act", "trip_plan", "trip_caught"]])
    if not out:
        return pd.DataFrame()
    C = pd.concat(out, ignore_index=True)
    C["missed"] = C["plan_act"] < C["act_arr"] + walk_s     # the planned bus left before the passenger got there
    C["extra_wait_s"] = (C["caught_act"] - C["plan_act"]).clip(lower=0).where(C["missed"], 0.0)
    C["short_by_s"] = (C["act_arr"] + walk_s - C["plan_act"]).clip(lower=0).where(C["missed"], 0.0)
    C["hour"] = (C["sched_arr"] // 3600).astype("int16")
    return C


def summary(C, by=None):
    def agg(c):
        m = c["missed"]
        return pd.Series({"connections": len(c), "missed_share": m.mean(),
                          "extra_wait_min_when_missed": c.loc[m, "extra_wait_s"].mean() / 60 if m.any() else 0.0,
                          "missed_by_60s_or_less": (c.loc[m, "short_by_s"] <= 60).mean() if m.any() else 0.0})
    if by is None:
        return agg(C).to_dict()
    return C.groupby(by, observed=True).apply(agg, include_groups=False).reset_index()


def hold(C, hold_s=60):
    """Connections recovered if the departing bus waits up to hold_s for a late arrival, and departures held."""
    saved = C["missed"] & (C["short_by_s"] <= hold_s)
    held = C.loc[saved].groupby(["day", "trip_plan", "station"], observed=True)["short_by_s"].max()
    return {"hold_s": hold_s, "missed": int(C["missed"].sum()), "recovered": int(saved.sum()),
            "missed_share_after": float((C["missed"] & ~saved).mean()),
            "departures_held": int(len(held)),
            "departures_held_share": float(len(held) / C[["day", "trip_plan", "station"]].drop_duplicates().shape[0]),
            "mean_hold_s": float(held.mean()) if len(held) else 0.0}
