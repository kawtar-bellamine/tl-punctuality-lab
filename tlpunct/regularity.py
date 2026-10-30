"""Headway regularity on frequent lines: bunching, gaps and the waiting time passengers actually get.

On a line that runs every few minutes, passengers do not read the timetable, they turn up and wait. What matters
then is the interval between two buses, not the delay of each one. Two late buses five minutes apart are fine;
one on time and the next one glued to it leave a long gap, a full bus and an empty one.

For every stop and direction (the pair stop, next stop) of each line, consecutive departures of the same day
give a scheduled headway and an actual one. Indicators, on weekdays, for scheduled headways of 10 minutes or less:

* bunching: actual headway under half the scheduled one;
* gap: actual headway over one and a half times the scheduled one;
* excess wait time (EWT), the standard indicator for frequent services: average wait passengers get minus the
  wait the timetable promises, for passengers arriving at random, wait = sum(h^2) / (2 sum(h)).
"""
import numpy as np
import pandas as pd

FREQUENT_MAX_S = 600
BUNCH, GAP = 0.5, 1.5


def headways(df, frequent_max_s=FREQUENT_MAX_S, hours=(6, 21)):
    """Consecutive departures at the same stop, same direction and line, same weekday."""
    from .data import HOLIDAYS
    d = df[(df.cancelled == 0)].sort_values(["trip_key", "seq"])
    d = d.assign(next_stop=d.groupby("trip_key")["stop"].shift(-1))
    d = d[d["next_stop"].notna() & d["dep_delay"].notna() & (d["dep_status"].astype("string") == "REAL")
          & d["dep_delay"].between(-600, 3600)]
    d = d[(d["day"].dt.weekday < 5) & ~d["day"].dt.strftime("%Y-%m-%d").isin(HOLIDAYS)]
    d = d[(d["sched_dep"] >= hours[0] * 3600) & (d["sched_dep"] < hours[1] * 3600)]
    d = d[["day", "line", "stop", "next_stop", "seq", "trip_key", "sched_dep", "dep_delay"]].copy()
    d["act_dep"] = d["sched_dep"] + d["dep_delay"]
    d = d.sort_values(["line", "stop", "next_stop", "day", "sched_dep"])
    g = d.groupby(["line", "stop", "next_stop", "day"], observed=True, sort=False)
    d["sched_h"] = g["sched_dep"].diff()
    d["act_h"] = d["act_dep"] - g["act_dep"].shift(1)
    h = d.dropna(subset=["sched_h", "act_h"])
    h = h[(h["sched_h"] > 0) & (h["sched_h"] <= frequent_max_s)].copy()
    h["act_h"] = h["act_h"].clip(lower=0)
    h["hour"] = (h["sched_dep"] // 3600).astype("int16")
    h["bunched"] = h["act_h"] < BUNCH * h["sched_h"]
    h["gap"] = h["act_h"] > GAP * h["sched_h"]
    return h.reset_index(drop=True)


def _wait(x):
    return float((x ** 2).sum() / (2 * x.sum())) if x.sum() > 0 else np.nan


def summary(h, by):
    """Bunching, gaps and excess wait time (minutes) grouped by `by`."""
    g = h.groupby(by, observed=True)
    out = pd.DataFrame({
        "headways": g.size(),
        "sched_headway_min": g["sched_h"].median() / 60,
        "bunched_share": g["bunched"].mean(),
        "gap_share": g["gap"].mean(),
        "swt_min": g["sched_h"].apply(_wait) / 60,
        "awt_min": g["act_h"].apply(_wait) / 60,
    })
    out["ewt_min"] = out["awt_min"] - out["swt_min"]
    return out.reset_index()


def main_pattern(df, line):
    """Most frequent ordered stop sequence of a line (one direction), as a list of stop ids."""
    d = df[(df.line == line) & (df.cancelled == 0)].sort_values(["trip_key", "seq"])
    seqs = d.groupby("trip_key", observed=True)["stop"].agg(lambda x: ">".join(map(str, x)))
    return seqs.value_counts().index[0].split(">") if len(seqs) else []


def along(h, pattern):
    """Bunching and excess wait at each stop of a stop pattern, in order: where regularity breaks down."""
    pairs = pd.DataFrame({"stop": pattern[:-1], "next_stop": pattern[1:], "position": range(len(pattern) - 1)})
    hh = h.merge(pairs, on=["stop", "next_stop"])
    if hh.empty:
        return pairs.assign(headways=0, bunched_share=np.nan, gap_share=np.nan, ewt_min=np.nan)
    s = summary(hh, ["position", "stop"])
    return pairs.merge(s, on=["position", "stop"], how="left")
