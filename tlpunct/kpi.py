"""Punctuality indicators by line, hour, stop and day type."""
import numpy as np
import pandas as pd

LATE_S = 180     # late from 3 minutes, the threshold used for Swiss public transport punctuality
EARLY_S = -60    # leaving more than a minute early makes passengers miss their bus


def summarise(ev, by):
    g = ev.groupby(by, observed=True)["delay"]
    out = pd.DataFrame({
        "events": g.size(),
        "on_time_share": g.apply(lambda s: float(((s > EARLY_S) & (s < LATE_S)).mean())),
        "late_share": g.apply(lambda s: float((s >= LATE_S).mean())),
        "early_share": g.apply(lambda s: float((s <= EARLY_S).mean())),
        "median_delay_s": g.median(),
        "p90_delay_s": g.quantile(0.9),
    })
    return out.reset_index()


def network(ev):
    d = ev["delay"]
    return {
        "events": int(len(ev)),
        "trips": int(ev["trip_key"].nunique()),
        "lines": int(ev["line"].nunique()),
        "stops": int(ev["stop"].nunique()),
        "on_time_share": float(((d > EARLY_S) & (d < LATE_S)).mean()),
        "late_share": float((d >= LATE_S).mean()),
        "early_share": float((d <= EARLY_S).mean()),
        "median_delay_s": float(np.median(d)),
    }


def delay_growth(ev, max_seq=40):
    """Median delay by position along the trip: how delay builds up from the first stop."""
    e = ev[ev["seq"] <= max_seq]
    return e.groupby("seq")["delay"].median().reset_index()
