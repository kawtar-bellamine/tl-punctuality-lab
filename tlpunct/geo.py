"""Where on the map delay is created.

Stations come from the official list of Swiss service points (didok, opentransportdata.swiss, via
data.sbb.ch), saved in data/raw/tl_stations.csv: station id (first 7 digits of the platform id), latitude,
longitude. Between two consecutive stops of a trip, the delay created is the change in measured delay from the
departure at the first stop to the departure at the next one: running time and time spent at the next stop
beyond the timetable, which is where traffic, lights and boarding show up. Summed over every weekday trip,
gains and recoveries together, in minutes per weekday.
"""
from pathlib import Path

import pandas as pd

from .data import RAW


def stations(path=None):
    path = Path(path or RAW / "tl_stations.csv")
    if not path.exists():
        return pd.DataFrame(columns=["station", "lat", "lon"])
    return pd.read_csv(path, dtype={"station": str})


def legs(ev):
    """Consecutive measured stop events of each trip on weekdays, with the delay created in between."""
    from .runtime import band_of
    e = ev[ev["daytype"] == "weekday"].sort_values(["trip_key", "seq"])
    nxt = e.groupby("trip_key")[["stop", "delay", "seq"]].shift(-1)
    t = pd.DataFrame({"day": e["day"], "line": e["line"], "from_stop": e["stop"], "to_stop": nxt["stop"],
                      "gain_s": nxt["delay"] - e["delay"], "band": band_of(e["hour"]), "hop": nxt["seq"] - e["seq"]})
    return t[t["to_stop"].notna() & (t["hop"] == 1)].drop(columns="hop")


def delay_created(lg, bands=None):
    """Delay created per station pair on weekdays: minutes per weekday, median gain, trips, lines."""
    s = lg[lg["band"].isin(bands)] if bands else lg
    s = s.assign(from_station=s["from_stop"].str[:7], to_station=s["to_stop"].str[:7])
    s = s[s["from_station"] != s["to_station"]]
    days = max(s["day"].nunique(), 1)
    g = s.groupby(["from_station", "to_station"], observed=True)
    out = pd.DataFrame({"trips_per_day": g.size() / days, "median_gain_s": g["gain_s"].median(),
                        "created_min_per_day": g["gain_s"].sum() / days / 60,
                        "lines": g["line"].agg(lambda x: " ".join(sorted(set(map(str, x)), key=lambda v: (len(v), v))))})
    out = out.reset_index().sort_values("created_min_per_day", ascending=False)
    return out[out["created_min_per_day"] > 0]


def with_coords(t, st=None):
    st = stations() if st is None else st
    c = st.set_index("station")[["lat", "lon"]]
    t = t.join(c.add_prefix("from_"), on="from_station").join(c.add_prefix("to_"), on="to_station")
    return t.dropna(subset=["from_lat", "to_lat"])
