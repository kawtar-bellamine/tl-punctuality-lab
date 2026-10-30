"""Load the tl extract of the Swiss actual-data (Ist-Daten) archive.

The extract keeps only the operator TL (Transports publics de la région lausannoise) from the daily files of
opentransportdata.swiss, one row per trip and stop, in a compact form:

    day, trip, seq, line, mode, cancelled, extra, stop, sched_arr, arr_delay, arr_status,
    sched_dep, dep_delay, dep_status

Times are seconds after midnight of the operating day, delays are seconds (actual or last forecast minus
scheduled), status is REAL (measured), PROG (forecast) or UNBE/GESC (unknown / estimated). `tools/extract_tl.js`
documents how the extract is built from the public archive.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
# public holidays in the canton of Vaud inside the extract (Lundi du Jeûne fédéral), run on a Sunday timetable
HOLIDAYS = {"2026-09-21"}
COLUMNS = ["day", "trip", "seq", "line", "mode", "cancelled", "extra", "stop", "sched_arr", "arr_delay",
           "arr_status", "sched_dep", "dep_delay", "dep_status"]
DTYPES = {"trip": "int32", "seq": "int16", "line": "string", "mode": "category", "cancelled": "int8",
          "extra": "int8", "stop": "string", "sched_arr": "float32", "arr_delay": "float32",
          "arr_status": "category", "sched_dep": "float32", "dep_delay": "float32", "dep_status": "category"}


def read_extract(paths=None):
    """Concatenate the gzipped extract parts into one DataFrame."""
    paths = paths or sorted(RAW.glob("tl_istdaten_*.csv.gz"))
    if not paths:
        raise FileNotFoundError(f"no extract in {RAW}; see README, section Data")
    frames = [pd.read_csv(p, names=COLUMNS, dtype=DTYPES, parse_dates=["day"], na_values=[""]) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    # one trip key per operating day (trip ints were assigned per extraction run)
    df["trip_key"] = df["day"].dt.strftime("%m%d") + "_" + df["trip"].astype(str)
    return df


def stop_events(df, measured_only=True, window=(-600, 3600)):
    """One row per stop event with a scheduled time, a delay and calendar features.

    measured_only keeps delays measured on the vehicle (status REAL); forecasts are what passengers saw, not
    what happened. Delays outside the window (default -10 to +60 min) are treated as data errors.
    """
    ev = df[df.cancelled == 0].copy()
    ev["sched"] = ev["sched_dep"].fillna(ev["sched_arr"])
    ev["delay"] = ev["dep_delay"].where(ev["dep_delay"].notna(), ev["arr_delay"])
    ev["status"] = ev["dep_status"].astype("string").where(ev["dep_delay"].notna(), ev["arr_status"].astype("string"))
    ev = ev.dropna(subset=["sched", "delay"])
    if measured_only:
        ev = ev[ev["status"] == "REAL"]
    lo, hi = window
    ev = ev[(ev["delay"] >= lo) & (ev["delay"] <= hi)]
    ev["hour"] = (ev["sched"] // 3600).astype("int16")
    ev["weekday"] = ev["day"].dt.weekday.astype("int8")
    holiday = ev["day"].dt.strftime("%Y-%m-%d").isin(HOLIDAYS)
    ev["daytype"] = np.select([holiday, ev.weekday <= 4, ev.weekday == 5], ["sunday", "weekday", "saturday"], "sunday")
    return ev.sort_values(["trip_key", "seq"]).reset_index(drop=True)


def stop_names(path=None):
    """Stop id to stop name, from data/raw/tl_stops.csv (written with the extract)."""
    path = path or RAW / "tl_stops.csv"
    if not Path(path).exists():
        return {}
    t = pd.read_csv(path, dtype=str)
    return dict(zip(t["stop"], t["name"]))
