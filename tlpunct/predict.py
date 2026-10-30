"""Predicting the delay a few stops ahead, from the delay measured now.

Passenger information usually assumes the current delay stays the same (persistence). Delays do not stay the
same: they grow on some segments and hours and shrink where the timetable has slack. The model learns, from
past weeks, how the delay of a bus at one stop turns into its delay k stops later, and is judged on later
days it never saw.
"""
import numpy as np
import pandas as pd

HORIZONS = (1, 3, 5, 10)               # horizons reported
TRAIN_HORIZONS = tuple(range(1, 11))   # horizons learnt (the dashboard simulator uses every one)
QUANTILES = (0.1, 0.9)                 # 80 % prediction interval
FEATURES = ["delay_now", "trend_2", "k", "sched_gap_min", "hour", "weekday", "seq", "line_c", "stop_c", "target_c"]
CATS = ["line_c", "stop_c", "target_c"]


def pairs(ev, horizons=HORIZONS, sample=None, seed=0):
    """(stop now, stop k ahead) pairs on the same trip, with the features known at the stop now."""
    ev = ev.sort_values(["trip_key", "seq"])
    g = ev.groupby("trip_key")
    prev2 = g["delay"].shift(2)
    base = pd.DataFrame({
        "trip_key": ev["trip_key"].to_numpy(), "day": ev["day"].to_numpy(), "line": ev["line"].to_numpy(),
        "stop": ev["stop"].to_numpy(), "seq": ev["seq"].to_numpy(), "hour": ev["hour"].to_numpy(),
        "weekday": ev["weekday"].to_numpy(), "sched": ev["sched"].to_numpy(),
        "delay_now": ev["delay"].to_numpy(), "trend_2": (ev["delay"] - prev2).to_numpy(),
    })
    frames = []
    for k in horizons:
        fut = g[["stop", "sched", "delay"]].shift(-k)
        f = base.copy()
        f["k"] = k
        f["target_stop"] = fut["stop"].to_numpy()
        f["sched_gap_min"] = (fut["sched"].to_numpy() - f["sched"]) / 60.0
        f["delay_future"] = fut["delay"].to_numpy()
        f = f.dropna(subset=["delay_future", "target_stop"])
        f = f[(f["sched_gap_min"] > 0) & (f["sched_gap_min"] < 90)]
        per_k = sample // len(horizons) if sample else None
        if per_k and len(f) > per_k:          # sample each horizon as it is built, to keep memory low
            f = f.sample(per_k, random_state=seed + k)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def encode(P, categories=None):
    X = P.copy()
    cats = categories or {}
    for col, src in (("line_c", "line"), ("stop_c", "stop"), ("target_c", "target_stop")):
        cats.setdefault(col, pd.Index(P[src].astype(str).unique()))
        X[col] = pd.Categorical(X[src].astype(str), categories=cats[col])
    X["trend_2"] = X["trend_2"].astype("float32")
    return X[FEATURES], cats


def fit(P_train, seed=0):
    import lightgbm as lgb
    X, cats = encode(P_train)
    y = P_train["delay_future"] - P_train["delay_now"]          # learn the change, not the level
    model = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=127, min_child_samples=50,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.9, objective="l1",
                              random_state=seed, verbose=-1)
    model.fit(X, y, categorical_feature=CATS)
    return model, cats


def fit_quantiles(P_train, alphas=QUANTILES, seed=0):
    """One smaller LightGBM per quantile of the delay change: the band the delay will fall in."""
    import lightgbm as lgb
    X, cats = encode(P_train)
    y = P_train["delay_future"] - P_train["delay_now"]
    models = {}
    for a in alphas:
        m = lgb.LGBMRegressor(n_estimators=250, learning_rate=0.08, num_leaves=63, min_child_samples=100,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.9, objective="quantile",
                              alpha=a, random_state=seed, verbose=-1)
        m.fit(X, y, categorical_feature=CATS)
        models[a] = m
    return models


def calibrate(models, cats, P_cal, target=0.8):
    """Split-conformal correction (conformalized quantile regression): by horizon, the margin to add on both
    sides so that `target` of a calibration set, kept out of the quantile training, falls inside the band."""
    X, _ = encode(P_cal, cats)
    lo = P_cal["delay_now"].to_numpy() + models[QUANTILES[0]].predict(X)
    hi = P_cal["delay_now"].to_numpy() + models[QUANTILES[1]].predict(X)
    y = P_cal["delay_future"].to_numpy()
    score = np.maximum(lo - y, y - hi)
    out = {}
    for k in sorted(P_cal["k"].unique()):
        m = (P_cal["k"] == k).to_numpy()
        n = int(m.sum())
        out[int(k)] = float(np.quantile(score[m], min(1.0, np.ceil((n + 1) * target) / n)))
    return out


def band(models, cats, P, margins=None):
    """P10 and P90 of the future delay, widened by the conformal margins when given."""
    X, _ = encode(P, cats)
    now = P["delay_now"].to_numpy()
    lo, hi = now + models[QUANTILES[0]].predict(X), now + models[QUANTILES[1]].predict(X)
    if margins:
        mg = P["k"].map(lambda k: margins.get(int(k), margins[max(margins)])).to_numpy()
        lo, hi = lo - mg, hi + mg
    return lo, hi


def interval_scores(P, lo, hi):
    """Coverage and mean width of the interval [lo, hi] by horizon."""
    y = P["delay_future"].to_numpy()
    out = []
    for k in sorted(P["k"].unique()):
        m = (P["k"] == k).to_numpy()
        out.append({"k": int(k), "coverage": float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m]))),
                    "mean_width_s": float(np.mean(hi[m] - lo[m]))})
    return pd.DataFrame(out)


def predict(model, cats, P):
    X, _ = encode(P, cats)
    return P["delay_now"].to_numpy() + model.predict(X)


def scores(P, pred):
    err = pred - P["delay_future"].to_numpy()
    base = P["delay_now"].to_numpy() - P["delay_future"].to_numpy()
    out = []
    for k in sorted(P["k"].unique()):
        m = (P["k"] == k).to_numpy()
        out.append({"k": int(k), "pairs": int(m.sum()),
                    "mae_persistence_s": float(np.mean(np.abs(base[m]))),
                    "mae_model_s": float(np.mean(np.abs(err[m]))),
                    "within_60s_persistence": float(np.mean(np.abs(base[m]) <= 60)),
                    "within_60s_model": float(np.mean(np.abs(err[m]) <= 60))})
    df = pd.DataFrame(out)
    df["mae_reduction"] = 1 - df["mae_model_s"] / df["mae_persistence_s"]
    return df


def save(folder, model, cats, quantiles=None):
    """Models as gzipped LightGBM text files (small enough to version) and their categories."""
    import gzip
    import json
    from pathlib import Path
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    boosters = {"median": model, **{f"q{int(a * 100):02d}": m for a, m in (quantiles or {}).items()}}
    for name, m in boosters.items():
        with gzip.open(folder / f"delay_{name}.txt.gz", "wt") as f:
            f.write(m.booster_.model_to_string())
    (folder / "delay_model_categories.json").write_text(json.dumps({k: list(map(str, v)) for k, v in cats.items()}))


def save_margins(folder, margins):
    import json
    from pathlib import Path
    (Path(folder) / "interval_margins.json").write_text(json.dumps({str(k): v for k, v in margins.items()}))


def load_margins(folder):
    import json
    from pathlib import Path
    p = Path(folder) / "interval_margins.json"
    return {int(k): v for k, v in json.loads(p.read_text()).items()} if p.exists() else {}


def load(folder):
    """Models saved by `save`: {"median": m, "q10": m, "q90": m} with .predict(X), and the categories."""
    import gzip
    import json
    from pathlib import Path

    import lightgbm as lgb
    folder = Path(folder)
    cats = {k: pd.Index(v) for k, v in json.loads((folder / "delay_model_categories.json").read_text()).items()}

    class _M:
        def __init__(self, b):
            self.b = b

        def predict(self, X):
            return self.b.predict(X)
    models = {}
    for f in sorted(folder.glob("delay_*.txt.gz")):
        with gzip.open(f, "rt") as fh:
            models[f.name[len("delay_"):-len(".txt.gz")]] = _M(lgb.Booster(model_str=fh.read()))
    return models, cats
