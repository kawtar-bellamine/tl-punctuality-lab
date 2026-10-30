"""Full study on the tl extract: punctuality, where delay is created, regularity, connections, delay prediction
and timetable proposals.

    python experiments/run_all.py            # whole extract in data/raw
    python experiments/run_all.py --sample   # committed sample (CI)

Training uses the first days of the month, every score is computed on the last days, never seen in training.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tlpunct import connections, data, geo, kpi, predict, regularity, runtime  # noqa: E402

RESULTS = ROOT / "results"


def split_days(df, test_days):
    days = sorted(df["day"].unique())
    cut = days[-test_days]
    return df[df["day"] < cut], df[df["day"] >= cut], cut


def main(sample):
    t0 = time.time()
    out = RESULTS / "sample" if sample else RESULTS
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted((data.RAW / "sample").glob("*.csv.gz")) if sample else None
    raw = data.read_extract(paths)
    ev = data.stop_events(raw)
    names = data.stop_names()
    stations_names = {}
    for k, v in names.items():
        stations_names.setdefault(k[:7], v)

    # ---- punctuality
    net = kpi.network(ev)
    by_line = kpi.summarise(ev, ["line"]).sort_values("events", ascending=False)
    by_hour = kpi.summarise(ev, ["daytype", "hour"])
    by_stop = kpi.summarise(ev, ["stop"])
    by_stop["name"] = by_stop["stop"].map(names)
    growth = kpi.delay_growth(ev)
    for name, t in (("punctuality_by_line", by_line), ("punctuality_by_hour", by_hour),
                    ("punctuality_by_stop", by_stop), ("delay_growth", growth)):
        t.to_csv(out / f"{name}.csv", index=False)

    # ---- prediction a few stops ahead
    test_days = 1 if sample else 7
    ev_train, ev_test, cut = split_days(ev, test_days)
    P_train = predict.pairs(ev_train, horizons=predict.TRAIN_HORIZONS, sample=400_000 if sample else 3_000_000, seed=1)
    P_test = predict.pairs(ev_test, sample=150_000 if sample else 1_000_000, seed=2)
    model, cats = predict.fit(P_train)
    pred = predict.predict(model, cats, P_test)
    sc = predict.scores(P_test, pred)
    sc.to_csv(out / "prediction_scores.csv", index=False)
    # quantile models on the first training days, conformal margins on the last ones, scores on the test week
    cal_cut = sorted(ev_train["day"].unique())[-max(1, test_days // 2)] if not sample else None
    if cal_cut is not None:
        q_fit, q_cal = P_train[P_train["day"] < cal_cut], P_train[P_train["day"] >= cal_cut]
    else:
        q_cal = P_train.sample(frac=0.25, random_state=4)
        q_fit = P_train.drop(q_cal.index)
    qs = predict.fit_quantiles(q_fit.sample(min(len(q_fit), 1_000_000), random_state=3))
    margins = predict.calibrate(qs, cats, q_cal)
    lo0, hi0 = predict.band(qs, cats, P_test)
    lo, hi = predict.band(qs, cats, P_test, margins)
    iv = predict.interval_scores(P_test, lo, hi).merge(
        predict.interval_scores(P_test, lo0, hi0).rename(columns={"coverage": "coverage_raw", "mean_width_s": "mean_width_raw_s"}),
        on="k")
    iv.to_csv(out / "prediction_intervals.csv", index=False)
    Xt = None
    predict.save(out / "model", model, cats, qs)
    predict.save_margins(out / "model", margins)
    imp = pd.Series(model.booster_.feature_importance("gain"), index=predict.FEATURES)
    (imp / imp.sum()).sort_values(ascending=False).rename("gain_share").to_csv(out / "prediction_importance.csv")
    n_train, n_test = len(P_train), len(P_test)
    del P_train, P_test, Xt, pred, lo, hi, lo0, hi0, q_fit, q_cal, ev_train, ev_test
    print("prediction done", round(time.time() - t0), "s", flush=True)

    # ---- running times and timetable proposals
    raw_train, raw_test, _ = split_days(raw[raw.cancelled == 0], test_days)
    seg_train = runtime.segments(raw_train)
    changes = runtime.propose(seg_train)
    changes["from_name"] = changes["from_stop"].map(names)
    changes["to_name"] = changes["to_stop"].map(names)
    changes.sort_values("median_gain_s", ascending=False).to_csv(out / "timetable_proposals.csv", index=False)
    ev_res = runtime.evaluate(raw_test, changes)
    seg_all = runtime.segments(raw)
    hot = (seg_all[seg_all.weekday].groupby(["line", "from_stop", "to_stop", "band"], observed=True)["gain_s"]
           .agg(["size", "median"]).reset_index().query("size >= 20").sort_values("median", ascending=False))
    hot["from_name"] = hot["from_stop"].map(names)
    hot["to_name"] = hot["to_stop"].map(names)
    hot.head(200).to_csv(out / "segment_hotspots.csv", index=False)

    # ---- where delay is created (map)
    lg = geo.legs(ev)
    created = geo.with_coords(geo.delay_created(lg))
    created_pm = geo.with_coords(geo.delay_created(lg, bands=["pm_peak"]))
    for t in (created, created_pm):
        t["from_name"] = t["from_station"].map(stations_names)
        t["to_name"] = t["to_station"].map(stations_names)
    created.to_csv(out / "delay_created.csv", index=False)
    per_line = []
    for line, sl in lg.groupby("line", observed=True):
        t = geo.with_coords(geo.delay_created(sl)).head(15)
        per_line.append(t.assign(line=str(line)))
    per_line = pd.concat(per_line)
    per_line["from_name"] = per_line["from_station"].map(stations_names)
    per_line["to_name"] = per_line["to_station"].map(stations_names)
    per_line.to_csv(out / "delay_created_by_line.csv", index=False)
    kpi.summarise(ev, ["line", "daytype", "hour"]).to_csv(out / "punctuality_by_line_hour.csv", index=False)
    created_pm.to_csv(out / "delay_created_pm_peak.csv", index=False)

    top10 = created.head(10)["created_min_per_day"].sum() / max(created["created_min_per_day"].sum(), 1e-9)

    # ---- headway regularity on frequent lines
    h = regularity.headways(raw)
    reg_line = regularity.summary(h, ["line"]).query("headways >= 500").sort_values("ewt_min", ascending=False)
    reg_line.to_csv(out / "regularity_by_line.csv", index=False)
    regularity.summary(h, ["hour"]).to_csv(out / "regularity_by_hour.csv", index=False)
    reg_net = regularity.summary(h.assign(all=1), ["all"]).iloc[0]

    # ---- main stop patterns of each line (both directions) and regularity along them
    patterns, along = {}, []
    for line in sorted(raw["line"].dropna().unique(), key=lambda v: (len(str(v)), str(v))):
        d = raw[(raw.line == line) & (raw.cancelled == 0)].sort_values(["trip_key", "seq"])
        tk = d.groupby("trip_key", observed=True)["stop"].agg(lambda x: ">".join(map(str, x)))
        seqs = tk.value_counts()
        pats, ends = [], set()
        for pat, n in seqs.items():
            pat = pat.split(">")
            if len(pat) < 4 or pat[-1][:7] in ends:     # one pattern per terminus: the two directions
                continue
            ends.add(pat[-1][:7])
            sel = d[d["trip_key"].isin(tk.index[tk == ">".join(pat)])]
            t_ = sel["sched_dep"].fillna(sel["sched_arr"])
            off = (t_ - t_.groupby(sel["trip_key"]).transform("min")).groupby(sel.groupby("trip_key").cumcount()).median()
            pats.append({"stops": list(pat), "names": [names.get(x, x) for x in pat], "trips": int(n),
                         "minutes": [round(float(v) / 60, 2) for v in off.reindex(range(len(pat))).ffill().fillna(0)]})
            if len(pats) == 2:
                break
        patterns[str(line)] = pats
        if str(line) in set(reg_line["line"].astype(str)):
            for i, ptn in enumerate(pats):
                a = regularity.along(h, ptn["stops"])
                along.append(a.assign(line=str(line), direction=i, name=a["stop"].map(names)))
    (out / "line_patterns.json").write_text(json.dumps(patterns, ensure_ascii=False))
    if along:
        pd.concat(along).to_csv(out / "regularity_along.csv", index=False)
    print("regularity done", round(time.time() - t0), "s", flush=True)

    # ---- connections at the main hubs
    hub_ids = connections.hubs(ev, n=12)
    C = connections.connections(raw, hub_ids)
    con_net = connections.summary(C)
    by_hub = connections.summary(C, ["station"])
    by_hub["name"] = by_hub["station"].map(stations_names)
    by_hub.sort_values("missed_share", ascending=False).to_csv(out / "connections_by_hub.csv", index=False)
    connections.summary(C, ["hour"]).to_csv(out / "connections_by_hour.csv", index=False)
    pairs_ = connections.summary(C, ["station", "line", "to_line"])
    pairs_ = pairs_[pairs_.connections >= 200].copy()
    pairs_["name"] = pairs_["station"].map(stations_names)
    pairs_.sort_values("missed_share", ascending=False).to_csv(out / "connections_pairs.csv", index=False)
    holds = [connections.hold(C, s) for s in (30, 60, 90)]
    walk_sens = {str(w): float(connections.connections(raw, hub_ids, walk_s=w)["missed"].mean()) for w in (60, 180)}
    walk_sens["120"] = float(C["missed"].mean())
    print("connections done", round(time.time() - t0), "s", flush=True)

    days = sorted(raw["day"].dt.strftime("%Y-%m-%d").unique())
    s = sc.set_index("k")
    ivk = iv.set_index("k")
    summary = {
        "days": len(days), "first_day": days[0], "last_day": days[-1], "test_from": str(pd.Timestamp(cut).date()),
        "raw_rows": int(len(raw)), "cancelled_share": float(raw["cancelled"].mean()),
        "measured_share": float((raw["dep_status"].astype("string").fillna(raw["arr_status"].astype("string")) == "REAL").mean()),
        **{f"net_{k}": v for k, v in net.items()},
        "worst_lines_late": by_line[by_line.events > 2000].nlargest(5, "late_share")[["line", "late_share", "events"]].to_dict("records"),
        "best_lines_on_time": by_line[by_line.events > 2000].nlargest(5, "on_time_share")[["line", "on_time_share", "events"]].to_dict("records"),
        "peak_hour_late_share": float(by_hour[(by_hour.daytype == "weekday") & by_hour.hour.isin([17, 18])]
                                      .eval("late_share * events").sum()
                                      / by_hour[(by_hour.daytype == "weekday") & by_hour.hour.isin([17, 18])].events.sum()),
        "prediction": sc.to_dict("records"),
        "prediction_intervals": iv.to_dict("records"),
        "interval_coverage_k5": float(ivk.loc[5, "coverage"]) if 5 in ivk.index else None,
        "mae_reduction_k5": float(s.loc[5, "mae_reduction"]) if 5 in s.index else None,
        "mae_reduction_k10": float(s.loc[10, "mae_reduction"]) if 10 in s.index else None,
        "train_pairs": int(n_train), "test_pairs": int(n_test),
        "timetable": ev_res,
        "delay_created": {"segments": int(len(created)), "min_per_weekday": float(created["created_min_per_day"].sum()),
                          "top10_share": float(top10),
                          "top": created.head(5)[["from_name", "to_name", "lines", "created_min_per_day", "median_gain_s"]].to_dict("records")},
        "regularity": {"headways": int(reg_net["headways"]), "bunched_share": float(reg_net["bunched_share"]),
                       "gap_share": float(reg_net["gap_share"]), "ewt_min": float(reg_net["ewt_min"]),
                       "swt_min": float(reg_net["swt_min"]), "awt_min": float(reg_net["awt_min"]),
                       "worst_lines": reg_line.head(5)[["line", "bunched_share", "ewt_min", "sched_headway_min"]].to_dict("records")},
        "connections": {"hubs": int(len(hub_ids)), "planned": int(len(C)), **{k: float(v) for k, v in con_net.items()},
                        "hold": holds, "missed_share_by_walk_s": walk_sens},
        "runtime_s": round(time.time() - t0, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    main(ap.parse_args().sample)
