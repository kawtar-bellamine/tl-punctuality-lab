"""README figures and banner from results/."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
S = json.loads((R / "summary.json").read_text())

RED, INK, MUTED, GREY, TEAL, BG, AMBER = "#C8102E", "#14213D", "#5B6675", "#B7C0CA", "#00909C", "#F6F7F9", "#E59500"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#C9D1D9",
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#EEF1F4", "axes.axisbelow": True})


def title(ax, t, sub=None):
    ax.set_title(t, loc="left", fontsize=13, fontweight="bold", color=INK, pad=22 if sub else 10)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=10, color=MUTED)


def hours():
    h = pd.read_csv(R / "punctuality_by_hour.csv")
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for dt, col, lab in (("weekday", RED, "weekdays"), ("saturday", TEAL, "Saturdays"), ("sunday", GREY, "Sundays and holiday")):
        d = h[(h.daytype == dt) & (h.events > 500) & (h.hour.between(5, 24))]
        ax.plot(d.hour, d.late_share * 100, color=col, lw=2.2, marker="o", ms=4, label=lab)
    ax.set_xlabel("scheduled hour")
    ax.set_ylabel("departures 3 min late or more (%)")
    ax.legend(frameon=False, loc="upper left")
    title(ax, "Lateness follows the evening rush", f"tl network, September 2026, {S['net_events']:,} measured departures")
    fig.tight_layout()
    fig.savefig(FIG / "lateness_by_hour.png", dpi=160)
    plt.close(fig)


def lines():
    b = pd.read_csv(R / "punctuality_by_line.csv", dtype={"line": str})
    b = b[b.events > 20000].sort_values("late_share")
    fig, ax = plt.subplots(figsize=(9, 6.2))
    cols = [RED if v >= 0.3 else (TEAL if v < 0.12 else GREY) for v in b.late_share]
    ax.barh(b.line, b.late_share * 100, color=cols, height=0.7)
    for i, v in enumerate(b.late_share * 100):
        ax.text(v + 0.4, i, f"{v:.0f} %", va="center", fontsize=8.5, color=INK)
    ax.set_xlabel("departures 3 min late or more (%)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=9)
    title(ax, "Some lines are late a third of the time, the metro almost never",
          "lines with more than 20,000 measured departures in the month")
    fig.tight_layout()
    fig.savefig(FIG / "lateness_by_line.png", dpi=160)
    plt.close(fig)


def prediction():
    p = pd.DataFrame(S["prediction"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(p))
    ax = axes[0]
    ax.bar(x - 0.2, p.mae_persistence_s, 0.4, color=GREY, label="current delay held constant")
    ax.bar(x + 0.2, p.mae_model_s, 0.4, color=RED, label="LightGBM")
    for i, r in p.iterrows():
        ax.text(i + 0.2, r.mae_model_s + 2, f"-{r.mae_reduction * 100:.0f} %", ha="center", fontsize=10, color=INK, fontweight="bold")
    ax.set_xticks(x, [f"{k} stop{'s' if k > 1 else ''}" for k in p.k])
    ax.set_ylabel("mean absolute error (s)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    title(ax, "Delay a few stops ahead", f"test week {S['test_from']} to {S['last_day']}, never seen in training")
    ax = axes[1]
    ax.plot(p.k, p.within_60s_persistence * 100, color=GREY, lw=2.2, marker="o", label="constant delay")
    ax.plot(p.k, p.within_60s_model * 100, color=RED, lw=2.2, marker="o", label="LightGBM")
    ax.set_xlabel("stops ahead")
    ax.set_ylabel("predictions within 1 minute (%)")
    ax.set_ylim(40, 100)
    ax.legend(frameon=False, fontsize=9)
    title(ax, "Within one minute of the real delay")
    fig.tight_layout()
    fig.savefig(FIG / "prediction.png", dpi=160)
    plt.close(fig)


def timetable():
    t = S["timetable"]
    props = pd.read_csv(R / "timetable_proposals.csv", dtype={"line": str})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    top = props.sort_values("median_gain_s", ascending=False).drop_duplicates(["line", "from_name", "to_name"]).head(10)[::-1]
    labels = [f"{r.line}  {r.from_name} → {r.to_name}" for r in top.itertuples()]
    ax.barh(labels, top.median_gain_s, color=GREY, height=0.65, label="time lost, median")
    ax.barh(labels, top.add_s, color=RED, height=0.65, label="time to add")
    ax.tick_params(axis="y", labelsize=8.5)
    ax.set_xlabel("seconds per trip")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(axis="y", visible=False)
    title(ax, "Where the timetable is too tight", "weekday segments losing time on most trips (training weeks)")
    ax = axes[1]
    cats = ["late", "on_time", "early"]
    names = ["late (≥ 3 min)", "on time", "early (> 1 min)"]
    x = np.arange(3)
    bv = [t["before"][c] * 100 for c in cats]
    av = [t["after"][c] * 100 for c in cats]
    ax.bar(x - 0.2, bv, 0.4, color=GREY, label="current timetable")
    ax.bar(x + 0.2, av, 0.4, color=RED, label="proposed timetable")
    for i in range(3):
        ax.text(i - 0.2, bv[i] + 1.5, f"{bv[i]:.1f}", ha="center", fontsize=9, color=MUTED)
        ax.text(i + 0.2, av[i] + 1.5, f"{av[i]:.1f}", ha="center", fontsize=9, color=RED, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_xticks(x, names)
    ax.set_ylabel("weekday departures, test week (%)")
    ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=1)
    title(ax, "Effect on the test week", f"{t['segments_changed']} segment and time-band changes")
    fig.tight_layout()
    fig.savefig(FIG / "timetable.png", dpi=160)
    plt.close(fig)


def growth():
    g = pd.read_csv(R / "delay_growth.csv")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(g.seq + 1, g.delay / 60, color=RED, lw=2.4)
    ax.fill_between(g.seq + 1, 0, g.delay / 60, color=RED, alpha=0.08)
    ax.set_xlabel("stop number along the trip")
    ax.set_ylabel("median delay (min)")
    title(ax, "Delay builds up stop after stop", "median over every tl trip of the month")
    fig.tight_layout()
    fig.savefig(FIG / "delay_growth.png", dpi=160)
    plt.close(fig)


def delay_map():
    t = pd.read_csv(R / "delay_created.csv", dtype={"from_station": str, "to_station": str})
    st = pd.read_csv(ROOT / "data" / "raw" / "tl_stations.csv", dtype={"station": str})
    top = t.nlargest(90, "created_min_per_day").iloc[::-1]
    vmax = top["created_min_per_day"].max()
    cmap = plt.get_cmap("YlOrRd")
    lab = t.nlargest(8, "created_min_per_day").reset_index(drop=True)
    centre = (6.618, 6.642, 46.5165, 46.5285)
    fig = plt.figure(figsize=(12, 7.6))
    ax = fig.add_axes([0.01, 0.02, 0.6, 0.86])
    ins = fig.add_axes([0.62, 0.36, 0.37, 0.52])

    def draw(a, lw_scale):
        for r in t.itertuples():
            a.plot([r.from_lon, r.to_lon], [r.from_lat, r.to_lat], color="#D5DBE1", lw=1, zorder=1, solid_capstyle="round")
        a.scatter(st.lon, st.lat, s=3, color="#9AA5B1", zorder=1.5)
        for r in top.itertuples():
            v = r.created_min_per_day / vmax
            a.plot([r.from_lon, r.to_lon], [r.from_lat, r.to_lat], color=cmap(0.25 + 0.75 * v), lw=(1.5 + 7 * v) * lw_scale,
                   zorder=2, solid_capstyle="round")
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
        a.set_aspect(1 / np.cos(np.radians(46.53)))

    draw(ax, 1)
    ax.set_xlim(6.54, 6.71); ax.set_ylim(46.498, 46.575)
    for sp in ax.spines.values():
        sp.set_visible(False)
    x0, x1, y0, y1 = centre
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=INK, lw=1, ls="--", zorder=4))
    draw(ins, 1.6)
    ins.set_xlim(x0, x1); ins.set_ylim(y0, y1)
    for sp in ins.spines.values():
        sp.set_edgecolor(INK); sp.set_linestyle("--")
    ins.set_title("city centre", loc="left", fontsize=10, color=MUTED)
    for i, r in lab.iterrows():
        x, y = (r.from_lon + r.to_lon) / 2, (r.from_lat + r.to_lat) / 2
        inside = x0 <= x <= x1 and y0 <= y <= y1
        a = ins if inside else ax
        ang = 2 * np.pi * i / len(lab)
        a.annotate(str(i + 1), (x, y), xytext=(22 * np.cos(ang), 22 * np.sin(ang)), textcoords="offset points",
                   fontsize=9.5, fontweight="bold", color=INK, ha="center", va="center", zorder=5,
                   bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec=RED, lw=1.2),
                   arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
        fig.text(0.625, 0.31 - i * 0.034, f"{i + 1}. {r.from_name} → {r.to_name}: {r.created_min_per_day:.0f} min/day",
                 fontsize=9.3, color=INK)
    fig.text(0.625, 0.31 + 0.034, "Segments creating the most delay", fontsize=10, color=INK, fontweight="bold")
    fig.text(0.01, 0.95, "Where tl buses lose time against the timetable", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.01, 0.915, f"delay created from one departure to the next, all lines, minutes per weekday; "
             f"thicker and redder = more minutes; {S['delay_created']['segments']} segments with a net loss",
             fontsize=10, color=MUTED)
    fig.savefig(FIG / "delay_map.png", dpi=160)
    plt.close(fig)


def regularity_fig():
    r = pd.read_csv(R / "regularity_by_line.csv", dtype={"line": str}).sort_values("ewt_min", ascending=False).head(14)
    al = pd.read_csv(R / "regularity_along.csv", dtype={"line": str})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 1.35]})
    ax = axes[0]
    ax.bar(r.line, r.ewt_min * 60, color=RED, width=0.65)
    for i, v in enumerate(r.ewt_min * 60):
        ax.text(i, v + 2, f"{v:.0f}", ha="center", fontsize=8.5, color=INK)
    ax.set_ylabel("extra wait per passenger (s)")
    ax.set_xlabel("line (every 10 min or more often)")
    title(ax, "Waiting longer than the timetable promises", "excess wait time, weekdays 6:00 to 21:00")
    ax = axes[1]
    line = r.sort_values("bunched_share", ascending=False).line.iloc[0]
    d = al[(al.line == line) & (al.direction == 0)].sort_values("position")
    ax.plot(range(len(d)), d.bunched_share * 100, color=AMBER, lw=2.4, marker="o", ms=3.5)
    ax.fill_between(range(len(d)), 0, d.bunched_share * 100, color=AMBER, alpha=0.12)
    step = max(1, len(d) // 9)
    ax.set_xticks(range(0, len(d), step), [n[:16] for n in d.name.iloc[::step]], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("buses bunched with the one ahead (%)")
    title(ax, f"Line {line}: regularity breaks down along the route", "bunched = actual headway under half the planned one")
    fig.tight_layout()
    fig.savefig(FIG / "regularity.png", dpi=160)
    plt.close(fig)


def connections_fig():
    k = S["connections"]
    hub = pd.read_csv(R / "connections_by_hub.csv").sort_values("missed_share")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 1]})
    ax = axes[0]
    ax.barh(hub["name"], hub.missed_share * 100, color=RED, height=0.65)
    for i, v in enumerate(hub.missed_share * 100):
        ax.text(v + 0.4, i, f"{v:.0f} %", va="center", fontsize=8.5, color=INK)
    ax.set_xlabel("planned connections missed (%)")
    ax.grid(axis="y", visible=False)
    title(ax, "Connections missed at the main hubs", f"{k['planned']:,} planned connections, weekdays, 2 min to change")
    ax = axes[1]
    hs = [0] + [h["hold_s"] for h in k["hold"]]
    ms = [k["missed_share"] * 100] + [h["missed_share_after"] * 100 for h in k["hold"]]
    ax.bar([f"{h} s" if h else "today" for h in hs], ms, color=[GREY] + [TEAL] * len(k["hold"]), width=0.6)
    for i, v in enumerate(ms):
        ax.text(i, v + 0.6, f"{v:.1f} %", ha="center", fontsize=10, color=INK, fontweight="bold")
    ax.set_ylabel("connections missed (%)")
    ax.set_xlabel("connecting bus may wait up to")
    ax.set_ylim(0, max(ms) * 1.25)
    title(ax, "If the connecting bus waits a little", "estimate on the same connections, without knock-on effects")
    fig.tight_layout()
    fig.savefig(FIG / "connections.png", dpi=160)
    plt.close(fig)


def banner():
    W, H = 1600, 560
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    F = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    f = lambda s, b=False: ImageFont.truetype(B if b else F, s)  # noqa: E731
    d.rectangle([0, 0, 18, H], fill=RED)
    x, y = 70, 60
    d.text((x, y), "PUBLIC TRANSPORT · DATA SCIENCE · LAUSANNE", font=f(22, True), fill=RED)
    d.text((x, y + 45), "tl punctuality lab", font=f(72, True), fill=INK)
    d.text((x, y + 145), "From the real departures of the Lausanne network\nto maps, predictions and operating actions.", font=f(28), fill=MUTED, spacing=8)
    stats = [(f"{S['raw_rows'] / 1e6:.1f} M", "real stop events, every tl\nbus and metro, Sept. 2026"),
             (f"{S['connections']['planned'] / 1e6:.1f} M", f"connections checked\nat {S['connections']['hubs']} hubs"),
             (f"-{S['mae_reduction_k5'] * 100:.0f} %", "error on the delay 5 stops\nahead, on an unseen week"),
             (f"{S['timetable']['segments_changed']}", "timetable changes,\ntested on that week")]
    for i, (big, small) in enumerate(stats):
        sx = x + i * 370
        d.text((sx, y + 260), big, font=f(60, True), fill=RED)
        d.text((sx, y + 340), small, font=f(21), fill=INK, spacing=5)
    d.text((x, H - 50), "Python · pandas · LightGBM · Streamlit · punctuality · regularity · connections · open data: opentransportdata.swiss",
           font=f(20), fill=MUTED)
    img.save(FIG / "banner.png")


if __name__ == "__main__":
    hours(); lines(); prediction(); timetable(); growth(); delay_map(); regularity_fig(); connections_fig(); banner()
    print("figures written")
