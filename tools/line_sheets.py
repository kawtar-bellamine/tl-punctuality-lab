"""One-page action sheet per line, in French, for the lines with the most late departures.

    python tools/line_sheets.py            # the 5 lines with the highest share of late departures
    python tools/line_sheets.py 9 18       # chosen lines

Writes docs/fiches/ligne_<line>.png and docs/fiches/fiches_lignes.pdf (every sheet, one per page).
"""
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "docs" / "fiches"
OUT.mkdir(parents=True, exist_ok=True)
S = json.loads((R / "summary.json").read_text())
RED, INK, MUTED, GREY, TEAL, AMBER, SOFT = "#C8102E", "#14213D", "#5B6675", "#B7C0CA", "#00909C", "#E59500", "#F5F7FA"
BAND = {"early": "avant 7 h", "am_peak": "pointe du matin", "day": "journée", "pm_peak": "pointe du soir", "evening": "soirée"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#C9D1D9", "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK})

rd = lambda n: pd.read_csv(R / n, dtype={"line": str, "to_line": str, "station": str})  # noqa: E731
by_line, lh, cl, props = rd("punctuality_by_line.csv"), rd("punctuality_by_line_hour.csv"), rd("delay_created_by_line.csv"), rd("timetable_proposals.csv")
reg, al, cp = rd("regularity_by_line.csv"), rd("regularity_along.csv"), rd("connections_pairs.csv")
pats = json.loads((R / "line_patterns.json").read_text())
fr = lambda v, d=1: f"{v:.{d}f}".replace(".", ",")  # noqa: E731


def sheet(line):
    L = by_line[by_line.line == line].iloc[0]
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    fig.add_artist(plt.Rectangle((0, 0.9), 1, 0.1, color=INK, transform=fig.transFigure))
    fig.text(0.035, 0.945, f"Ligne {line}", fontsize=26, fontweight="bold", color="white", va="center")
    ends = " · ".join(dict.fromkeys(f"vers {p['names'][-1]}" for p in pats.get(line, [])))
    fig.text(0.215, 0.955, "Fiche d'actions ponctualité", fontsize=13, color="white", va="center", fontweight="bold")
    fig.text(0.215, 0.928, f"{ends}   |   données réelles tl, {S['first_day']} au {S['last_day']}", fontsize=9.5,
             color="#C9D1D9", va="center")
    R_ = reg[reg.line == line]
    kpis = [(f"{fr(L.on_time_share * 100)} %", "à l'heure"),
            (f"{fr(L.late_share * 100)} %", f"en retard (réseau {fr(S['net_late_share'] * 100)} %)"),
            (f"{L.median_delay_s:.0f} s", "retard médian"),
            (f"{L.events:,.0f}".replace(",", " "), "passages mesurés"),
            ((f"{R_.ewt_min.iloc[0] * 60:.0f} s" if len(R_) else "-"), "attente en plus (lignes fréquentes)")]
    for i, (big, small) in enumerate(kpis):
        x = 0.035 + i * 0.19
        fig.add_artist(plt.Rectangle((x, 0.775), 0.175, 0.095, color=SOFT, transform=fig.transFigure))
        fig.text(x + 0.012, 0.835, big, fontsize=19, fontweight="bold", color=RED)
        fig.text(x + 0.012, 0.79, small, fontsize=8.8, color=INK)

    ax = fig.add_axes([0.06, 0.43, 0.38, 0.28])
    d = lh[(lh.line == line) & (lh.daytype == "weekday") & (lh.events > 100) & lh.hour.between(5, 24)]
    ax.bar(d.hour, d.late_share * 100, color=[RED if v >= 0.3 else GREY for v in d.late_share], width=0.75)
    ax.axhline(S["net_late_share"] * 100, color=INK, ls=":", lw=1)
    ax.text(d.hour.min() - 0.4, S["net_late_share"] * 100 + 1.2, "moyenne réseau", fontsize=8, color=INK,
            bbox=dict(fc="white", ec="none", pad=1), zorder=5)
    ax.set_xticks(range(int(d.hour.min()), int(d.hour.max()) + 1, 2))
    ax.set_xlabel("heure, jours de semaine")
    ax.set_ylabel("% de départs en retard")
    ax.set_title("Quand la ligne est en retard", loc="left", fontweight="bold", color=INK, fontsize=11)

    ax = fig.add_axes([0.6, 0.43, 0.36, 0.28])
    c = cl[cl.line == line].head(7).iloc[::-1]
    ax.barh([textwrap.shorten(f"{a} → {b}", 34, placeholder="…") for a, b in zip(c.from_name, c.to_name)],
            c.created_min_per_day, color=AMBER, height=0.6)
    ax.set_xlabel("retard créé (minutes par jour de semaine)")
    ax.tick_params(axis="y", labelsize=8.5)
    ax.set_title("Où elle perd du temps", loc="left", fontweight="bold", color=INK, fontsize=11)

    ax = fig.add_axes([0.06, 0.12, 0.38, 0.22])
    a = al[(al.line == line) & (al.direction == 0)].sort_values("position") if len(al) else al
    if len(a) and a.bunched_share.notna().sum() > 3:
        ax.plot(range(len(a)), a.bunched_share * 100, color=AMBER, lw=2.2, marker="o", ms=3)
        step = max(1, len(a) // 7)
        ax.set_xticks(range(0, len(a), step), [textwrap.shorten(n, 14, placeholder="…") for n in a.name.iloc[::step]],
                      rotation=30, ha="right", fontsize=7.5)
        ax.set_ylabel("% de bus en peloton")
        ax.set_ylim(0, max(5, a.bunched_share.max() * 100 * 1.2))
        ax.set_title(f"Régularité le long de la ligne, vers {pats[line][0]['names'][-1]}", loc="left", fontweight="bold",
                     color=INK, fontsize=11)
    else:
        ax.axis("off")
        ax.text(0, 0.5, "Ligne à moins d'un bus toutes les 10 minutes :\nla ponctualité à l'horaire compte plus que l'intervalle.",
                fontsize=10, color=MUTED)

    acts = []
    p = props[props.line == line].sort_values("median_gain_s", ascending=False)
    if len(p):
        x = p.iloc[0]
        acts.append(f"Temps de parcours : {len(p)} ajustements proposés, dont {x.from_name} → {x.to_name} "
                    f"({BAND.get(x.band, x.band)}) : {x.median_gain_s:.0f} s perdues en médiane, {x.add_s:.0f} s à ajouter.")
    if len(c):
        x = c.iloc[-1]
        acts.append(f"Priorité bus : {x.from_name} → {x.to_name} crée {x.created_min_per_day:.0f} min de retard par jour ; "
                    "feux, voie réservée ou arrêt à examiner sur place.")
    if len(a) and a.bunched_share.notna().sum() > 3:
        x = a.loc[a.bunched_share.idxmax()]
        acts.append(f"Régularité : jusqu'à {fr(x.bunched_share * 100)} % de bus en peloton à {x['name']} ; "
                    "régulation à l'intervalle à placer en amont.")
    q = cp[(cp.to_line == line) & (cp.connections >= 300)].sort_values("missed_share", ascending=False)
    if len(q):
        x = q.iloc[0]
        acts.append(f"Correspondance : à {x['name']}, {fr(x.missed_share * 100, 0)} % des correspondances depuis la ligne "
                    f"{x.line} manquées ; attente courte du bus à évaluer.")
    fig.text(0.6, 0.33, "Actions à étudier en premier", fontsize=11, fontweight="bold", color=INK)
    y = 0.295
    for i, t in enumerate(acts, 1):
        lines = textwrap.wrap(t, 62)
        fig.text(0.6, y, f"{i}.", fontsize=9.5, color=RED, fontweight="bold", va="top")
        fig.text(0.62, y, "\n".join(lines), fontsize=9.2, color=INK, va="top", linespacing=1.35)
        y -= 0.022 * len(lines) + 0.022
    fig.text(0.035, 0.012, "tl punctuality lab · Abderrahmane Boulami et Kawtar Bellamine · données opentransportdata.swiss",
             fontsize=7.5, color=MUTED)
    return fig


if __name__ == "__main__":
    chosen = sys.argv[1:] or [d["line"] for d in S["worst_lines_late"]]
    with PdfPages(OUT / "fiches_lignes.pdf") as pdf:
        for ln in chosen:
            f = sheet(str(ln))
            f.savefig(OUT / f"ligne_{ln}.png", dpi=130)
            pdf.savefig(f)
            plt.close(f)
    print("sheets:", ", ".join(map(str, chosen)))
