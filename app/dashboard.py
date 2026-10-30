"""Ponctualité tl : du constat aux actions. Tableau de bord sur les données réelles du réseau lausannois."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tlpunct import predict  # noqa: E402

RED, INK, TEAL, SOFT, GREY, AMBER = "#C8102E", "#14213D", "#00909C", "#F5F7FA", "#8A96A3", "#E59500"
BAND_FR = {"early": "avant 7 h", "am_peak": "pointe du matin", "day": "journée", "pm_peak": "pointe du soir",
           "evening": "soirée"}

st.set_page_config(page_title="Ponctualité tl", page_icon="🚌", layout="wide")
st.markdown(f"""<style>
.block-container {{padding-top: 1.4rem; max-width: 1280px;}}
h1, h2, h3 {{color: {INK};}}
div[data-testid="stMetric"] {{background: {SOFT}; border-radius: 12px; padding: 10px 14px;}}
div[data-testid="stMetricValue"] {{color: {RED};}}
.kicker {{color: {RED}; font-weight: 700; letter-spacing: .08em; font-size: .78rem; text-transform: uppercase;}}
.action {{border-left: 4px solid {RED}; background: {SOFT}; padding: 10px 14px; border-radius: 6px; margin: 6px 0;}}
</style>""", unsafe_allow_html=True)


def results_dir():
    full = ROOT / "results"
    return full if (full / "summary.json").exists() else full / "sample"


FOLDER = results_dir()
STR = {"stop": str, "from_stop": str, "to_stop": str, "line": str, "to_line": str, "station": str,
       "from_station": str, "to_station": str}


@st.cache_data
def csv(name):
    p = FOLDER / name
    return pd.read_csv(p, dtype=STR) if p.exists() else pd.DataFrame()


@st.cache_data
def summary():
    return json.loads((FOLDER / "summary.json").read_text())


@st.cache_data
def patterns():
    p = FOLDER / "line_patterns.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_resource
def models():
    try:
        m, c = predict.load(FOLDER / "model")
        return m, c, predict.load_margins(FOLDER / "model")
    except Exception:   # noqa: BLE001
        return {}, {}, {}


S = summary()
pct = lambda v: f"{v * 100:.1f} %".replace(".", ",")  # noqa: E731
num = lambda v: f"{v:,.0f}".replace(",", " ")  # noqa: E731
line_key = lambda v: (len(str(v)), str(v))  # noqa: E731
layout = dict(margin=dict(t=50, l=10, r=10), plot_bgcolor="white")


def action(text):
    st.markdown(f'<div class="action">{text}</div>', unsafe_allow_html=True)


st.markdown('<div class="kicker">Transports publics · Data science · Lausanne</div>', unsafe_allow_html=True)
st.title("Ponctualité tl : du constat aux actions")
st.caption(f"Données réelles des tl (opentransportdata.swiss, Ist-Daten), du {S['first_day']} au {S['last_day']} : "
           f"{num(S['net_events'])} passages mesurés à {num(S['net_stops'])} quais sur {S['net_lines']} lignes. "
           "En retard = 3 minutes ou plus ; en avance = départ plus d'une minute avant l'horaire.")

TABS = ["Réseau", "Carte du retard", "Régularité", "Correspondances", "Prédire le retard", "Ajuster les horaires",
        "Fiche ligne"]
t_net, t_map, t_reg, t_con, t_pred, t_tt, t_line = st.tabs(TABS)

# ---------------------------------------------------------------- réseau
with t_net:
    c = st.columns(4)
    c[0].metric("À l'heure", pct(S["net_on_time_share"]))
    c[1].metric("En retard (≥ 3 min)", pct(S["net_late_share"]))
    c[2].metric("En avance (> 1 min)", pct(S["net_early_share"]))
    c[3].metric("Retard en pointe du soir", pct(S["peak_hour_late_share"]), help="Semaine, 17 h à 19 h")
    a, b = st.columns([1.15, 1])
    by_hour = csv("punctuality_by_hour.csv")
    fig = go.Figure()
    for dt, col, name in (("weekday", RED, "semaine"), ("saturday", TEAL, "samedi"), ("sunday", GREY, "dimanche")):
        d = by_hour[(by_hour.daytype == dt) & (by_hour.events > 2000) & by_hour.hour.between(5, 24)]
        fig.add_scatter(x=d.hour, y=d.late_share * 100, mode="lines+markers", name=name, line=dict(color=col, width=2))
    fig.update_layout(title="Part des passages en retard par heure", xaxis_title="heure", yaxis_title="% en retard",
                      height=380, legend=dict(orientation="h", y=-0.2), **layout)
    a.plotly_chart(fig, width="stretch")
    growth = csv("delay_growth.csv")
    g = px.line(growth, x="seq", y=growth["delay"] / 60, labels={"seq": "rang de l'arrêt dans la course", "y": "retard médian (min)"})
    g.update_traces(line_color=RED)
    g.update_layout(title="Le retard se construit le long de la course", height=380, **layout)
    b.plotly_chart(g, width="stretch")
    by_line = csv("punctuality_by_line.csv")
    lines = by_line[by_line.events > 1000].sort_values("late_share", ascending=False)
    fig = px.bar(lines, x="line", y=lines.late_share * 100, color=lines.late_share * 100, color_continuous_scale="Reds",
                 labels={"line": "ligne", "y": "% en retard", "color": "% en retard"})
    fig.update_layout(title="Part des passages en retard par ligne", height=360, coloraxis_showscale=False,
                      xaxis_type="category", **layout)
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- carte
with t_map:
    dc = S.get("delay_created", {})
    st.write("Chaque trait relie deux arrêts consécutifs. Son épaisseur et sa couleur montrent le retard que les bus y "
             "accumulent au-delà de l'horaire (trajet et arrêt suivant), cumulé sur toutes les courses d'un jour de "
             "semaine : c'est là que des mesures d'exploitation (priorité aux feux, voie bus, temps de parcours, "
             "montée des voyageurs) rapportent le plus.")
    if dc:
        c = st.columns(3)
        c[0].metric("Retard créé par jour de semaine", f"{num(dc['min_per_weekday'] / 60)} h")
        c[1].metric("Part des 10 segments les plus coûteux", pct(dc["top10_share"]))
        c[2].metric("Segments mesurés", num(dc["segments"]))
    period = st.radio("Période", ["Toute la journée", "Pointe du soir (16 h à 19 h)"], horizontal=True)
    mp = csv("delay_created.csv" if period.startswith("Toute") else "delay_created_pm_peak.csv")
    if not mp.empty:
        top = mp.nlargest(400, "created_min_per_day")
        vmax = float(top["created_min_per_day"].quantile(0.98))
        fig = go.Figure()
        for _, r in top.iloc[::-1].iterrows():
            v = min(r.created_min_per_day / vmax, 1.0)
            fig.add_trace(go.Scattermap(lat=[r.from_lat, r.to_lat], lon=[r.from_lon, r.to_lon], mode="lines",
                                        line=dict(width=1.5 + 9 * v, color=px.colors.sample_colorscale("YlOrRd", [0.15 + 0.85 * v])[0]),
                                        hoverinfo="text", showlegend=False,
                                        text=f"{r.from_name} → {r.to_name}<br>lignes {r.lines}<br>"
                                             f"{r.created_min_per_day:.0f} min de retard créées par jour"))
        fig.update_layout(map=dict(style="carto-positron", center=dict(lat=46.535, lon=6.625), zoom=11.3),
                          height=620, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig, width="stretch")
        show = mp.head(15)[["from_name", "to_name", "lines", "trips_per_day", "median_gain_s", "created_min_per_day"]]
        st.dataframe(show.round(1).rename(columns={"from_name": "de", "to_name": "à", "lines": "lignes",
                                                   "trips_per_day": "courses par jour", "median_gain_s": "temps perdu médian (s)",
                                                   "created_min_per_day": "retard créé (min par jour)"}),
                     hide_index=True, width="stretch")

# ---------------------------------------------------------------- régularité
with t_reg:
    R = S.get("regularity")
    st.write("Sur une ligne fréquente, les voyageurs ne regardent pas l'horaire : ils attendent le prochain bus. "
             "Ce qui compte est alors l'intervalle entre deux bus. Deux bus collés laissent un trou derrière eux : "
             "un bus plein, un bus vide, et une attente plus longue que promis.")
    if R:
        c = st.columns(4)
        c[0].metric("Bus en peloton", pct(R["bunched_share"]), help="Intervalle réel inférieur à la moitié de l'intervalle prévu")
        c[1].metric("Trous dans le service", pct(R["gap_share"]), help="Intervalle réel supérieur à 1,5 fois l'intervalle prévu")
        c[2].metric("Attente promise", f"{R['swt_min']:.1f} min".replace(".", ","))
        c[3].metric("Attente en plus", f"{R['ewt_min'] * 60:.0f} s", help="Excess wait time : attente réelle moins attente promise, voyageurs arrivant au hasard")
        reg = csv("regularity_by_line.csv").sort_values("ewt_min", ascending=False)
        a, b = st.columns(2)
        fig = px.bar(reg, x="line", y=reg.ewt_min * 60, labels={"line": "ligne", "y": "attente en plus (s)"},
                     color_discrete_sequence=[RED])
        fig.update_layout(title="Attente en plus de ce que promet l'horaire", xaxis_type="category", height=360, **layout)
        a.plotly_chart(fig, width="stretch")
        fig = px.bar(reg.sort_values("bunched_share", ascending=False), x="line", y="bunched_share",
                     labels={"line": "ligne", "bunched_share": "part des bus en peloton"}, color_discrete_sequence=[AMBER])
        fig.update_layout(title="Bus en peloton par ligne", xaxis_type="category", height=360, yaxis_tickformat=".0%", **layout)
        b.plotly_chart(fig, width="stretch")
        al = csv("regularity_along.csv")
        if not al.empty:
            pick = st.selectbox("Où la régularité se dégrade le long de la ligne", sorted(al.line.unique(), key=line_key))
            pats = patterns().get(pick, [])
            opts = [f"vers {p['names'][-1]}" for p in pats] or ["sens 1"]
            di = st.radio("Sens", range(len(opts)), format_func=lambda i: opts[i], horizontal=True, key="reg_dir")
            d = al[(al.line == pick) & (al.direction.astype(int) == di)].sort_values("position")
            fig = go.Figure()
            fig.add_scatter(x=d["name"], y=d.bunched_share * 100, mode="lines+markers", name="bus en peloton (%)",
                            line=dict(color=AMBER, width=2.5))
            fig.add_scatter(x=d["name"], y=d.ewt_min * 60, mode="lines+markers", name="attente en plus (s)",
                            line=dict(color=RED, width=2.5), yaxis="y2")
            fig.update_layout(height=420, xaxis=dict(tickangle=-45, type="category"),
                              yaxis=dict(title="bus en peloton (%)"),
                              yaxis2=dict(title="attente en plus (s)", overlaying="y", side="right"),
                              legend=dict(orientation="h", y=1.1), **layout)
            st.plotly_chart(fig, width="stretch")
            if d["bunched_share"].notna().sum() > 3:
                first = d.dropna(subset=["bunched_share"]).iloc[:3]["bunched_share"].mean()
                peak = d.loc[d["bunched_share"].idxmax()]
                action(f"<b>Ligne {pick}</b> : {pct(first)} de bus en peloton aux premiers arrêts, jusqu'à "
                       f"<b>{pct(peak.bunched_share)}</b> à <b>{peak['name']}</b>. Une régulation à l'intervalle "
                       "(retenir quelques secondes un bus trop proche du précédent) est à placer avant ce point.")

# ---------------------------------------------------------------- correspondances
with t_con:
    K = S.get("connections")
    st.write("Aux grands pôles d'échange, chaque arrivée d'un bus est associée au premier départ d'une autre ligne "
             "prévu entre 3 et 10 minutes plus tard. La correspondance est tenue si le voyageur, avec 2 minutes pour "
             "changer, arrive avant le départ réel. Sinon, il attend le bus suivant.")
    if K:
        c = st.columns(4)
        c[0].metric("Correspondances prévues", num(K["planned"]), help=f"{K['hubs']} pôles, jours de semaine, 6 h à 21 h")
        c[1].metric("Manquées", pct(K["missed_share"]))
        c[2].metric("Attente en plus si manquée", f"{K['extra_wait_min_when_missed']:.1f} min".replace(".", ","))
        c[3].metric("Manquées de 60 s ou moins", pct(K["missed_by_60s_or_less"]))
        a, b = st.columns(2)
        hub = csv("connections_by_hub.csv").sort_values("missed_share")
        fig = px.bar(hub, y="name", x="missed_share", orientation="h", labels={"name": "", "missed_share": "part manquée"},
                     color_discrete_sequence=[RED])
        fig.update_layout(title="Correspondances manquées par pôle", height=420, xaxis_tickformat=".0%", **layout)
        a.plotly_chart(fig, width="stretch")
        hr = csv("connections_by_hour.csv")
        fig = px.line(hr, x="hour", y="missed_share", markers=True, labels={"hour": "heure", "missed_share": "part manquée"})
        fig.update_traces(line_color=RED)
        fig.update_layout(title="Correspondances manquées par heure", height=420, yaxis_tickformat=".0%", **layout)
        b.plotly_chart(fig, width="stretch")
        st.subheader("Et si le bus attendait quelques secondes ?")
        holds = {h["hold_s"]: h for h in K["hold"]}
        hs = st.select_slider("Attente maximale du bus en correspondance", options=sorted(holds), value=60,
                              format_func=lambda v: f"{v} s")
        h = holds[hs]
        c = st.columns(3)
        c[0].metric("Correspondances manquées", pct(h["missed_share_after"]),
                    f"{(h['missed_share_after'] - K['missed_share']) * 100:+.1f} pts".replace(".", ","), delta_color="inverse")
        c[1].metric("Correspondances sauvées", num(h["recovered"]))
        c[2].metric("Attente moyenne d'un départ retenu", f"{h['mean_hold_s']:.0f} s")
        st.caption("Estimation sans effet en chaîne : un bus retenu part plus tard pour ses propres voyageurs. "
                   "À tester sur un pôle pilote, en priorité sur les correspondances les plus fréquentées.")
        pr = csv("connections_pairs.csv").head(20)
        st.dataframe(pr[["name", "line", "to_line", "connections", "missed_share", "extra_wait_min_when_missed"]]
                     .assign(missed_share=lambda x: (x.missed_share * 100).round(1),
                             extra_wait_min_when_missed=lambda x: x.extra_wait_min_when_missed.round(1))
                     .rename(columns={"name": "pôle", "line": "de la ligne", "to_line": "vers la ligne",
                                      "connections": "correspondances", "missed_share": "% manquées",
                                      "extra_wait_min_when_missed": "attente en plus (min)"}),
                     hide_index=True, width="stretch")

# ---------------------------------------------------------------- prédiction
with t_pred:
    st.write("Les écrans annoncent en général le retard actuel pour tous les arrêts suivants. Le modèle apprend, sur les "
             "semaines passées, comment ce retard évolue réellement, et il est jugé sur des jours qu'il n'a jamais vus. "
             "Deux modèles de quantiles donnent aussi une fourchette, recalibrée sur des jours mis de côté pour "
             "contenir le retard réel dans 80 % des cas.")
    sc = csv("prediction_scores.csv")
    c = st.columns(len(sc))
    for i, r in sc.iterrows():
        c[i].metric(f"{int(r.k)} arrêt(s) plus loin", f"{r.mae_model_s:.0f} s",
                    f"-{r.mae_reduction * 100:.0f} % d'erreur", delta_color="inverse",
                    help=f"Erreur moyenne du modèle ; en supposant le retard constant : {r.mae_persistence_s:.0f} s")
    a, b = st.columns(2)
    fig = go.Figure()
    fig.add_bar(x=[f"{k} arrêt(s)" for k in sc.k], y=sc.mae_persistence_s, name="retard supposé constant", marker_color=GREY)
    fig.add_bar(x=[f"{k} arrêt(s)" for k in sc.k], y=sc.mae_model_s, name="modèle LightGBM", marker_color=RED)
    fig.update_layout(barmode="group", title="Erreur moyenne sur la semaine de test", yaxis_title="erreur moyenne (s)",
                      height=340, legend=dict(orientation="h", y=-0.25), **layout)
    a.plotly_chart(fig, width="stretch")
    iv = csv("prediction_intervals.csv")
    if not iv.empty:
        fig = go.Figure()
        if "coverage_raw" in iv:
            fig.add_bar(x=[f"{k} arrêt(s)" for k in iv.k], y=iv.coverage_raw * 100, marker_color=GREY, name="quantiles bruts")
        fig.add_bar(x=[f"{k} arrêt(s)" for k in iv.k], y=iv.coverage * 100, marker_color=TEAL, name="après calibration",
                    text=[f"{v * 100:.0f} %" for v in iv.coverage], textposition="inside", insidetextanchor="end", textfont_color="white")
        fig.add_hline(y=80, line_dash="dot", line_color=INK, annotation_text="objectif 80 %", annotation_position="bottom left")
        fig.update_layout(title="Fourchette à 80 % : le retard réel y tombe-t-il ?", yaxis_range=[0, 105], barmode="group",
                          yaxis_title="% des cas", height=340, legend=dict(orientation="h", y=-0.25), **layout)
        b.plotly_chart(fig, width="stretch")

    st.subheader("Simulateur : un bus a du retard, que dire aux voyageurs plus loin ?")
    M, cats, margins = models()
    pats = patterns()
    if M and pats:
        c1, c2, c3 = st.columns(3)
        ln = c1.selectbox("Ligne", sorted(pats, key=line_key), index=sorted(pats, key=line_key).index("9") if "9" in pats else 0)
        pp = pats[ln]
        di = c2.radio("Sens", range(len(pp)), format_func=lambda i: f"vers {pp[i]['names'][-1]}", key="sim_dir")
        pat = pp[di]
        pos = c3.selectbox("Arrêt actuel", range(len(pat["stops"]) - 1), format_func=lambda i: pat["names"][i],
                           index=min(3, len(pat["stops"]) - 2))
        c1, c2, c3, c4 = st.columns(4)
        now = c1.slider("Retard actuel (min)", -1.0, 10.0, 3.0, 0.5)
        before = c2.slider("Retard deux arrêts avant (min)", -1.0, 10.0, 2.0, 0.5)
        hour = c3.slider("Heure", 6, 22, 17)
        wd = c4.selectbox("Jour", range(7), index=1, format_func=lambda i: ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"][i])
        n_ahead = min(10, len(pat["stops"]) - 1 - pos)
        ks = np.arange(1, n_ahead + 1)
        P = pd.DataFrame({"delay_now": now * 60, "trend_2": (now - before) * 60, "k": ks,
                          "sched_gap_min": [max(pat["minutes"][pos + k] - pat["minutes"][pos], 0.5) for k in ks], "hour": hour, "weekday": wd, "seq": pos,
                          "line": ln, "stop": pat["stops"][pos], "target_stop": [pat["stops"][pos + k] for k in ks]})
        X, _ = predict.encode(P, cats)
        med = now * 60 + M["median"].predict(X)
        if "q10" in M and "q90" in M:
            lo, hi = predict.band({0.1: M["q10"], 0.9: M["q90"]}, cats, P, margins)
        else:
            lo = hi = med
        lo, hi = np.minimum(lo, med), np.maximum(hi, med)
        names = [pat["names"][pos + k] for k in ks]
        fig = go.Figure()
        fig.add_scatter(x=names + names[::-1], y=list(hi / 60) + list(lo[::-1] / 60), fill="toself",
                        fillcolor="rgba(200,16,46,0.12)", line=dict(width=0), name="fourchette à 80 %")
        fig.add_scatter(x=names, y=med / 60, mode="lines+markers", line=dict(color=RED, width=3), name="prévision du modèle")
        fig.add_scatter(x=names, y=[now] * len(names), mode="lines", line=dict(color=GREY, dash="dash"), name="retard affiché constant")
        fig.update_layout(height=420, yaxis_title="retard prévu (min)", xaxis=dict(type="category", tickangle=-30),
                          legend=dict(orientation="h", y=1.12), **layout)
        st.plotly_chart(fig, width="stretch")
        st.caption("Temps de parcours prévus : médiane de l'horaire de ce parcours. Le modèle utilise la ligne, l'arrêt "
                   "actuel, l'arrêt visé, l'écart horaire, l'heure et le jour.")
    else:
        st.info("Lancer `python experiments/run_all.py` pour entraîner les modèles du simulateur.")

# ---------------------------------------------------------------- horaires
with t_tt:
    r = S["timetable"]
    st.write("Sur les jours d'apprentissage, l'outil repère les segments où les bus perdent du temps sur la plupart des "
             "courses, à la même tranche horaire, et propose d'y donner la moitié du temps perdu médian (90 s au plus). "
             "L'effet est mesuré sur la semaine de test, que l'outil n'a pas vue.")
    c = st.columns(4)
    c[0].metric("Segments ajustés", f"{r['segments_changed']}")
    c[1].metric("Retard, horaire actuel", pct(r["before"]["late"]))
    c[2].metric("Retard, horaire proposé", pct(r["after"]["late"]),
                f"{(r['after']['late'] - r['before']['late']) * 100:+.1f} pts".replace(".", ","), delta_color="inverse")
    c[3].metric("À l'heure, horaire proposé", pct(r["after"]["on_time"]),
                f"{(r['after']['on_time'] - r['before']['on_time']) * 100:+.1f} pts".replace(".", ","))
    props = csv("timetable_proposals.csv")
    props = props.assign(positive_share=(props.positive_share * 100).round(0), median_gain_s=props.median_gain_s.round(0),
                         band=props.band.map(BAND_FR))
    st.dataframe(props.head(40)[["line", "from_name", "to_name", "band", "trips", "median_gain_s", "positive_share", "add_s"]]
                 .rename(columns={"line": "ligne", "from_name": "de", "to_name": "à", "band": "tranche", "trips": "courses",
                                  "median_gain_s": "temps perdu médian (s)", "positive_share": "% des courses en perte",
                                  "add_s": "à ajouter (s)"}), hide_index=True, width="stretch")

# ---------------------------------------------------------------- fiche ligne
with t_line:
    st.write("Tout ce que les données disent d'une ligne, sur une page, avec les actions à étudier en premier.")
    by_line = csv("punctuality_by_line.csv")
    choices = sorted(by_line[by_line.events > 2000].line, key=line_key)
    worst = S["worst_lines_late"][0]["line"] if S.get("worst_lines_late") else choices[0]
    ln = st.selectbox("Ligne", choices, index=choices.index(str(worst)) if str(worst) in choices else 0, key="sheet_line")
    L = by_line[by_line.line == ln].iloc[0]
    reg = csv("regularity_by_line.csv")
    R = reg[reg.line == ln]
    c = st.columns(5)
    c[0].metric("Passages mesurés", num(L.events))
    c[1].metric("À l'heure", pct(L.on_time_share))
    c[2].metric("En retard", pct(L.late_share), f"{(L.late_share - S['net_late_share']) * 100:+.1f} pts vs réseau".replace(".", ","),
                delta_color="inverse")
    c[3].metric("Retard médian", f"{L.median_delay_s:.0f} s")
    c[4].metric("Attente en plus", f"{R.ewt_min.iloc[0] * 60:.0f} s" if len(R) else "ligne peu fréquente")
    a, b = st.columns(2)
    lh = csv("punctuality_by_line_hour.csv")
    d = lh[(lh.line == ln) & (lh.daytype == "weekday") & (lh.events > 100) & lh.hour.between(5, 24)]
    fig = px.bar(d, x="hour", y=d.late_share * 100, labels={"hour": "heure", "y": "% en retard"}, color_discrete_sequence=[RED])
    fig.update_layout(title="En semaine, heure par heure", height=330, **layout)
    a.plotly_chart(fig, width="stretch")
    cl = csv("delay_created_by_line.csv")
    cl = cl[cl.line == ln].head(8)
    if not cl.empty:
        fig = px.bar(cl.iloc[::-1], y=cl.iloc[::-1].from_name + " → " + cl.iloc[::-1].to_name, x="created_min_per_day",
                     orientation="h", labels={"y": "", "created_min_per_day": "retard créé (min par jour de semaine)"},
                     color_discrete_sequence=[AMBER])
        fig.update_layout(title="Où la ligne perd du temps", height=330, **layout)
        b.plotly_chart(fig, width="stretch")
    st.markdown("**Actions à étudier**")
    pr = csv("timetable_proposals.csv")
    pr = pr[pr.line == ln].sort_values("median_gain_s", ascending=False)
    if len(pr):
        tot = pr.add_s.sum()
        x = pr.iloc[0]
        action(f"<b>Temps de parcours</b> : {len(pr)} ajustements proposés ; le plus net entre <b>{x.from_name}</b> et "
               f"<b>{x.to_name}</b> ({BAND_FR.get(x.band, x.band)}), {x.median_gain_s:.0f} s perdues en médiane, "
               f"{x.add_s:.0f} s à ajouter.")
    if not cl.empty:
        x = cl.iloc[0]
        action(f"<b>Priorité bus</b> : le segment <b>{x.from_name} → {x.to_name}</b> crée {x.created_min_per_day:.0f} min "
               "de retard par jour de semaine sur cette ligne ; feux, voie réservée ou arrêt à examiner sur place.")
    al = csv("regularity_along.csv")
    al = al[al.line == ln].dropna(subset=["bunched_share"]) if not al.empty else al
    if len(al):
        x = al.loc[al.bunched_share.idxmax()]
        action(f"<b>Régularité</b> : jusqu'à {pct(x.bunched_share)} de bus en peloton à <b>{x['name']}</b> ; "
               "une régulation à l'intervalle en amont limiterait les trous dans le service.")
    cp = csv("connections_pairs.csv")
    cp = cp[(cp.to_line == ln) & (cp.connections >= 300)].sort_values("missed_share", ascending=False)
    if len(cp):
        x = cp.iloc[0]
        wait = f"{x.extra_wait_min_when_missed:.1f}".replace(".", ",")
        action(f"<b>Correspondance</b> : à <b>{x['name']}</b>, {pct(x.missed_share)} des correspondances de la ligne "
               f"{x.line} vers la ligne {ln} sont manquées ({wait} min d'attente en plus) ; "
               "une attente courte du bus en correspondance est à évaluer sur ce pôle.")
