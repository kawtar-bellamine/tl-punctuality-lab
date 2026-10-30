"""README.md from results/summary.json, so every number comes from the last run."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "results" / "summary.json").read_text())
P = pd.DataFrame(S["prediction"]).set_index("k")
IV = pd.DataFrame(S["prediction_intervals"]).set_index("k")
T, D, G, K = S["timetable"], S["delay_created"], S["regularity"], S["connections"]
H = {h["hold_s"]: h for h in K["hold"]}
props = pd.read_csv(ROOT / "results" / "timetable_proposals.csv", dtype={"line": str})
top = props.sort_values("median_gain_s", ascending=False).iloc[0]
hub = pd.read_csv(ROOT / "results" / "connections_by_hub.csv").sort_values("missed_share", ascending=False)
pct = lambda v, d=1: f"{v * 100:.{d}f} %".replace(".", ",")  # noqa: E731
pct_en = lambda v, d=1: f"{v * 100:.{d}f}%"  # noqa: E731
n = lambda v: f"{v:,.0f}".replace(",", " ")  # noqa: E731
f1 = lambda v: f"{v:.1f}".replace(".", ",")  # noqa: E731
worst = ", ".join(f"{d['line']} ({d['late_share'] * 100:.0f} %)" for d in S["worst_lines_late"][:4])
m1 = next((d for d in S["best_lines_on_time"] if d["line"] == "m1"), S["best_lines_on_time"][0])
top_seg = D["top"][0]
reg_worst = G["worst_lines"][0]
sheets = " · ".join(f"[ligne {d['line']}](docs/fiches/ligne_{d['line']}.png)" for d in S["worst_lines_late"])

README = f"""<p align="center"><img src="docs/figures/banner.png" alt="tl punctuality lab" width="100%"></p>

<p align="center">
<a href="https://github.com/kawtar-bellamine/tl-punctuality-lab/actions"><img src="https://github.com/kawtar-bellamine/tl-punctuality-lab/actions/workflows/tests.yml/badge.svg" alt="tests"></a>
<img src="https://img.shields.io/badge/python-3.12-C8102E" alt="python">
<img src="https://img.shields.io/badge/données-opentransportdata.swiss-14213D" alt="data">
<img src="https://img.shields.io/badge/modèle-LightGBM-00909C" alt="model">
<img src="https://img.shields.io/badge/tableau%20de%20bord-Streamlit-E59500" alt="dashboard">
</p>

**Projet de [Abderrahmane BOULAMI](https://github.com/Abderrahmane-BOULAMI) et [Kawtar BELLAMINE](https://github.com/kawtar-bellamine), élèves ingénieurs aux Arts et Métiers ParisTech.**

Les données réelles de **chaque bus et métro tl**, à **chaque arrêt**, sur **tout le mois de septembre 2026**
({n(S['raw_rows'])} passages, {S['net_lines']} lignes, {S['net_stops']} quais), transformées en décisions
d'exploitation :

| | Question | Ce que l'outil livre |
|---|---|---|
| 1 | **Où et quand** la ponctualité se perd-elle ? | indicateurs par ligne, heure, arrêt |
| 2 | **Où sur la carte** le retard se crée-t-il ? | carte des segments qui coûtent le plus de minutes par jour |
| 3 | Les bus fréquents passent-ils **régulièrement** ? | bus en peloton, trous, attente en plus pour les voyageurs |
| 4 | Les **correspondances** sont-elles tenues ? | taux de correspondances manquées par pôle, effet d'une attente courte |
| 5 | Quel retard **annoncer** quelques arrêts plus loin ? | prévision LightGBM et fourchette à 80 %, simulateur |
| 6 | Quels **horaires** ajuster ? | ajustements de temps de parcours, testés sur une semaine inédite |
| 7 | Que faire **sur une ligne** ? | fiche d'actions d'une page par ligne |

## Résultats

| | Résultat |
|---|---|
| Passages mesurés (statut REAL) | {n(S['net_events'])} sur {n(S['raw_rows'])} enregistrements ({pct(S['measured_share'])} mesurés) |
| Ponctualité du réseau | **{pct(S['net_on_time_share'])}** à l'heure, {pct(S['net_late_share'])} en retard (≥ 3 min), {pct(S['net_early_share'])} en avance (> 1 min) |
| Pointe du soir (semaine, 17 h à 19 h) | **{pct(S['peak_hour_late_share'])}** des départs en retard |
| Lignes les plus en retard | {worst} ; métro m1 : {pct(m1['on_time_share'])} à l'heure |
| Où le retard se crée | **10 segments** sur {n(D['segments'])} ({pct(10 / D['segments'])}) créent **{pct(D['top10_share'], 0)}** du retard ; le premier, {top_seg['from_name']} → {top_seg['to_name']}, coûte {n(top_seg['created_min_per_day'])} min par jour de semaine |
| Régularité des lignes fréquentes | {pct(G['bunched_share'])} de bus en peloton, **{G['ewt_min'] * 60:.0f} s** d'attente en plus par voyageur ; ligne {reg_worst['line']} : {pct(reg_worst['bunched_share'])} en peloton |
| Correspondances aux {K['hubs']} grands pôles | {n(K['planned'])} correspondances prévues, **{pct(K['missed_share'])} manquées**, {f1(K['extra_wait_min_when_missed'])} min d'attente en plus quand elles le sont |
| Attente courte du bus en correspondance | jusqu'à 60 s : correspondances manquées **{pct(K['missed_share'])} → {pct(H[60]['missed_share_after'])}** |
| Retard 1 arrêt plus loin | erreur moyenne **{P.loc[1, 'mae_model_s']:.0f} s** contre {P.loc[1, 'mae_persistence_s']:.0f} s en gardant le retard actuel (**-{P.loc[1, 'mae_reduction'] * 100:.0f} %**) |
| Retard 5 arrêts plus loin | **{P.loc[5, 'mae_model_s']:.0f} s** contre {P.loc[5, 'mae_persistence_s']:.0f} s (**-{P.loc[5, 'mae_reduction'] * 100:.0f} %**), {P.loc[5, 'within_60s_model'] * 100:.0f} % des prévisions à moins d'une minute contre {P.loc[5, 'within_60s_persistence'] * 100:.0f} % |
| Fourchette à 80 % | le retard réel y tombe dans **{pct(IV.loc[5, 'coverage'], 0)}** des cas à 5 arrêts (calibration conforme) |
| Horaires proposés | {T['segments_changed']} ajustements (segment × tranche horaire), {T['median_minutes_added_per_trip'] * 60:.0f} s ajoutées par course concernée en médiane |
| Effet sur la semaine de test (jours ouvrés) | départs en retard **{pct(T['before']['late'])} → {pct(T['after']['late'])}**, à l'heure {pct(T['before']['on_time'])} → {pct(T['after']['on_time'])} |

Toutes les mesures de prédiction et d'horaires sont faites sur la semaine du {S['test_from']} au {S['last_day']},
jamais utilisée pour apprendre.

## 1. Mesurer

<p align="center"><img src="docs/figures/lateness_by_hour.png" width="85%"></p>

En semaine, la part de départs en retard monte dès 7 h et culmine à la pointe du soir ; le week-end, le réseau reste
bien plus ponctuel. Le lundi du Jeûne fédéral (21 septembre, férié dans le canton de Vaud) est traité comme un
dimanche.

<p align="center"><img src="docs/figures/lateness_by_line.png" width="85%"></p>

## 2. Localiser : où le retard se crée

<p align="center"><img src="docs/figures/delay_map.png" width="85%"></p>

Entre deux arrêts consécutifs, le retard créé est la variation du retard mesuré d'un départ au suivant (trajet et
arrêt), cumulée sur toutes les courses d'un jour de semaine. Les arrêts sont placés grâce à la liste officielle des
points d'arrêt suisses. Les segments du centre (Chauderon, Bel-Air, St-François) concentrent le retard : c'est là
que priorité aux feux, voies réservées ou organisation de la montée rapportent le plus de minutes.

## 3. Régularité des lignes fréquentes

<p align="center"><img src="docs/figures/regularity.png" width="95%"></p>

Sur une ligne à un bus toutes les 10 minutes ou mieux, les voyageurs n'ont pas d'horaire en tête : ce qui compte est
l'intervalle. Pour chaque arrêt et chaque sens, les intervalles réels sont comparés aux intervalles prévus.

* **Peloton** : intervalle réel inférieur à la moitié de l'intervalle prévu ; **trou** : plus de 1,5 fois.
* **Attente en plus** (excess wait time) : attente moyenne réelle moins attente promise par l'horaire, pour des
  voyageurs qui arrivent au hasard. Promise : {f1(G['swt_min'])} min ; réelle : {f1(G['awt_min'])} min.
* Le profil le long de la ligne montre **où** la régularité se dégrade, donc où placer une régulation à l'intervalle.

## 4. Correspondances aux grands pôles

<p align="center"><img src="docs/figures/connections.png" width="95%"></p>

Aux {K['hubs']} arrêts desservis par le plus de lignes, chaque arrivée est associée au premier départ d'une autre
ligne prévu entre 3 et 10 minutes plus tard. Avec 2 minutes pour changer, la correspondance est manquée si ce bus
part avant le voyageur, qui attend alors le suivant. Le pôle le plus exposé est {hub.iloc[0]['name']}
({pct(hub.iloc[0]['missed_share'])} manquées). Avec 1 minute pour changer, la part manquée serait de
{pct(K['missed_share_by_walk_s']['60'])} ; avec 3 minutes, {pct(K['missed_share_by_walk_s']['180'])}.

Une attente du bus en correspondance limitée à 30, 60 ou 90 s est simulée sur les mêmes correspondances :
{pct(H[30]['missed_share_after'])}, {pct(H[60]['missed_share_after'])} et {pct(H[90]['missed_share_after'])} manquées.
À 60 s, {pct(H[60]['departures_held_share'], 0)} des départs concernés seraient retenus, {H[60]['mean_hold_s']:.0f} s en
moyenne. L'estimation ignore l'effet en chaîne sur les voyageurs déjà à bord : c'est un argument pour un essai sur un
pôle, pas une conclusion.

## 5. Prédire le retard quelques arrêts plus loin

<p align="center"><img src="docs/figures/prediction.png" width="95%"></p>

* **Question.** Un bus a 2 minutes de retard à cet arrêt : quel retard aura-t-il 1, 3, 5 ou 10 arrêts plus loin ?
* **Référence.** Garder le retard actuel, ce que font la plupart des affichages.
* **Modèle.** LightGBM sur le changement de retard, avec le retard actuel, sa tendance sur les deux arrêts
  précédents, la ligne, l'arrêt de départ et l'arrêt visé, l'écart horaire prévu, l'heure et le jour.
  {n(S['train_pairs'])} paires d'apprentissage tirées des trois premières semaines,
  {n(S['test_pairs'])} paires de test sur la dernière semaine.
* **Fourchette.** Deux modèles de quantiles (10 % et 90 %) donnent une fourchette, recalibrée par prédiction conforme
  sur les derniers jours d'apprentissage : sur la semaine de test, le retard réel y tombe dans
  {pct(IV.loc[1, 'coverage'], 0)} à {pct(IV.loc[10, 'coverage'], 0)} des cas selon l'horizon, pour un objectif de 80 %.
* **Simulateur.** Dans le tableau de bord, on choisit une ligne, un arrêt et un retard : la prévision et sa
  fourchette s'affichent pour les dix arrêts suivants.

## 6. Agir sur les horaires

<p align="center"><img src="docs/figures/timetable.png" width="95%"></p>

Pour chaque segment entre deux arrêts, le temps de parcours réel est comparé au temps prévu. Quand un bus perd du
temps sur la plupart des courses, à la même tranche horaire, l'horaire est trop serré à cet endroit. Sur les semaines
d'apprentissage, l'outil propose d'y ajouter la moitié du temps perdu médian (90 s au plus) ; il applique ensuite ces
ajustements à la semaine de test et recalcule la ponctualité. Exemple : ligne {top['line']}, {top['from_name']} → {top['to_name']},
{top['band'].replace('_', ' ')}, {top['median_gain_s']:.0f} s perdues en médiane sur {int(top['trips'])} courses, {int(top['add_s'])} s proposées.

Les bus sont supposés rouler comme ils l'ont fait ; seul l'horaire qui les juge change. Ajouter seulement la moitié
du temps perdu garde les départs en avance sous contrôle ({pct(T['before']['early'])} → {pct(T['after']['early'])}).

## 7. Une fiche d'actions par ligne

<p align="center"><img src="docs/fiches/ligne_{S['worst_lines_late'][0]['line']}.png" width="95%"></p>

`tools/line_sheets.py` produit, pour chaque ligne, une page qui rassemble ponctualité heure par heure, segments qui
coûtent le plus, régularité le long de la ligne et actions à étudier en premier : {sheets}
([toutes en PDF](docs/fiches/fiches_lignes.pdf)).

## Tableau de bord

```bash
pip install -r requirements.txt
streamlit run app/dashboard.py
```

Sept onglets en français : réseau, carte du retard, régularité, correspondances, prédiction (avec simulateur),
horaires, fiche ligne.

| Régularité | Simulateur de retard |
|---|---|
| ![régularité](docs/screenshots/regularity.png) | ![prédiction](docs/screenshots/prediction.png) |
| **Correspondances** | **Fiche ligne** |
| ![correspondances](docs/screenshots/connections.png) | ![fiche](docs/screenshots/line_sheet.png) |

## Données

* **Données réelles (Ist-Daten)** d'[opentransportdata.swiss](https://opentransportdata.swiss) : pour chaque course
  de chaque entreprise de transport suisse, l'heure prévue et l'heure réelle (ou la dernière prévision) à chaque
  arrêt. L'archive de septembre 2026 pèse 1,4 Go pour toute la Suisse ; `tools/extract_tl.py` la lit par requêtes
  partielles, décompresse chaque jour à la volée et ne garde que l'opérateur TL, soit environ 123 000 lignes par jour
  ouvré et 39 Mo pour le mois (non versionnés ; deux jours sont inclus dans `data/raw/sample/`).
* **Points d'arrêt** (didok, opentransportdata.swiss, via data.sbb.ch) : coordonnées des {n(458)} arrêts desservis,
  dans `data/raw/tl_stations.csv`.
* Un passage compte s'il est **mesuré** (statut REAL), non supprimé, avec un écart entre -10 et +60 min.
  **En retard** : 3 minutes ou plus, seuil de ponctualité des transports publics suisses. **En avance** : départ plus
  d'une minute avant l'horaire.

## Reproduire

```bash
python tools/extract_tl.py 2026-09         # reconstruit l'extrait depuis l'archive publique (optionnel)
python experiments/run_all.py              # étude complète, environ 13 minutes
python experiments/run_all.py --sample     # deux jours inclus dans le dépôt (CI)
python -m pytest                           # réseaux synthétiques à vérité connue + contrôles sur l'extrait réel
python tools/figures.py && python tools/line_sheets.py && python tools/write_readme.py
```

Les tests construisent des réseaux dont la vérité est connue : un segment lent doit être retrouvé, un service
parfaitement régulier ne doit montrer aucune attente en plus, des bus décalés deux à deux doivent apparaître en
peloton à partir du bon arrêt, une correspondance tenue ou manquée de 30 s doit être classée comme telle.

## Limites et suites

* Un mois de données : la météo, les travaux et les événements ne sont pas encore des variables. Les données de
  MétéoSuisse et les annonces de perturbations sont les prochaines à ajouter.
* La fréquentation n'est pas publique : chaque correspondance et chaque passage comptent autant. Avec les comptages
  des tl, les actions seraient classées par nombre de voyageurs touchés.
* Les effets des horaires et de l'attente en correspondance sont des estimations sans effet en chaîne ; un essai sur
  une ligne ou un pôle pilote est la mesure définitive.

## English summary

Real departure data for every tl bus and metro at every stop in September 2026 ({n(S['raw_rows'])} stop events),
turned into operating decisions: punctuality by line and hour (on time {pct_en(S['net_on_time_share'])},
{pct_en(S['peak_hour_late_share'])} late in the weekday evening peak), a map of where delay is created
(10 segments out of {D['segments']} create {pct_en(D['top10_share'], 0)} of it), headway regularity on frequent lines ({G['ewt_min'] * 60:.0f} s excess
wait per passenger), connection reliability at {K['hubs']} hubs ({pct_en(K['missed_share'])} missed, {pct_en(H[60]['missed_share_after'])}
with a 60 s hold), delay prediction a few stops ahead with LightGBM ({P.loc[5, 'mae_reduction'] * 100:.0f}% lower
error than holding the current delay five stops ahead, on an unseen week, with a conformally calibrated 80% band),
{T['segments_changed']} running-time adjustments that cut lateness on the test week from {pct_en(T['before']['late'])}
to {pct_en(T['after']['late'])}, and a one-page action sheet per line.

Data: opentransportdata.swiss (actual data and service points), terms of use of opentransportdata.swiss. Code: MIT.
"""
(ROOT / "README.md").write_text(README)
print("README.md written")
