"""Dashboard screenshots (needs a running app: streamlit run app/dashboard.py)."""
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

SHOTS = Path(__file__).resolve().parents[1] / "docs" / "screenshots"
URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
TABS = [("Réseau", "network"), ("Régularité", "regularity"), ("Correspondances", "connections"),
        ("Prédire le retard", "prediction"), ("Ajuster les horaires", "timetable"), ("Fiche ligne", "line_sheet")]

SHOTS.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 1440, "height": 1100})
    page.goto(URL, wait_until="networkidle")
    page.wait_for_selector(".js-plotly-plot", timeout=180_000)
    time.sleep(5)
    for tab, name in TABS:
        page.get_by_role("tab", name=tab).click()
        time.sleep(5)
        if name == "prediction":             # the simulator is the part worth showing
            page.get_by_text("Simulateur", exact=False).first.scroll_into_view_if_needed()
            page.mouse.wheel(0, 520)
            time.sleep(3)
        page.screenshot(path=str(SHOTS / f"{name}.png"), full_page=True)
    b.close()
print("screenshots in", SHOTS)
