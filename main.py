import streamlit as st

st.set_page_config(page_title="PrognoseTrainer", layout="wide")
st.title("Willkommen zum PrognoseTrainer")

st.markdown("""
### Navigation

Bitte nutze die linke Seitenleiste, um durch die Module zu navigieren:

1. **Daten vorbereiten** – Lade deine Excel-Datei hoch und generiere automatisch die Trainingsdaten
2. **Modelle trainieren** – Trainiere zwei KI-Modelle für Mathematik & Raumvorstellung (AutoGluon)
3. **Teilnehmer prognostizieren** – Gib bekannte Leistungen ein und simuliere den Verlauf bis Woche 16
4. **Modelle verwalten** – Backup, Wiederherstellung oder Löschen der trainierten Modelle
""")
