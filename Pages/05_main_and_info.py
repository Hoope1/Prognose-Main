import streamlit as st

st.set_page_config(page_title="PrognoseTrainer – Info & Anleitung", layout="wide")
st.title("Info & Anleitung")

st.markdown("""
### Überblick

Dieses Tool ermöglicht die Prognose der Leistung neuer Teilnehmer in **Mathematik** und **Raumvorstellung** auf Basis vergangener Daten.  
Es besteht aus folgenden Modulen:

1. **Daten vorbereiten** – Excel hochladen & Features automatisch berechnen
2. **Modelle trainieren** – Zwei KI-Modelle (AutoGluon) erstellen
3. **Teilnehmer prognostizieren** – Neue Eingabe und fortlaufende KI-Prognose
4. **Modelle verwalten** – Backup, Wiederherstellung, Löschen

---

### Setup-Hinweise (lokal ausführen)

1. Erstelle eine neue virtuelle Umgebung *(optional, empfohlen)*
2. Installiere alle Pakete mit:
```bash
pip install -r requirements.txt
