import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="PrognoseTrainer – Daten vorbereiten", layout="wide")
st.title("Daten vorbereiten für PrognoseTrainer")

uploaded_file = st.file_uploader("Lade deine Excel-Datei hoch:", type=["xlsx"])

def extract_features(df_long: pd.DataFrame) -> pd.DataFrame:
    feature_rows = []

    for teilnehmer_id, gruppe in df_long.groupby("Teilnehmer-ID"):
        gruppe = gruppe.sort_values("Woche")
        for woche in range(2, 17):  # ab Woche 2 möglich
            bisher = gruppe[gruppe["Woche"] < woche]
            if bisher.empty:
                continue

            mathe_werte = bisher["Mathematik"].dropna()
            raum_werte = bisher["Raumvorstellung"].dropna()

            letzte_math = mathe_werte.values[-1] if not mathe_werte.empty else np.nan
            letzte_raum = raum_werte.values[-1] if not raum_werte.empty else np.nan

            Ø_math = mathe_werte.mean()
            Ø_raum = raum_werte.mean()

            trend_math = Ø_math - mathe_werte.iloc[0] if len(mathe_werte) > 1 else 0
            trend_raum = Ø_raum - raum_werte.iloc[0] if len(raum_werte) > 1 else 0

            std_math = mathe_werte.std() if len(mathe_werte) > 1 else 0
            std_raum = raum_werte.std() if len(raum_werte) > 1 else 0

            delta_letzteØ_math = letzte_math - Ø_math if not np.isnan(letzte_math) else 0
            delta_letzteØ_raum = letzte_raum - Ø_raum if not np.isnan(letzte_raum) else 0

            row = {
                "Teilnehmer-ID": teilnehmer_id,
                "Woche": woche,
                "Ø_Mathe_bis_Woche": Ø_math,
                "Letzte_Mathe": letzte_math,
                "Trend_Mathe": trend_math,
                "STD_Mathe_bis_Woche": std_math,
                "Delta_LetzteØ_Mathe": delta_letzteØ_math,
                "Ø_Raum_bis_Woche": Ø_raum,
                "Letzte_Raum": letzte_raum,
                "Trend_Raum": trend_raum,
                "STD_Raum_bis_Woche": std_raum,
                "Delta_LetzteØ_Raum": delta_letzteØ_raum,
                "Rel_Woche": woche / 16
            }

            ziel = gruppe[gruppe["Woche"] == woche]
            if not ziel.empty:
                row["Mathematik"] = ziel["Mathematik"].values[0]
                row["Raumvorstellung"] = ziel["Raumvorstellung"].values[0]

            feature_rows.append(row)

    return pd.DataFrame(feature_rows)

# === Hauptprogramm ===
if uploaded_file:
    df_excel = pd.read_excel(uploaded_file)

    if "Teilnehmer-Name" in df_excel.columns:
        df_excel = df_excel.drop(columns=["Teilnehmer-Name"])  # Anonymisierung

    # Daten transformieren
    df_long = pd.melt(df_excel, id_vars=["Teilnehmer-ID"], var_name="Woche_Fach", value_name="Wert")
    df_long[['Woche', 'Fach']] = df_long['Woche_Fach'].str.extract(r"Woche (\d+) - (.+)")
    
    # NEU: Klammern wie (%) aus dem Fachnamen entfernen
    df_long["Fach"] = df_long["Fach"].str.replace(r"\(.*\)", "", regex=True).str.strip()
    
    df_long = df_long.drop(columns=["Woche_Fach"])
    df_long = df_long.dropna(subset=["Woche"])
    df_long["Woche"] = df_long["Woche"].astype(int)

    df_long = df_long.pivot_table(index=["Teilnehmer-ID", "Woche"], columns="Fach", values="Wert", aggfunc='first').reset_index()

    if "Mathematik" in df_long.columns:
        df_long["Mathematik"] = pd.to_numeric(df_long["Mathematik"], errors="coerce")
    else:
        df_long["Mathematik"] = np.nan

    if "Raumvorstellung" in df_long.columns:
        df_long["Raumvorstellung"] = pd.to_numeric(df_long["Raumvorstellung"], errors="coerce")
    else:
        df_long["Raumvorstellung"] = np.nan

    df_features = extract_features(df_long)
    st.success("Feature-Engineering abgeschlossen. Vorschau:")
    st.dataframe(df_features.head(20))

    csv = df_features.to_csv(index=False).encode("utf-8")
    st.download_button("Feature-Tabelle als CSV speichern", data=csv, file_name="features.csv", mime="text/csv")
