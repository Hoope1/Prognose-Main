import pandas as pd
import numpy as np

def extract_features(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Wandelt Langformat-Daten (Woche, Fach, Wert) in Feature-Zeilen um.
    Jede Zeile beschreibt Woche w für einen Teilnehmer (basierend auf w-1).
    """
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
