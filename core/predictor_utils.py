import os
import pandas as pd
from autogluon.tabular import TabularPredictor

def load_predictor(path):
    """Lädt ein gespeichertes AutoGluon-Modell aus dem angegebenen Pfad."""
    if os.path.exists(path):
        return TabularPredictor.load(path)
    return None

def predict_weeks_autoregressiv(df_eingabe, letzte_woche, teilnehmer_id, predictor_mathe, predictor_raum):
    """
    Autoregressive Prognose von Woche X+1 bis Woche 16.
    Nutzt jeweils die vorhergesagten Werte als Input für die nächste Woche.
    """
    df_sorted = df_eingabe.sort_values("Woche")
    prognose_ergebnis = {
        "Woche": [],
        "Mathematik (prognostiziert)": [],
        "Raumvorstellung (prognostiziert)": [],
        "Teilnehmer-ID": []
    }

    # Initialwerte aus echten Daten
    df_math = df_sorted["Mathematik"].dropna()
    df_raum = df_sorted["Raumvorstellung"].dropna()

    durchschnitt_mathe = df_math.mean()
    durchschnitt_raum = df_raum.mean()

    letzte_mathe = df_math.iloc[-1]
    letzte_raum = df_raum.iloc[-1]

    trend_mathe = durchschnitt_mathe - df_math.iloc[0] if len(df_math) > 1 else 0
    trend_raum = durchschnitt_raum - df_raum.iloc[0] if len(df_raum) > 1 else 0

    std_mathe = df_math.std() if len(df_math) > 1 else 0
    std_raum = df_raum.std() if len(df_raum) > 1 else 0

    for woche in range(letzte_woche + 1, 17):
        delta_letzteØ_mathe = letzte_mathe - durchschnitt_mathe if not pd.isna(letzte_mathe) else 0
        delta_letzteØ_raum = letzte_raum - durchschnitt_raum if not pd.isna(letzte_raum) else 0

        feature_row = pd.DataFrame([{
            "Teilnehmer-ID": teilnehmer_id,
            "Woche": woche,
            "Ø_Mathe_bis_Woche": durchschnitt_mathe,
            "Letzte_Mathe": letzte_mathe,
            "Trend_Mathe": trend_mathe,
            "STD_Mathe_bis_Woche": std_mathe,
            "Delta_LetzteØ_Mathe": delta_letzteØ_mathe,
            "Ø_Raum_bis_Woche": durchschnitt_raum,
            "Letzte_Raum": letzte_raum,
            "Trend_Raum": trend_raum,
            "STD_Raum_bis_Woche": std_raum,
            "Delta_LetzteØ_Raum": delta_letzteØ_raum,
            "Rel_Woche": woche / 16
        }])

        # Vorhersagen erzeugen
        mathe_pred = predictor_mathe.predict(feature_row)[0]
        raum_pred = predictor_raum.predict(feature_row)[0]

        # Speichern
        prognose_ergebnis["Woche"].append(woche)
        prognose_ergebnis["Mathematik (prognostiziert)"].append(mathe_pred)
        prognose_ergebnis["Raumvorstellung (prognostiziert)"].append(raum_pred)
        prognose_ergebnis["Teilnehmer-ID"].append(teilnehmer_id)

        # Update für nächste Runde
        df_math = pd.concat([df_math, pd.Series([mathe_pred])])
        df_raum = pd.concat([df_raum, pd.Series([raum_pred])])

        letzte_mathe = mathe_pred
        letzte_raum = raum_pred
        durchschnitt_mathe = df_math.mean()
        durchschnitt_raum = df_raum.mean()
        trend_mathe = durchschnitt_mathe - df_math.iloc[0] if len(df_math) > 1 else 0
        trend_raum = durchschnitt_raum - df_raum.iloc[0] if len(df_raum) > 1 else 0
        std_mathe = df_math.std() if len(df_math) > 1 else 0
        std_raum = df_raum.std() if len(df_raum) > 1 else 0

    return pd.DataFrame(prognose_ergebnis)
import os
import h2o

def load_h2o_models(model_dir="models/h2o"):
    """
    Lädt die H2O-Modelle für Mathematik und Raumvorstellung aus dem angegebenen Verzeichnis.
    Gibt ein Tupel (modell_mathe, modell_raum) zurück.
    """
    h2o.init(max_mem_size="8G", nthreads=-1)

    model_files = os.listdir(model_dir)
    math_model_file = next((f for f in model_files if "Mathematik" in f), None)
    raum_model_file = next((f for f in model_files if "Raumvorstellung" in f), None)

    if not math_model_file or not raum_model_file:
        raise FileNotFoundError("Eines der Modelle konnte nicht gefunden werden.")

    model_math = h2o.load_model(os.path.join(model_dir, math_model_file))
    model_raum = h2o.load_model(os.path.join(model_dir, raum_model_file))

    return model_math, model_raum
