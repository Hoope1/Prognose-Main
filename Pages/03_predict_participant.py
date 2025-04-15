import streamlit as st
import pandas as pd
import h2o
import os

from h2o.frame import H2OFrame
from h2o.estimators import H2OAutoML

st.set_page_config(page_title="PrognoseTester – H2O", layout="wide")
st.title("H2O AutoML Prognose für neuen Teilnehmer")

uploaded_csv = st.file_uploader("Lade Teilnehmerdaten (ohne Zielwerte) hoch", type=["csv"])

if uploaded_csv:
    df_input = pd.read_csv(uploaded_csv)
    st.subheader("Eingabedaten für Vorhersage")
    st.dataframe(df_input)

    if not os.path.exists("models/h2o"):
        st.error("❌ Es wurden keine H2O-Modelle gefunden. Bitte zuerst Training durchführen.")
        st.stop()

    # Starte H2O
    h2o.init(max_mem_size="8G", nthreads=-1)

    # Konvertiere Eingabedaten in H2OFrame
    input_h2o = h2o.H2OFrame(df_input)

    # Lade Modelle
    model_files = os.listdir("models/h2o")
    math_model_file = next((f for f in model_files if "Mathematik" in f), None)
    raum_model_file = next((f for f in model_files if "Raumvorstellung" in f), None)

    if not math_model_file or not raum_model_file:
        st.error("❌ Es konnten nicht beide Modelle (Mathematik & Raumvorstellung) geladen werden.")
        h2o.shutdown(prompt=False)
        st.stop()

    model_math = h2o.load_model(os.path.join("models/h2o", math_model_file))
    model_raum = h2o.load_model(os.path.join("models/h2o", raum_model_file))

    # Vorhersagen durchführen
    st.info("Führe Vorhersagen durch...")

    prediction_math = model_math.predict(input_h2o).as_data_frame().rename(columns={"predict": "Prognose_Mathematik"})
    prediction_raum = model_raum.predict(input_h2o).as_data_frame().rename(columns={"predict": "Prognose_Raumvorstellung"})

    # Ergebnisse kombinieren
    results = pd.concat([df_input.reset_index(drop=True), prediction_math, prediction_raum], axis=1)

    st.success("✅ Vorhersagen erfolgreich durchgeführt!")
    st.subheader("Vorhersageergebnisse")
    st.dataframe(results)

    # Optional: Download anbieten
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button("Vorhersagen als CSV herunterladen", data=csv, file_name="prognosen_h2o.csv", mime="text/csv")

    h2o.shutdown(prompt=False)
