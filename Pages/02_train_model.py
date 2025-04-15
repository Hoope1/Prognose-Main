# Neue Datei schreiben mit H2O AutoML (Variante 1: eine Session, zwei Modelle)
h2o_train_model_path = os.path.join(pages_dir, "02_train_model_h2o.py")

h2o_train_model_code = """
import streamlit as st
import pandas as pd
import h2o
from h2o.automl import H2OAutoML
import time
import os

st.set_page_config(page_title="PrognoseTrainer – H2O AutoML", layout="wide")
st.title("H2O AutoML: Gemeinsames GPU/CPU-Training beider Zielspalten")

uploaded_csv = st.file_uploader("Lade die vorbereiteten Feature-Daten hoch (CSV aus Teil 1)", type=["csv"])

if uploaded_csv:
    df = pd.read_csv(uploaded_csv)
    st.subheader("Vorschau der Trainingsdaten")
    st.dataframe(df.head(20))

    # Starte H2O
    h2o.init(max_mem_size="8G", nthreads=-1)

    # Konvertiere Pandas nach H2OFrame
    h2o_df = h2o.H2OFrame(df)

    # Spalten definieren
    target_math = "Mathematik"
    target_raum = "Raumvorstellung"
    feature_cols = [col for col in df.columns if col not in [target_math, target_raum]]

    # Nur Zeilen mit gültigem Zielwert
    df_math = h2o_df[h2o_df[target_math].isna() == False]
    df_raum = h2o_df[h2o_df[target_raum].isna() == False]

    if df_math.nrows == 0:
        st.error("❌ Keine gültigen Werte in der Zielspalte 'Mathematik'.")
        st.stop()
    if df_raum.nrows == 0:
        st.error("❌ Keine gültigen Werte in der Zielspalte 'Raumvorstellung'.")
        st.stop()

    if st.button("H2O AutoML Training starten"):
        st.info("Training beginnt. Beide Modelle werden nacheinander mit denselben Einstellungen trainiert...")
        progress = st.progress(0)
        start = time.time()

        # Mathematik-Modell trainieren
        aml_math = H2OAutoML(max_runtime_secs=600, seed=42, sort_metric="RMSE")
        aml_math.train(x=feature_cols, y=target_math, training_frame=df_math)
        progress.progress(50)

        # Raumvorstellung-Modell trainieren
        aml_raum = H2OAutoML(max_runtime_secs=600, seed=42, sort_metric="RMSE")
        aml_raum.train(x=feature_cols, y=target_raum, training_frame=df_raum)
        progress.progress(100)

        # Modelle speichern
        os.makedirs("models/h2o", exist_ok=True)
        h2o.save_model(model=aml_math.leader, path="models/h2o", force=True)
        h2o.save_model(model=aml_raum.leader, path="models/h2o", force=True)

        st.success(f"Training abgeschlossen in {round(time.time() - start, 2)} Sekunden.")
        st.write("Beide Modelle wurden in 'models/h2o' gespeichert.")

        h2o.shutdown(prompt=False)
"""

# Datei speichern
with open(h2o_train_model_path, "w", encoding="utf-8") as f:
    f.write(h2o_train_model_code)

h2o_train_model_path
