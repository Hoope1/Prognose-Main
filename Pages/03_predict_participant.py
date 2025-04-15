import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from core.predictor_utils import load_predictor, predict_weeks_autoregressiv

st.set_page_config(page_title="03 - Teilnehmer Prognose", layout="wide")

st.title("📈 Teilnehmer-Prognose")
st.markdown("Gib die bekannten Werte ein – der PrognoseTrainer sagt dir, wie es weitergeht!")

# === 1. Modelle laden ===
predictor_mathe = load_predictor("models/model_mathematik")
predictor_raum = load_predictor("models/model_raumvorstellung")

if predictor_mathe is None or predictor_raum is None:
    st.error("❗ Modelle nicht gefunden. Bitte zuerst trainieren (Seite: 02 - Training).")
    st.stop()

# === 2. Teilnehmerdaten eingeben ===
with st.form("prognose_formular"):
    teilnehmer_id = st.number_input("Teilnehmer-ID", min_value=1, max_value=9999, step=1)
    letzte_woche = st.slider("Letzte bekannte Woche", min_value=1, max_value=15, value=4)

    st.markdown("### Eingabewerte Woche 1 bis letzte bekannte Woche")
    eingabedaten = []
    for woche in range(1, letzte_woche + 1):
        col1, col2 = st.columns(2)
        with col1:
            mathe = st.number_input(f"Mathematik Woche {woche}", min_value=0, max_value=100, step=1, key=f"mathe_{woche}")
        with col2:
            raum = st.number_input(f"Raumvorstellung Woche {woche}", min_value=0, max_value=100, step=1, key=f"raum_{woche}")
        eingabedaten.append({
            "Teilnehmer-ID": teilnehmer_id,
            "Woche": woche,
            "Mathematik": mathe,
            "Raumvorstellung": raum
        })

    submitted = st.form_submit_button("Prognose starten")

if submitted:
    df_eingabe = pd.DataFrame(eingabedaten)
    prognose_df = predict_weeks_autoregressiv(df_eingabe, letzte_woche, teilnehmer_id, predictor_mathe, predictor_raum)

    st.subheader("Prognose bis Woche 16")
    st.dataframe(prognose_df)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prognose_df["Woche"], y=prognose_df["Mathematik (prognostiziert)"],
                             mode='lines+markers', name='Mathematik'))
    fig.add_trace(go.Scatter(x=prognose_df["Woche"], y=prognose_df["Raumvorstellung (prognostiziert)"],
                             mode='lines+markers', name='Raumvorstellung'))
    fig.update_layout(title="Prognoseverlauf", xaxis_title="Woche", yaxis_title="Wert (%)")
    st.plotly_chart(fig, use_container_width=True)
