import streamlit as st
import pandas as pd
import time
import torch
from autogluon.tabular import TabularPredictor

# Streamlit-Seitenkonfiguration
st.set_page_config(page_title="PrognoseTrainer – Auto-GPU-Ensemble (Korrekt)", layout="wide")
st.title("Automatisches GPU-basiertes Training mit Ensemble (Valide AutoGluon Modelle)")

# CSV-Upload
uploaded_csv = st.file_uploader("Lade die vorbereiteten Feature-Daten hoch (CSV aus Teil 1)", type=["csv"])

# GPU-Konfiguration (dynamisch prüfen)
def get_gpu_config():
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return {"ag_args_fit": {"fit": {"num_gpus": 1}}}
    else:
        return {}  # Fallback auf CPU

ag_args_gpu = get_gpu_config()

if uploaded_csv:
    df = pd.read_csv(uploaded_csv)
    st.subheader("Vorschau der Trainingsdaten")
    st.dataframe(df.head(20))

    model_path_base = "models/auto_gpu_ensemble_valid"
    df_math = df.dropna(subset=["Mathematik"]).copy()
    df_raum = df.dropna(subset=["Raumvorstellung"]).copy()

    if df_math["Mathematik"].dropna().empty:
        st.error("❌ Keine gültigen Werte in der Zielspalte 'Mathematik'.")
        st.stop()
    if df_raum["Raumvorstellung"].dropna().empty:
        st.error("❌ Keine gültigen Werte in der Zielspalte 'Raumvorstellung'.")
        st.stop()

    if st.button("Automatisches GPU-Ensemble-Training starten (Valide AutoGluon Modelle)"):
        st.info("Automatisches Training valider Modelle beginnt...")
        progress = st.progress(0)
        status_area = st.empty()
        start_time = time.time()
        gpu_capable_models_math = set()
        gpu_capable_models_raum = set()

        valid_models = ['LR', 'LGB', 'GBM', 'XGB', 'CAT', 'NN_TORCH', 'FASTAI', 'RF', 'XT', 'KNN']
        time_limit_probe = 60
        total_steps = len(valid_models) * 2
        step_increment = 50 / total_steps
        current_progress = 0

        st.subheader("GPU-Prüfung für Modelle...")
        for model in valid_models:
            status_area.text(f"Prüfe GPU-Fähigkeit: Mathematik – {model}")
            try:
                TabularPredictor(label="Mathematik", problem_type="regression").fit(
                    train_data=df_math.sample(frac=0.1, random_state=42),
                    hyperparameters={model: [{}]},
                    time_limit=time_limit_probe,
                    verbosity=0,
                    **ag_args_gpu
                )
                gpu_capable_models_math.add(model)
                st.info(f"✅ {model} ist GPU-fähig (Mathematik)")
            except Exception as e:
                st.warning(f"❌ {model} nicht GPU-fähig (Mathematik): {e}")
            current_progress += step_increment
            progress.progress(int(current_progress))

            status_area.text(f"Prüfe GPU-Fähigkeit: Raumvorstellung – {model}")
            try:
                TabularPredictor(label="Raumvorstellung", problem_type="regression").fit(
                    train_data=df_raum.sample(frac=0.1, random_state=42),
                    hyperparameters={model: [{}]},
                    time_limit=time_limit_probe,
                    verbosity=0,
                    **ag_args_gpu
                )
                gpu_capable_models_raum.add(model)
                st.info(f"✅ {model} ist GPU-fähig (Raumvorstellung)")
            except Exception as e:
                st.warning(f"❌ {model} nicht GPU-fähig (Raumvorstellung): {e}")
            current_progress += step_increment
            progress.progress(int(current_progress))

        st.subheader("Starte Haupttraining (GPU-Validierte Modelle)...")
        main_start_time = time.time()
        step_increment = 50 / 2

        if gpu_capable_models_math:
            status_area.text(f"Trainiere Mathematik mit: {gpu_capable_models_math}")
            predictor_math = TabularPredictor(
                label="Mathematik",
                path=f"{model_path_base}_math",
                problem_type="regression"
            ).fit(
                train_data=df_math,
                presets="best_quality",
                included_model_types=list(gpu_capable_models_math),
                refit_full=True,
                set_best_to_refit_full=True,
                verbosity=2,
                **ag_args_gpu
            )
            st.success("✅ Mathematik-Modell erfolgreich trainiert.")
            st.dataframe(predictor_math.leaderboard(silent=True))
        else:
            st.warning("⚠️ Keine GPU-fähigen Modelle für Mathematik verfügbar.")
        current_progress += step_increment
        progress.progress(int(current_progress))

        if gpu_capable_models_raum:
            status_area.text(f"Trainiere Raumvorstellung mit: {gpu_capable_models_raum}")
            predictor_raum = TabularPredictor(
                label="Raumvorstellung",
                path=f"{model_path_base}_raum",
                problem_type="regression"
            ).fit(
                train_data=df_raum,
                presets="best_quality",
                included_model_types=list(gpu_capable_models_raum),
                refit_full=True,
                set_best_to_refit_full=True,
                verbosity=2,
                **ag_args_gpu
            )
            st.success("✅ Raumvorstellungs-Modell erfolgreich trainiert.")
            st.dataframe(predictor_raum.leaderboard(silent=True))
        else:
            st.warning("⚠️ Keine GPU-fähigen Modelle für Raumvorstellung verfügbar.")
        current_progress += step_increment
        progress.progress(100)

        end_time = time.time()
        status_area.success(f"Training abgeschlossen in {int(end_time - start_time)} Sekunden.")
        st.balloons()
else:
    st.info("Bitte lade eine CSV-Datei hoch, um zu starten.")
