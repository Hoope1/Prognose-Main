import streamlit as st
import shutil
import os
from datetime import datetime

st.set_page_config(page_title="PrognoseTrainer – Modelle verwalten", layout="wide")
st.title("Modellverwaltung – Laden, Löschen, Backup")

model_path_math = "models/model_mathematik"
model_path_raum = "models/model_raumvorstellung"
backup_path = "models/_backups/"

def model_info(path):
    if not os.path.exists(path):
        return "❌ Nicht vorhanden"
    size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, filenames in os.walk(path) for f in filenames)
    size_mb = size / 1024 / 1024
    last_modified = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
    return f"✅ {round(size_mb, 1)} MB – geändert am {last_modified}"

def backup_model(src, name):
    os.makedirs(backup_path, exist_ok=True)
    dst = os.path.join(backup_path, f"{name}_backup")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

def restore_model(name):
    src = os.path.join(backup_path, f"{name}_backup")
    dst = f"models/model_{name}"
    if os.path.exists(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

def delete_model(path):
    if os.path.exists(path):
        shutil.rmtree(path)

st.markdown("### Modellstatus")
col1, col2 = st.columns(2)
with col1:
    st.write("**Mathematik-Modell:**")
    st.info(model_info(model_path_math))
with col2:
    st.write("**Raumvorstellungs-Modell:**")
    st.info(model_info(model_path_raum))

st.markdown("---")
st.subheader("Aktionen")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Backup beider Modelle"):
        if os.path.exists(model_path_math):
            backup_model(model_path_math, "mathematik")
        if os.path.exists(model_path_raum):
            backup_model(model_path_raum, "raumvorstellung")
        st.success("Backup erstellt.")

with col2:
    if st.button("Backup wiederherstellen"):
        restore_model("mathematik")
        restore_model("raumvorstellung")
        st.success("Modelle wiederhergestellt.")

with col3:
    if st.button("Modelle löschen"):
        delete_model(model_path_math)
        delete_model(model_path_raum)
        st.warning("Modelle gelöscht.")

