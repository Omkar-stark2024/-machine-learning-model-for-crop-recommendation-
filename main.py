# app_streamlit_predict.py
import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# CONFIG - point this to the joblib file your Flask app produced
ARTIFACT_PATH = "artifacts.joblib"   # change if your file is named differently
FEATURE_NAMES = ['N','P','K','temperature','humidity','ph','rainfall']

# Minimal crop info DB (will attempt to augment from saved df if available)
DEFAULT_CROP_INFO = {
    "rice": {"growth_duration":"3-6 months","pesticides_fertilizers":["Phosphorus","Potassium"],"common_diseases":["Rice blast"],"image":None,"notes":"Requires high water."},
    "wheat": {"growth_duration":"4-5 months","pesticides_fertilizers":["Balanced NPK"],"common_diseases":["Rusts"],"image":None,"notes":"Cool season crop."}
}

# Input key mapping (same as your Flask code)
INPUT_KEY_MAP = {
    'n': 'N', 'nitrogen': 'N', 'N': 'N',
    'p': 'P', 'phosphorus': 'P', 'P': 'P',
    'k': 'K', 'potassium': 'K', 'K': 'K',
    'temperature': 'temperature', 'temp': 'temperature',
    'humidity': 'humidity',
    'ph': 'ph', 'soilph': 'ph',
    'rainfall': 'rainfall', 'rain': 'rainfall'
}

def normalize_payload_keys(payload):
    normalized = {}
    for k, v in payload.items():
        key_lower = k.strip().lower()
        normalized[INPUT_KEY_MAP.get(key_lower, k)] = v
    return normalized

def load_artifacts(path=ARTIFACT_PATH):
    if not os.path.exists(path):
        return None, f"Artifact file not found at: {path}"
    try:
        loaded = joblib.load(path)
    except Exception as e:
        return None, f"Failed to load joblib file: {e}"

    # Support different saved formats:
    # - saved as {'artifacts': artifacts, 'df': df}
    # - saved as artifacts dict directly
    if isinstance(loaded, dict) and 'artifacts' in loaded:
        artifacts = loaded['artifacts']
        df = loaded.get('df')
    else:
        # could be artifacts dict directly or a Pipeline etc.
        artifacts = loaded
        df = None

    # Basic validation
    needed = ['scaler','models'] if 'models' in artifacts else ['scaler','dt','nb']
    # we'll not strictly enforce; just return artifacts
    return {'artifacts': artifacts, 'df': df}, None

def prepare_crop_info(artifacts_wrapper):
    # load default and augment with dataset labels if available
    crop_info = dict(DEFAULT_CROP_INFO)
    df = artifacts_wrapper.get('df')
    if df is not None and 'label' in df.columns:
        for crop in df['label'].unique():
            key = str(crop).lower()
            if key not in crop_info:
                crop_info[key] = {
                    "growth_duration":"Varies",
                    "pesticides_fertilizers":["Local recommendation needed"],
                    "common_diseases":["Local recommendation needed"],
                    "image":None,
                    "notes":"Placeholder info"
                }
    return crop_info

def predict_from_artifacts(artifacts, sample_raw, model_choice=None):
    """
    artifacts: the artifacts dict produced by your training pipeline
    sample_raw: 2D numpy array shaped (1, n_features) or DataFrame with columns FEATURE_NAMES
    model_choice: key of a model in artifacts['models'] if available, or 'dt'/'nb' if old format
    returns: top_label (str), list of {'crop':..., 'prob':...}
    """
    # Accept DataFrame or ndarray. Convert to DataFrame to preserve feature names.
    if isinstance(sample_raw, np.ndarray):
        sample_df = pd.DataFrame(sample_raw, columns=artifacts['feature_names'] if 'feature_names' in artifacts else FEATURE_NAMES)
    elif isinstance(sample_raw, pd.DataFrame):
        sample_df = sample_raw.copy()
    else:
        raise ValueError("sample_raw must be numpy array or pandas DataFrame")

    # scale using fitted scaler, avoid sklearn warning by passing DataFrame with same columns
    scaler = artifacts.get('scaler') or artifacts.get('scaler_')
    if scaler is None:
        raise ValueError("No scaler found in artifacts")
    sample_scaled = scaler.transform(sample_df)

    # Choose model API: new format uses 'models' dict; older used 'dt' and 'nb'
    models_dict = artifacts.get('models')
    if models_dict:
        # if no model_choice provided, prefer 'VotingEnsemble', else first model
        if model_choice is None:
            model_choice = 'VotingEnsemble' if 'VotingEnsemble' in models_dict else sorted(models_dict.keys())[0]
        model = models_dict.get(model_choice)
        if model is None:
            raise ValueError(f"Model '{model_choice}' not found. Available: {list(models_dict.keys())}")
        # compute probabilities if possible
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(sample_scaled)[0]
            # model.classes_ are encoded label ints OR label encoder indices depending on training
            try:
                classes_enc = model.classes_
            except Exception:
                classes_enc = None
            # If artifacts contains a label_encoder, map encoded labels back
            if 'label_encoder' in artifacts:
                le = artifacts['label_encoder']
                # If classes_enc are ints that correspond to le.transform outputs
                try:
                    labels = [str(le.inverse_transform([int(c)])[0]) for c in classes_enc]
                except Exception:
                    # fallback: attempt decode from range
                    labels = [str(le.inverse_transform([i])[0]) for i in range(len(probs))]
            else:
                # fallback: if artifacts has 'classes_dt' or 'classes_nb' use them, else guess strings
                classes_dt = artifacts.get('classes_dt') or artifacts.get('classes_nb')
                if classes_dt and len(classes_dt) == len(probs):
                    labels = [str(c) for c in classes_dt]
                elif classes_enc is not None and len(classes_enc) == len(probs):
                    labels = [str(c) for c in classes_enc]
                else:
                    labels = [str(i) for i in range(len(probs))]
            paired = [{'crop': lab, 'prob': float(p)} for lab,p in zip(labels, probs)]
        else:
            pred_enc = model.predict(sample_scaled)[0]
            # decode pred_enc similarly
            if 'label_encoder' in artifacts:
                le = artifacts['label_encoder']
                try:
                    crop = str(le.inverse_transform([int(pred_enc)])[0])
                except Exception:
                    crop = str(pred_enc)
            else:
                crop = str(pred_enc)
            paired = [{'crop': crop, 'prob': 1.0}]
    else:
        # fallback to older format where 'dt' and 'nb' are top-level
        dt = artifacts.get('dt')
        nb = artifacts.get('nb')
        if model_choice is None:
            model_choice = 'nb'
        if model_choice == 'dt' and dt is not None:
            model = dt
        elif model_choice == 'nb' and nb is not None:
            model = nb
        else:
            # average probs between dt and nb if both exist
            if dt is not None and nb is not None:
                probs_dt = dt.predict_proba(sample_scaled)[0]
                probs_nb = nb.predict_proba(sample_scaled)[0]
                # assume classes are the same ordering (best-effort)
                classes = list(dt.classes_)
                avg = (probs_dt + probs_nb) / 2.0
                paired = [{'crop': str(c), 'prob': float(p)} for c,p in zip(classes, avg)]
            else:
                raise ValueError("No usable models found in artifacts")
        # if model was set
        if 'paired' not in locals():
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(sample_scaled)[0]
                classes = list(model.classes_)
                paired = [{'crop': str(c), 'prob': float(p)} for c,p in zip(classes, probs)]
            else:
                pred = model.predict(sample_scaled)[0]
                paired = [{'crop': str(pred), 'prob': 1.0}]

    paired_sorted = sorted(paired, key=lambda x: x['prob'], reverse=True)
    top = paired_sorted[0]['crop'] if paired_sorted else None
    return top, paired_sorted

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Crop Recommendation - Predict Only", layout="centered")
st.title("Crop Recommendation — Prediction (use existing model only)")

st.markdown("""
This UI **loads your saved artifacts** (no training).  
Put the `artifacts.joblib` produced by your Flask app in the same folder as this script (or change `ARTIFACT_PATH`), then provide feature inputs and press **Predict**.
""")

# show artifact load status
artifacts_wrapper, err = load_artifacts(ARTIFACT_PATH)
if err:
    st.error(err)
    st.stop()

artifacts = artifacts_wrapper['artifacts']
df_saved = artifacts_wrapper.get('df')
CROP_INFO = prepare_crop_info(artifacts_wrapper)

st.sidebar.header("Artifact details")
st.sidebar.write("Available keys in artifacts:")
st.sidebar.write(list(artifacts.keys()))
if df_saved is not None:
    st.sidebar.write("Saved dataset sample:")
    try:
        st.sidebar.dataframe(df_saved.head())
    except Exception:
        pass

# Model selection
models_list = []
if 'models' in artifacts:
    models_list = sorted(list(artifacts['models'].keys()))
else:
    # fallback possible keys
    if 'dt' in artifacts: models_list.append('dt')
    if 'nb' in artifacts: models_list.append('nb')

if not models_list:
    st.warning("No models found inside artifacts. Check your joblib file structure.")
    st.stop()

model_choice = st.selectbox("Choose model to use for prediction", models_list)

st.markdown("### Input features")
cols = st.columns(3)
defaults = {c: 0.0 for c in FEATURE_NAMES}
# If saved df present, use first row as default
if df_saved is not None:
    try:
        for c in FEATURE_NAMES:
            defaults[c] = float(df_saved[c].iloc[0])
    except Exception:
        pass

with cols[0]:
    N = st.number_input("N (Nitrogen)", value=float(defaults['N']))
    P = st.number_input("P (Phosphorus)", value=float(defaults['P']))
    K = st.number_input("K (Potassium)", value=float(defaults['K']))
with cols[1]:
    temperature = st.number_input("Temperature (°C)", value=float(defaults['temperature']))
    humidity = st.number_input("Humidity (%)", value=float(defaults['humidity']))
with cols[2]:
    ph = st.number_input("pH", value=float(defaults['ph']))
    rainfall = st.number_input("Rainfall (mm)", value=float(defaults['rainfall']))

if st.button("Predict crop"):
    # build sample as DataFrame with correct columns
    sample = pd.DataFrame([[N,P,K,temperature,humidity,ph,rainfall]], columns=artifacts.get('feature_names', FEATURE_NAMES))
    try:
        top, probs_sorted = predict_from_artifacts(artifacts, sample, model_choice=model_choice)
        st.success(f"Predicted crop: **{top}**")
        st.write("Top probabilities:")
        st.table(pd.DataFrame(probs_sorted).head(10))
        info = CROP_INFO.get(str(top).lower())
        if info:
            st.markdown("**Crop info:**")
            st.json(info)
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption(f"Artifact loaded from: {os.path.abspath(ARTIFACT_PATH)}")
