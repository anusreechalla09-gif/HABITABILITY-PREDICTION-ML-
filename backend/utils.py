"""
utils.py — ExoHabitAI Backend Utilities
"""

import os
import json
import random
import pickle
import joblib
import zipfile
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "models", "xgboost.pkl")
CSV_PATH      = os.path.join(BASE_DIR, "data", "habitability_ranked.csv")
SAMPLES_PATH  = os.path.join(BASE_DIR, "data", "sample_planets.json")
FEATURES_PATH = os.path.join(BASE_DIR, "data", "feature_cols.json")

# ── Auto-unzip model ───────────────────────────────────────────────────────────
_model_zip = os.path.join(BASE_DIR, "models", "xgboost.zip")
if not os.path.exists(MODEL_PATH) and os.path.exists(_model_zip):
    print("⬇️  Unzipping model...")
    with zipfile.ZipFile(_model_zip, "r") as z:
        z.extractall(os.path.join(BASE_DIR, "models"))
    print("✅ Model unzipped successfully")

# ── Auto-unzip CSV ─────────────────────────────────────────────────────────────
_csv_zip = os.path.join(BASE_DIR, "data", "habitability_ranked.zip")
if not os.path.exists(CSV_PATH) and os.path.exists(_csv_zip):
    print("⬇️  Unzipping CSV dataset...")
    with zipfile.ZipFile(_csv_zip, "r") as z:
        z.extractall(os.path.join(BASE_DIR, "data"))
    print("✅ CSV unzipped successfully")

# ── Singleton model loader ─────────────────────────────────────────────────────
_model        = None
_feature_cols = None
_spec_cols    = None

def get_model():
    global _model
    if _model is None:
        try:
            _model = joblib.load(MODEL_PATH)
        except Exception:
            with open(MODEL_PATH, "rb") as f:
                _model = pickle.load(f)
    return _model


def get_feature_cols():
    global _feature_cols
    if _feature_cols is None:
        if os.path.exists(FEATURES_PATH):
            with open(FEATURES_PATH) as f:
                _feature_cols = json.load(f)
        else:
            model = get_model()
            _feature_cols = model.get_booster().feature_names
    return _feature_cols


def get_spec_cols():
    global _spec_cols
    if _spec_cols is None:
        _spec_cols = [c for c in get_feature_cols() if c.startswith("st_spectype_")]
    return _spec_cols


def build_feature_vector(
    pl_rade, pl_bmasse, pl_orbper, pl_orbsmax, pl_eqt, pl_dens,
    st_teff, st_lum, st_met, st_spectype="G",
    stellar_temp_score=None, stellar_lum_score=None,
    stellar_compatibility_index=None, orbital_stability_factor=None,
):
    if stellar_temp_score is None:
        stellar_temp_score = -abs(st_teff) * 0.15
    if stellar_lum_score is None:
        stellar_lum_score = -abs(st_lum) * 0.3
    if stellar_compatibility_index is None:
        stellar_compatibility_index = 0.6 * stellar_temp_score + 0.4 * stellar_lum_score
    if orbital_stability_factor is None:
        orbital_stability_factor = (pl_orbsmax - pl_orbper) * 0.1

    row = {
        "pl_rade": float(pl_rade), "pl_bmasse": float(pl_bmasse),
        "pl_orbper": float(pl_orbper), "pl_orbsmax": float(pl_orbsmax),
        "pl_eqt": float(pl_eqt), "pl_dens": float(pl_dens),
        "st_teff": float(st_teff), "st_lum": float(st_lum), "st_met": float(st_met),
        "stellar_temp_score": float(stellar_temp_score),
        "stellar_lum_score": float(stellar_lum_score),
        "stellar_compatibility_index": float(stellar_compatibility_index),
        "orbital_stability_factor": float(orbital_stability_factor),
    }

    spec_cols = get_spec_cols()
    ohe_dict  = {col: 0 for col in spec_cols}
    spec_key  = f"st_spectype_{st_spectype.strip()}"
    if spec_key in ohe_dict:
        ohe_dict[spec_key] = 1
    else:
        for col in spec_cols:
            if col.startswith(f"st_spectype_{st_spectype.strip()}"):
                ohe_dict[col] = 1
                break
    row.update(ohe_dict)

    feature_cols = get_feature_cols()
    df = pd.DataFrame([row])
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df[feature_cols]


def predict(input_dict):
    def _f(k, default=0.0):
        try:
            return float(input_dict.get(k, default))
        except (TypeError, ValueError):
            return default

    df = build_feature_vector(
        pl_rade=_f("pl_rade"), pl_bmasse=_f("pl_bmasse"),
        pl_orbper=_f("pl_orbper"), pl_orbsmax=_f("pl_orbsmax"),
        pl_eqt=_f("pl_eqt"), pl_dens=_f("pl_dens"),
        st_teff=_f("st_teff"), st_lum=_f("st_lum"), st_met=_f("st_met"),
        st_spectype=str(input_dict.get("st_spectype", "G")).strip(),
        stellar_temp_score=_f("stellar_temp_score") if "stellar_temp_score" in input_dict else None,
        stellar_lum_score=_f("stellar_lum_score") if "stellar_lum_score" in input_dict else None,
        stellar_compatibility_index=_f("stellar_compatibility_index") if "stellar_compatibility_index" in input_dict else None,
        orbital_stability_factor=_f("orbital_stability_factor") if "orbital_stability_factor" in input_dict else None,
    )

    model      = get_model()
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1]) * 100
    confidence  = "High" if probability >= 75 else "Moderate" if probability >= 45 else "Low"

    return {
        "habitable": prediction,
        "probability": round(probability, 2),
        "label": "Potentially Habitable" if prediction == 1 else "Not Habitable",
        "confidence_band": confidence,
        "status": "success",
    }


def get_sample_planets(n=12):
    if os.path.exists(SAMPLES_PATH):
        with open(SAMPLES_PATH) as f:
            return json.load(f)[:n]
    return _sample_from_csv(n)


def get_random_sample():
    samples = get_sample_planets(12)
    return random.choice(samples) if samples else _sample_from_csv(1)[0]


def _sample_from_csv(n):
    if not os.path.exists(CSV_PATH):
        return []
    df        = pd.read_csv(CSV_PATH)
    spec_cols = [c for c in df.columns if c.startswith("st_spectype_")]
    base_cols = ["pl_rade","pl_bmasse","pl_orbper","pl_orbsmax","pl_eqt","pl_dens","st_teff","st_lum","st_met"]
    results   = []
    for _, row in df.sample(min(n, len(df)), random_state=42).iterrows():
 active = [c.replace("st_spectype_", "") for c in spec_cols if row.get(c, False)]
