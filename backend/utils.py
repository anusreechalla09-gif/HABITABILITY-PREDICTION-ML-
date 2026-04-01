import os
import json
import random
import pickle
import joblib
import zipfile
import urllib.request
import numpy as np
import pandas as pd

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "models", "xgboost.pkl")
CSV_PATH      = os.path.join(BASE_DIR, "data", "habitability_ranked.csv")
SAMPLES_PATH  = os.path.join(BASE_DIR, "data", "sample_planets.json")
FEATURES_PATH = os.path.join(BASE_DIR, "data", "feature_cols.json")

# ── Download model from Google Drive if not present ───────────────────────────
GDRIVE_FILE_ID = "1CfD0hebCkQtQLOX7g6w1Z9xMJ8t-Wgok"   # ← replace this

def download_model():
    if os.path.exists(MODEL_PATH):
        print("Model already exists, skipping download")
        return
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print("Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
    try:
        urllib.request.urlretrieve(url, MODEL_PATH)
        size = os.path.getsize(MODEL_PATH)
        print(f"Model downloaded successfully ({size} bytes)")
        if size < 1000:
            os.remove(MODEL_PATH)
            print("ERROR: Downloaded file too small - check Google Drive sharing settings")
    except Exception as e:
        print(f"Download failed: {e}")

download_model()

# ── Auto-unzip CSV if not present ─────────────────────────────────────────────
_csv_zip = os.path.join(BASE_DIR, "data", "habitability_ranked.zip")
if not os.path.exists(CSV_PATH) and os.path.exists(_csv_zip):
    print("Unzipping CSV...")
    with zipfile.ZipFile(_csv_zip, "r") as z:
        z.extractall(os.path.join(BASE_DIR, "data"))
    print("CSV unzipped successfully")

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
    pl_rade, pl_bmasse, pl_orbper, pl_orbsmax,
    pl_eqt, pl_dens, st_teff, st_lum, st_met,
    st_spectype="G", stellar_temp_score=None,
    stellar_lum_score=None, stellar_compatibility_index=None,
    orbital_stability_factor=None,
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
        "pl_rade": float(pl_rade),
        "pl_bmasse": float(pl_bmasse),
        "pl_orbper": float(pl_orbper),
        "pl_orbsmax": float(pl_orbsmax),
        "pl_eqt": float(pl_eqt),
        "pl_dens": float(pl_dens),
        "st_teff": float(st_teff),
        "st_lum": float(st_lum),
        "st_met": float(st_met),
        "stellar_temp_score": float(stellar_temp_score),
        "stellar_lum_score": float(stellar_lum_score),
        "stellar_compatibility_index": float(stellar_compatibility_index),
        "orbital_stability_factor": float(orbital_stability_factor),
    }

    spec_cols = get_spec_cols()
    ohe_dict = {col: 0 for col in spec_cols}
    spec_key = "st_spectype_" + st_spectype.strip()
    if spec_key in ohe_dict:
        ohe_dict[spec_key] = 1
    else:
        for col in spec_cols:
            if col.startswith("st_spectype_" + st_spectype.strip()):
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
        pl_rade=_f("pl_rade"),
        pl_bmasse=_f("pl_bmasse"),
        pl_orbper=_f("pl_orbper"),
        pl_orbsmax=_f("pl_orbsmax"),
        pl_eqt=_f("pl_eqt"),
        pl_dens=_f("pl_dens"),
        st_teff=_f("st_teff"),
        st_lum=_f("st_lum"),
        st_met=_f("st_met"),
        st_spectype=str(input_dict.get("st_spectype", "G")).strip(),
        stellar_temp_score=_f("stellar_temp_score") if "stellar_temp_score" in input_dict else None,
        stellar_lum_score=_f("stellar_lum_score") if "stellar_lum_score" in input_dict else None,
        stellar_compatibility_index=_f("stellar_compatibility_index") if "stellar_compatibility_index" in input_dict else None,
        orbital_stability_factor=_f("orbital_stability_factor") if "orbital_stability_factor" in input_dict else None,
    )

    model = get_model()
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1]) * 100

    if probability >= 75:
        confidence = "High"
    elif probability >= 45:
        confidence = "Moderate"
    else:
        confidence = "Low"

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
    if samples:
        return random.choice(samples)
    return _sample_from_csv(1)[0]


def _sample_from_csv(n):
    if not os.path.exists(CSV_PATH):
        return []
    df = pd.read_csv(CSV_PATH, nrows=500)
    spec_cols = [c for c in df.columns if c.startswith("st_spectype_")]
    base_cols = ["pl_rade", "pl_bmasse", "pl_orbper", "pl_orbsmax",
                 "pl_eqt", "pl_dens", "st_teff", "st_lum", "st_met"]
    results = []
    for _, row in df.sample(min(n, len(df)), random_state=42).iterrows():
        active = [c.replace("st_spectype_", "") for c in spec_cols if row.get(c, False)]
        rec = {k: round(float(row[k]), 4) for k in base_cols}
        rec["st_spectype"] = active[0] if active else "G"
        rec["habitability_probability"] = round(float(row.get("habitability_probability", 0)) * 100, 1)
        rec["habitability_binary"] = int(row.get("habitability_binary", 0))
        results.append(rec)
    return results


def get_top_planets(n=10):
    if not os.path.exists(CSV_PATH):
        return []
    df = pd.read_csv(CSV_PATH, nrows=5000)
    spec_cols = [c for c in df.columns if c.startswith("st_spectype_")]
    top = df.nlargest(n, "habitability_probability")
    results = []
    for i, (_, row) in enumerate(top.iterrows(), 1):
        active = [c.replace("st_spectype_", "") for c in spec_cols if row.get(c, False)]
        results.append({
            "rank": i,
            "planet_id": "EXO-" + str(i).zfill(4),
            "st_spectype": active[0] if active else "Unknown",
            "pl_rade": round(float(row["pl_rade"]), 3),
            "pl_bmasse": round(float(row["pl_bmasse"]), 3),
            "pl_orbper": round(float(row["pl_orbper"]), 3),
            "pl_orbsmax": round(float(row["pl_orbsmax"]), 3),
            "pl_eqt": round(float(row["pl_eqt"]), 3),
            "st_teff": round(float(row["st_teff"]), 3),
            "habitability_probability": round(float(row["habitability_probability"]) * 100, 2),
            "habitable": int(row["habitability_binary"]),
        })
    return results


REQUIRED_FIELDS = [
    "pl_rade", "pl_bmasse", "pl_orbper", "pl_orbsmax",
    "pl_eqt", "pl_dens", "st_teff", "st_lum", "st_met", "st_spectype"
]


def validate_input(data):
    for field in REQUIRED_FIELDS:
        if field not in data:
            return False, "Missing required field: " + field
    for field in REQUIRED_FIELDS[:-1]:
        try:
            float(data[field])
        except (TypeError, ValueError):
            return False, "Field " + field + " must be a valid number."
    if not isinstance(data.get("st_spectype"), str) or not data["st_spectype"].strip():
        return False, "Field st_spectype must be a non-empty string."
    return True, ""


def validate_batch(rows):
    if not isinstance(rows, list) or len(rows) == 0:
        return False, "Batch must be a non-empty list."
    if len(rows) > 500:
        return False, "Batch size cannot exceed 500 rows."
    for i, row in enumerate(rows):
        ok, msg = validate_input(row)
        if not ok:
            return False, "Row " + str(i + 1) + ": " + msg
    return True, ""
