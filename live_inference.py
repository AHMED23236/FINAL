"""
live_inference.py
=================
Live prediction pipeline for the kiln predictive-maintenance system.
Uses feature_builder.py (feature engineering) and rule_engine.py (maintenance
recommendations) to produce a full prediction for the current moment.
Public API
----------
    load_model(path)                               -> model
    load_alarm_history(history_dir,
                       base_history_path,
                       days_back=30)              -> pd.DataFrame
    predict_now(model, alarm_history, t=None)      -> dict
    log_prediction(result, log_path)               -> None
"""
import os
import sys
import re
import glob
import json
import pprint
from datetime import datetime, timedelta
import joblib
import numpy as np
import pandas as pd
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from scripts.feature_builder import build_features_for_timestamp, FEATURE_COLUMNS
import rule_engine
_HERE      = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR   = os.path.dirname(_HERE)
_ROOT      = os.path.dirname(_SRC_DIR)
MODEL_PATH   = os.path.join(_SRC_DIR, "models", "kiln_xgb_model.pkl")
HISTORY_DIR  = os.path.join(_ROOT, "data", "daily")
BASE_HISTORY = os.path.join(_ROOT, "data", "cleaned", "Allarmi_TRAIN_READY.csv")
LOG_PATH     = os.path.join(_ROOT, "data", "predictions_log.csv")
_ML_COLS = [c for c in FEATURE_COLUMNS if not c.startswith("latest_alarm_")]
_FTYPE_RULES = [
    (re.compile(r"br[uû]leur|allumage|flamme",     re.I), "burner_failure"),
    (re.compile(r"temp[eé]r|ecart|gradient|therm", re.I), "temperature_failure"),
    (re.compile(r"pression|ventil|filtre|pressostat|soufflerie", re.I), "pression_failure"),
]
_URGENCY_MAP = {
    "burner_failure":       "High",
    "pression_failure":     "High",
    "temperature_failure":  "Medium",
    "alimentation_failure": "Medium",
}
def _infer_failure_type(description: str) -> str:
    for pattern, ftype in _FTYPE_RULES:
        if pattern.search(description):
            return ftype
    return "alimentation_failure"
def _infer_urgency(failure_type: str) -> str:
    return _URGENCY_MAP.get(failure_type, "Medium")
def load_model(path: str):
    """
    Load a scikit-learn-compatible model from *path* using joblib.
    Accepts both .joblib and .pkl files.
    """
    return joblib.load(path)
def load_alarm_history(
    history_dir: str,
    base_history_path: str,
    days_back: int = 30,
) -> pd.DataFrame:
    """
    Build the alarm history DataFrame used by build_features_for_timestamp.
    Strategy
    --------
    1. Load the base history CSV (Allarmi_HISTORY.csv or fallback).
    2. Find daily files (DD-MM-YYYY.csv) inside *history_dir* that fall
       within the last *days_back* days.
    3. For each daily file, extract the '### ALARMS ###' section, map its
       columns to the standard schema, and append.
    4. Sort by Inizio and return.
    Required output columns
    -----------------------
    Inizio, Descrizione, Codice, failure_type, urgency, duration_minutes
    """
    frames = []
    if os.path.exists(base_history_path):
        base = pd.read_csv(base_history_path)
        base["Inizio"] = pd.to_datetime(base["Inizio"], errors="coerce")
        base = base.dropna(subset=["Inizio"])
        frames.append(base)
    cutoff = datetime.now() - timedelta(days=days_back)
    daily_pattern = os.path.join(history_dir, "??-??-????.csv")
    daily_files = sorted(glob.glob(daily_pattern))
    for fpath in daily_files:
        fname = os.path.basename(fpath)
        try:
            file_date = datetime.strptime(fname[:-4], "%d-%m-%Y")
        except ValueError:
            continue
        if file_date < cutoff:
            continue
        chunk = _parse_daily_alarms(fpath, file_date)
        if chunk is not None and not chunk.empty:
            frames.append(chunk)
    if not frames:
        raise RuntimeError(
            "No alarm history data found. "
            f"Checked: {base_history_path!r} and {history_dir!r}"
        )
    df = pd.concat(frames, ignore_index=True)
    df["Inizio"] = pd.to_datetime(df["Inizio"], errors="coerce")
    df = df.dropna(subset=["Inizio"]).sort_values("Inizio").reset_index(drop=True)
    df["Descrizione"] = df["Descrizione"].astype(str).str.strip()
    df["Codice"]      = df["Codice"].astype(str).str.strip()
    return df
def _parse_daily_alarms(fpath: str, file_date: datetime) -> pd.DataFrame | None:
    """
    Extract the '### ALARMS ###' block from a daily CSV and return a DataFrame
    with the standard schema columns.
    Daily ALARMS columns: ID_alarme, Objet, Description, Codice, durée, nbre_occurence
    Daily CSVs have no per-event timestamp — we assign file_date at noon.
    Duration ('durée') is treated as total seconds; dividing by 60 gives minutes.
    """
    try:
        with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return None
    start = None
    for i, line in enumerate(lines):
        if line.strip() == "### ALARMS ###":
            start = i + 1
            break
    if start is None:
        return None
    alarm_lines = []
    for line in lines[start:]:
        if line.startswith("###"):
            break
        alarm_lines.append(line)
    if not alarm_lines:
        return None
    from io import StringIO
    try:
        raw = pd.read_csv(StringIO("".join(alarm_lines)))
    except Exception:
        return None
    raw.columns = raw.columns.str.strip()
    desc_col = next((c for c in raw.columns if c.lower().startswith("desc")), None)
    cod_col  = next((c for c in raw.columns if c.lower() == "codice"), None)
    dur_col  = next((c for c in raw.columns
                     if c.lower() in ("durée", "duree", "durée", "dur")), None)
    if desc_col is None:
        return None
    rows = []
    now = pd.Timestamp(datetime.now())
    base_time = max(
        pd.Timestamp(file_date.replace(hour=0, minute=0, second=0)),
        now - pd.Timedelta(hours=12)
    )
    n_alarms = len(raw)
    for i, (_, r) in enumerate(raw.iterrows()):
        desc     = str(r.get(desc_col, "")).strip()
        codice   = str(r.get(cod_col, "")).strip() if cod_col else ""
        dur_raw  = float(r.get(dur_col, 0)) if dur_col else 0.0
        dur_min  = dur_raw / 60.0
        ftype    = _infer_failure_type(desc)
        urgency  = _infer_urgency(ftype)
        if n_alarms > 1:
            offset_minutes = int(i * (12 * 60) / n_alarms)
        else:
            offset_minutes = 0
        alarm_time = base_time + pd.Timedelta(minutes=offset_minutes)
        rows.append({
            "Inizio":           alarm_time,
            "Descrizione":      desc,
            "Codice":           codice,
            "failure_type":     ftype,
            "urgency":          urgency,
            "duration_minutes": dur_min,
        })
    alarm_time = pd.Timestamp(file_date.replace(hour=12, minute=0, second=0))
    for _, r in raw.iterrows():
        desc     = str(r.get(desc_col, "")).strip()
        codice   = str(r.get(cod_col, "")).strip() if cod_col else ""
        dur_raw  = float(r.get(dur_col, 0)) if dur_col else 0.0
        dur_min  = dur_raw / 60.0
        ftype    = _infer_failure_type(desc)
        urgency  = _infer_urgency(ftype)
        rows.append({
            "Inizio":           alarm_time,
            "Descrizione":      desc,
            "Codice":           codice,
            "failure_type":     ftype,
            "urgency":          urgency,
            "duration_minutes": dur_min,
        })
    return pd.DataFrame(rows) if rows else None
def predict_now(model, alarm_history: pd.DataFrame, t: pd.Timestamp = None) -> dict:
    """
    Produce a live prediction for timestamp *t*.
    Parameters
    ----------
    model         : fitted sklearn-compatible model
    alarm_history : DataFrame from load_alarm_history()
    t             : prediction timestamp; defaults to now() floored to the hour
    Returns
    -------
    dict with keys: timestamp, prediction, probability, recommendation,
                    features_used
    """
    if t is None:
        t = pd.Timestamp(datetime.now()).floor("h")
    features = build_features_for_timestamp(t, alarm_history)
    X = pd.DataFrame([{col: features[col] for col in _ML_COLS}])
    pred  = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])
    rec = rule_engine.get_recommendation(features, pred, proba)
    return {
        "timestamp":      t,
        "prediction":     pred,
        "probability":    proba,
        "recommendation": rec,
        "features_used":  features,
    }
def log_prediction(result: dict, log_path: str = LOG_PATH) -> None:
    """
    Append one row to *log_path* (data/predictions_log.csv).
    Columns written
    ---------------
    timestamp, prediction, probability, alert_level,
    latest_alarm_description, latest_alarm_codice
    """
    feats = result["features_used"]
    rec   = result["recommendation"]
    row = {
        "timestamp":                 result["timestamp"].isoformat(),
        "prediction":                result["prediction"],
        "probability":               round(result["probability"], 4),
        "alert_level":               rec.get("alert_level", ""),
        "latest_alarm_description":  feats.get("latest_alarm_description", ""),
        "latest_alarm_codice":       feats.get("latest_alarm_codice", ""),
    }
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    write_header = not os.path.exists(log_path)
    df_row = pd.DataFrame([row])
    df_row.to_csv(log_path, mode="a", header=write_header, index=False)
if __name__ == "__main__":
    import os as _os
    _ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
    _MODEL_PATH = _os.path.join(_ROOT, MODEL_PATH)
    if not _os.path.exists(_MODEL_PATH):
        _MODEL_PATH = _os.path.join(_ROOT, "src", "models", "kiln_xgb_model.pkl")
    _BASE_HIST = _os.path.join(_ROOT, BASE_HISTORY)
    if not _os.path.exists(_BASE_HIST):
        _BASE_HIST = _os.path.join(_ROOT, "data", "cleaned", "Allarmi_TRAIN_READY.csv")
    _HISTORY_DIR = _os.path.join(_ROOT, HISTORY_DIR)
    _LOG_PATH    = _os.path.join(_ROOT, LOG_PATH)
    print("=" * 60)
    print("  LIVE INFERENCE — self-test")
    print("=" * 60)
    print(f"\n[1] Loading model from: {_MODEL_PATH}")
    model = load_model(_MODEL_PATH)
    print(f"    Model type: {type(model).__name__}")
    print(f"\n[2] Loading alarm history from: {_BASE_HIST}")
    alarm_hist = load_alarm_history(
        history_dir=_HISTORY_DIR,
        base_history_path=_BASE_HIST,
        days_back=30,
    )
    print(f"    History rows: {len(alarm_hist):,}  "
          f"({alarm_hist['Inizio'].min()} → {alarm_hist['Inizio'].max()})")
    print("\n[3] Running predict_now()...")
    result = predict_now(model, alarm_hist)
    print("\n--- Result ---")
    pprint.pprint({
        "timestamp":   str(result["timestamp"]),
        "prediction":  result["prediction"],
        "probability": f"{result['probability']:.1%}",
        "alert_level": result["recommendation"]["alert_level"],
        "headline":    result["recommendation"]["headline"],
        "latest_alarm_description": result["features_used"]["latest_alarm_description"],
    })
    print("\n[4] Logging prediction...")
    log_prediction(result, log_path=_LOG_PATH)
    print(f"    Written to: {_LOG_PATH}")
    print("\nDone.")

