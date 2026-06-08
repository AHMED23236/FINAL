"""
feature_builder.py
==================
Reusable module that computes the feature vector for a single prediction
timestamp from alarm history.  Logic is identical to feature_engineering.py v2
so there is no train/serve skew.
Public API
----------
    build_features_for_timestamp(t, alarm_history) -> dict
"""
import numpy as np
import pandas as pd
_ONE_H = np.timedelta64(1, "h")
_SIX_H = np.timedelta64(6, "h")
_DAY   = np.timedelta64(24, "h")
_FAILURE_TYPES = [
    "temperature_failure",
    "burner_failure",
    "pression_failure",
    "alimentation_failure",
]
FEATURE_COLUMNS = [
    "hour", "dayofweek", "is_weekend", "is_night", "is_peak_hours",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "alarms_last_1h", "alarms_last_6h", "alarms_last_24h",
    "high_last_6h", "medium_last_6h", "low_last_6h", "high_last_24h",
    "temp_fail_last_6h", "burner_fail_last_6h",
    "pression_fail_last_6h", "alim_fail_last_6h",
    "mean_duration_6h", "max_duration_6h",
    "unique_zones_24h",
    "minutes_since_last_alarm", "minutes_since_last_high",
    "latest_alarm_description", "latest_alarm_codice",
    "latest_alarm_failure_type", "latest_alarm_urgency",
]
def build_features_for_timestamp(
    t: pd.Timestamp,
    alarm_history: pd.DataFrame,
) -> dict:
    """
    Build one feature row for timestamp *t*, using only alarms strictly
    before *t*.
    Parameters
    ----------
    t : pd.Timestamp
        The prediction point.
    alarm_history : pd.DataFrame
        Must contain at least: Inizio, Descrizione, Codice,
        failure_type, urgency, duration_minutes.
    Returns
    -------
    dict
        Keys in exactly the order of FEATURE_COLUMNS:
        25 numeric ML features + 4 latest_alarm_* context fields.
    """
    times    = alarm_history["Inizio"].values
    failure  = alarm_history["failure_type"].values
    urgency  = alarm_history["urgency"].values
    duration = alarm_history["duration_minutes"].fillna(0).values
    descs    = alarm_history["Descrizione"].astype(str).str.strip().values
    codices  = alarm_history["Codice"].astype(str).str.strip().values
    zone = (
        alarm_history["Descrizione"]
        .str.extract(r"Zona:\s*S(\d+)", expand=False)
        .astype(float)
        .fillna(-1)
        .values
    )
    t_np = np.datetime64(t)
    high_mask   = urgency == "High"
    medium_mask = urgency == "Medium"
    low_mask    = urgency == "Low"
    ftype_masks = {f: (failure == f) for f in _FAILURE_TYPES}
    past_any   = times < t_np
    past_1h    = (times >= t_np - _ONE_H) & (times < t_np)
    past_6h    = (times >= t_np - _SIX_H) & (times < t_np)
    past_24h   = (times >= t_np - _DAY)   & (times < t_np)
    if past_any.any():
        last_idx      = np.where(past_any)[0][-1]
        latest_desc   = descs[last_idx]
        latest_codice = codices[last_idx]
        latest_ftype  = failure[last_idx]
        latest_urg    = urgency[last_idx]
    else:
        latest_desc = latest_codice = latest_ftype = latest_urg = ""
    if past_any.any():
        minutes_last = float(
            (t_np - times[past_any][-1]) / np.timedelta64(1, "m")
        )
    else:
        minutes_last = 1440.0
    high_before = (times < t_np) & high_mask
    if high_before.any():
        minutes_high = float(
            (t_np - times[high_before][-1]) / np.timedelta64(1, "m")
        )
    else:
        minutes_high = 1440.0
    minutes_last = min(minutes_last, 1440.0)
    minutes_high = min(minutes_high, 1440.0)
    dur_6h = duration[past_6h]
    mean_dur = float(dur_6h.mean()) if past_6h.any() else 0.0
    max_dur  = float(dur_6h.max())  if past_6h.any() else 0.0
    zones_24h = alarm_zone_unique_24h(zone, past_24h)
    return {
        "hour":         t.hour,
        "dayofweek":    t.dayofweek,
        "is_weekend":   int(t.dayofweek >= 5),
        "is_night":     int(t.hour < 6 or t.hour >= 22),
        "is_peak_hours": int(8 <= t.hour <= 18),
        "hour_sin": np.sin(2 * np.pi * t.hour / 24),
        "hour_cos": np.cos(2 * np.pi * t.hour / 24),
        "dow_sin":  np.sin(2 * np.pi * t.dayofweek / 7),
        "dow_cos":  np.cos(2 * np.pi * t.dayofweek / 7),
        "alarms_last_1h":  int(past_1h.sum()),
        "alarms_last_6h":  int(past_6h.sum()),
        "alarms_last_24h": int(past_24h.sum()),
        "high_last_6h":   int((past_6h & high_mask).sum()),
        "medium_last_6h": int((past_6h & medium_mask).sum()),
        "low_last_6h":    int((past_6h & low_mask).sum()),
        "high_last_24h":  int((past_24h & high_mask).sum()),
        "temp_fail_last_6h":     int((past_6h & ftype_masks["temperature_failure"]).sum()),
        "burner_fail_last_6h":   int((past_6h & ftype_masks["burner_failure"]).sum()),
        "pression_fail_last_6h": int((past_6h & ftype_masks["pression_failure"]).sum()),
        "alim_fail_last_6h":     int((past_6h & ftype_masks["alimentation_failure"]).sum()),
        "mean_duration_6h": mean_dur,
        "max_duration_6h":  max_dur,
        "unique_zones_24h": zones_24h,
        "minutes_since_last_alarm": minutes_last,
        "minutes_since_last_high":  minutes_high,
        "latest_alarm_description":  latest_desc,
        "latest_alarm_codice":       latest_codice,
        "latest_alarm_failure_type": latest_ftype,
        "latest_alarm_urgency":      latest_urg,
    }
def alarm_zone_unique_24h(zone: np.ndarray, past_24h: np.ndarray) -> int:
    """Count unique valid zones (zone >= 0) in the past-24h window."""
    if not past_24h.any():
        return 0
    valid = zone[past_24h & (zone >= 0)]
    return int(np.unique(valid).size)
if __name__ == "__main__":
    import os
    TRAIN_PATH   = os.path.join(
        os.path.dirname(__file__),
        "../../data/cleaned/Allarmi_TRAIN_READY.csv",
    )
    FEATURES_PATH = os.path.join(
        os.path.dirname(__file__),
        "../data/features/Allarmi_FEATURES.csv",
    )
    ML_COLS = [c for c in FEATURE_COLUMNS if not c.startswith("latest_alarm_")]
    RTOL = 1e-6
    print("Loading alarm history...")
    alarm_hist = pd.read_csv(TRAIN_PATH)
    alarm_hist["Inizio"] = pd.to_datetime(alarm_hist["Inizio"], errors="coerce")
    alarm_hist = (
        alarm_hist.dropna(subset=["Inizio"])
        .sort_values("Inizio")
        .reset_index(drop=True)
    )
    print("Loading reference features...")
    feat_ref = pd.read_csv(FEATURES_PATH)
    feat_ref["timestamp"] = pd.to_datetime(feat_ref["timestamp"])
    n = len(feat_ref)
    pick_indices = [0, n // 2, n - 1]
    test_rows = feat_ref.iloc[pick_indices].reset_index(drop=True)
    print(f"\nRunning {len(test_rows)} test cases...\n")
    all_pass = True
    for i, ref_row in test_rows.iterrows():
        t = ref_row["timestamp"]
        computed = build_features_for_timestamp(t, alarm_hist)
        failures = []
        for col in ML_COLS:
            ref_val  = ref_row[col]
            comp_val = computed[col]
            if isinstance(ref_val, float) or isinstance(comp_val, float):
                ok = abs(float(comp_val) - float(ref_val)) <= RTOL * max(1.0, abs(float(ref_val)))
            else:
                ok = (comp_val == ref_val)
            if not ok:
                failures.append(f"  {col}: expected={ref_val!r}  got={comp_val!r}")
        status = "PASS" if not failures else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"[{status}] timestamp={t}")
        for msg in failures:
            print(msg)
    print()
    print("Overall:", "PASS" if all_pass else "FAIL")

