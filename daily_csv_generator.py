#!/usr/bin/env python3
"""
daily_csv_generator.py
Génère le fichier CSV journalier SCADA (ALARMS + TEMPERATURE + VELOCITA)
dans cjo maintenance/cjo-maintenance-data-cleaned/daily/

Usage:
    python daily_csv_generator.py              # génère aujourd'hui
    python daily_csv_generator.py 14-05-2026   # génère un jour précis
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ─── OUTPUT DIR : dossier daily du projet cjo maintenance ─────────────────────
_HERE      = Path(__file__).parent
OUTPUT_DIR = str(_HERE / "cjo maintenance" / "cjo-maintenance-data-cleaned" / "daily")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TEMP_MEAN                 = 70.0
TEMP_NOISE                = 3.0
TEMP_SPIKE_PROB           = 0.02
TEMP_SPIKE_HIGH           = 88.0

SPEED_MEAN                = 1300.0
SPEED_NOISE               = 50.0

NUM_ALARMS_RANGE          = (20, 40)
SAMPLING_INTERVAL_MINUTES = 1
CURVA_VALUE               = 5.0

# ─── COLONNES ─────────────────────────────────────────────────────────────────
ALARM_COLS = [
    "ID_alarme", "Objet", "Description ", "Codice", "durée", "nbre_occurence",
]

_T_SENSORS = [f"Temp{i}"  for i in range(1, 61)]
_T_SETS    = [f"SetT{i}"  for i in range(1, 61)]
_T_PERCS   = [f"PercT{i}" for i in range(1, 61)]
TEMP_COLS  = ["Data", "Minuti", "Curva"] + _T_SENSORS + _T_SETS + _T_PERCS + ["ora", "timestamp"]

VEL_COLS = [
    "Data", "Minuti", "Curva",
    "Vel1",  "SetV1",  "Vel2",  "SetV2",  "Vel3",  "SetV3",
    "Vel4",  "SetV4",  "Vel5",  "SetV5",  "Vel6",  "SetV6",
    "Vel7",  "SetV62", "Vel8",  "SetV63", "Vel9",  "SetV64",
    "Vel10", "SetV65", "Vel11", "SetV66", "Vel12", "SetV67",
    "Vel13", "SetV68", "Vel14", "SetV69", "Vel15", "SetV610",
    "Vel16", "SetV611","Vel17", "SetV612","Vel18", "SetV613",
    "Vel19", "SetV614","Vel20", "SetV615","Vel21", "SetV616",
    "Vel22", "SetV617",
    "SetV618", "SetV619", "SetV620", "SetV621",
]

# ─── CATALOGUE D'ALARMES ──────────────────────────────────────────────────────
ALARM_CATALOGUE = [
    (15,  "Four ", "Conditionneur Tableau Electrique non OK ", "200.10 "),
    (80,  "Four ", "Décl. Phtocel. sur Plan Rouleaux ",        "202.06 "),
    (106, "Four ", "Brûleurs zone 2 côté extraction ",         "209.01 "),
    (107, "Four ", "Brûleurs zone 2 côté motorisé ",           "209.00 "),
    (111, "Four ", "Brûleurs zone 4 côté motorisé ",           "209.04 "),
    (216, "Four ", "Oscillation Entraînements ",               "209.10 "),
    (226, "Four ", "Gradient positif de Tempér. Zone: S23 ",   "210.00 "),
    (227, "Four ", "Gradient négatif de Tempér. Zone: S23 ",   "210.01 "),
    (228, "Four ", "Ecart max. de Tempér. Zone: K27 ",         "210.02 "),
    (228, "Four ", "Ecart max. de Tempér. Zone: KE1 ",         "210.02 "),
    (228, "Four ", "Ecart max. de Tempér. Zone: S18 ",         "210.02 "),
    (229, "Four ", "Ecart min. de Tempér. Zone: K2 ",          "210.03 "),
    (229, "Four ", "Ecart min. de Tempér. Zone: S20 ",         "210.03 "),
    (231, "Four ", "Gradient positif: P-RR ",                  "210.04 "),
    (232, "Four ", "Gradient négatif: P-RR ",                  "210.05 "),
]

# ─── SIMULATION ───────────────────────────────────────────────────────────────
def _timestamps(date, n):
    return [date + timedelta(minutes=i * SAMPLING_INTERVAL_MINUTES) for i in range(n)]

def _simulate_temperatures(n, rng):
    t = np.linspace(0, 2 * np.pi, n)
    drift   = 5.0 * np.sin(t)
    offsets = rng.uniform(-4.0, 4.0, size=60)
    base    = TEMP_MEAN + drift[:, None] + offsets[None, :]
    noise   = rng.normal(0.0, TEMP_NOISE, size=(n, 60))
    temps   = base + noise
    spikes  = np.where(rng.random(size=n) < TEMP_SPIKE_PROB)[0]
    for idx in spikes:
        sensors = rng.choice(60, size=int(rng.integers(1, 6)), replace=False)
        temps[idx, sensors] = rng.uniform(TEMP_SPIKE_HIGH - 4, TEMP_SPIKE_HIGH, size=len(sensors))
    return np.round(temps, 1)

def _simulate_setpoints(n, n_sensors):
    per_sensor = TEMP_MEAN + np.linspace(-2.0, 2.0, n_sensors)
    return np.round(np.tile(per_sensor, (n, 1)), 1)

def _simulate_actuator_percentages(temps, setpoints, rng):
    ratio = temps / np.maximum(setpoints, 1.0)
    perc  = np.clip(ratio * 100.0, 0, 127) + rng.normal(0, 2.0, size=ratio.shape)
    return np.round(np.clip(perc, 0, 127), 1)

def _simulate_velocities(n, rng):
    offsets = rng.uniform(-30.0, 30.0, size=22)
    noise   = rng.normal(0.0, SPEED_NOISE, size=(n, 22))
    vels    = SPEED_MEAN + offsets[None, :] + noise
    dips    = np.where(rng.random(size=n) < 0.01)[0]
    for idx in dips:
        vels[idx, :] -= rng.uniform(80.0, 250.0)
    return np.round(np.maximum(vels, 0.0), 1)

# ─── CONSTRUCTION DES SECTIONS ────────────────────────────────────────────────
def build_alarm_df(date, rng):
    n = int(rng.integers(*NUM_ALARMS_RANGE))
    rows = []
    for _ in range(n):
        aid, obj, desc, cod = ALARM_CATALOGUE[int(rng.integers(len(ALARM_CATALOGUE)))]
        rows.append({
            "ID_alarme": aid, "Objet": obj, "Description ": desc,
            "Codice": cod, "durée": round(float(rng.integers(5, 60000)), 1),
            "nbre_occurence": int(rng.integers(1, 31)),
        })
    return pd.DataFrame(rows, columns=ALARM_COLS)

def build_temperature_df(date, rng):
    n        = 24 * 60
    ts_list  = _timestamps(date, n)
    date_str = date.strftime("%Y-%m-%d")
    temps    = _simulate_temperatures(n, rng)
    setpts   = _simulate_setpoints(n, 60)
    percs    = _simulate_actuator_percentages(temps, setpts, rng)
    row_data = {"Data": [date_str]*n, "Minuti": list(range(1, n+1)), "Curva": [CURVA_VALUE]*n}
    for i in range(60):
        row_data[f"Temp{i+1}"]  = temps[:, i]
        row_data[f"SetT{i+1}"]  = setpts[:, i]
        row_data[f"PercT{i+1}"] = percs[:, i]
    row_data["ora"]       = [ts.strftime("%H:%M") for ts in ts_list]
    row_data["timestamp"] = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in ts_list]
    return pd.DataFrame(row_data, columns=TEMP_COLS)

def build_velocita_df(date, rng):
    n        = 24 * 60
    date_str = date.strftime("%Y-%m-%d")
    vels     = _simulate_velocities(n, rng)
    setpt    = round(SPEED_MEAN, 1)
    vel_names  = [c for c in VEL_COLS if c.startswith("Vel")]
    setv_names = [c for c in VEL_COLS if c.startswith("SetV")]
    row_data   = {"Data": [date_str]*n, "Minuti": list(range(1, n+1)), "Curva": [CURVA_VALUE]*n}
    for i, col in enumerate(vel_names):
        row_data[col] = vels[:, i]
    for col in setv_names:
        row_data[col] = setpt
    return pd.DataFrame(row_data, columns=VEL_COLS)

# ─── ÉCRITURE ─────────────────────────────────────────────────────────────────
def write_output(date, alarm_df, temp_df, vel_df):
    filename = date.strftime("%d-%m-%Y") + ".csv"
    filepath = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(filepath):
        answer = input(f"'{filename}' existe déjà. Écraser ? [y/N] ").strip().lower()
        if answer != "y":
            print("Annulé — fichier existant conservé.")
            sys.exit(0)

    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        fh.write("### ALARMS ###\n")
        alarm_df.to_csv(fh, index=False)
        fh.write("\n### TEMPERATURE ###\n")
        temp_df.to_csv(fh, index=False)
        fh.write("\n### VELOCITA ###\n")
        vel_df.to_csv(fh, index=False)

    print(f"\nGénéré : {os.path.abspath(filepath)}")
    print(f"  Alarmes      : {len(alarm_df):>5} lignes")
    print(f"  Température  : {len(temp_df):>5} lignes  ({len(temp_df.columns)} colonnes)")
    print(f"  Vitesse      : {len(vel_df):>5} lignes  ({len(vel_df.columns)} colonnes)")

# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
def _parse_date(arg):
    try:
        return datetime.strptime(arg, "%d-%m-%Y")
    except ValueError:
        print(f"Erreur : format attendu DD-MM-YYYY, reçu '{arg}'")
        sys.exit(1)

def main():
    if len(sys.argv) == 1:
        target = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    elif len(sys.argv) == 2:
        target = _parse_date(sys.argv[1])
    else:
        print("Usage: python daily_csv_generator.py [DD-MM-YYYY]")
        sys.exit(1)

    rng = np.random.default_rng(seed=int(target.timestamp()))
    print(f"Génération des données pour le {target.strftime('%d-%m-%Y')} …")
    alarm_df = build_alarm_df(target, rng)
    temp_df  = build_temperature_df(target, rng)
    vel_df   = build_velocita_df(target, rng)
    write_output(target, alarm_df, temp_df, vel_df)

if __name__ == "__main__":
    main()
