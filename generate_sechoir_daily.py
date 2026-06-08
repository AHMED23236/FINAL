import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
TODAY    = datetime.now().date()
DATA_DIR = Path(__file__).parent.parent / "cjo-maintenance-data-cleaned"
OUT_FILE = DATA_DIR / "Alarme_sechoir_ML.csv"
np.random.seed(None)
print(f"{'='*60}")
print(f"  Génération données séchoir — {TODAY.strftime('%d/%m/%Y')}")
print(f"{'='*60}")
N_WINDOWS = 288
P_NO_FAILURE  = 0.50
P_THERMAL     = 0.15
P_MECHANICAL  = 0.35
DUR_MECHANICAL = {'mean': 2.5,   'std': 5.0,   'min': 0.0, 'max': 170.0}
DUR_THERMAL = {'mean': 40.0, 'std': 30.0, 'min': 2.0, 'max': 180.0}
DUR_NO_FAILURE = {'mean': 0.0,   'std': 0.0,   'min': 0.0, 'max': 0.0}
PRESSOSTAT_PARAMS = {
    'EAU11': {'mean_nz': 0.226, 'std_nz': 0.584, 'max': 2.50, 'spike_prob': 0.027},
    'EAU12': {'mean_nz': 0.415, 'std_nz': 0.942, 'max': 2.51, 'spike_prob': 0.027},
    'EAU21': {'mean_nz': 0.937, 'std_nz': 1.165, 'max': 3.33, 'spike_prob': 0.027},
    'EAU22': {'mean_nz': 1.133, 'std_nz': 1.578, 'max': 4.44, 'spike_prob': 0.027},
}
print(f"\n[1] Génération des {N_WINDOWS} fenêtres de 5 min...")
start_dt = datetime.combine(TODAY, datetime.min.time()) + timedelta(minutes=5)
windows_start = [start_dt + timedelta(minutes=5*i) for i in range(N_WINDOWS)]
windows_end   = [ws + timedelta(minutes=5) for ws in windows_start]
df = pd.DataFrame({
    'window_start': windows_start,
    'window_end':   windows_end,
})
print("[2] Génération des labels (failure_type + severity)...")
ft_choices = np.random.choice(
    ['No_Failure', 'Thermal_Anomaly', 'Mechanical_Stop'],
    size=N_WINDOWS,
    p=[P_NO_FAILURE, P_THERMAL, P_MECHANICAL]
)
ft_smooth = ft_choices.copy()
i = 0
while i < N_WINDOWS:
    if ft_smooth[i] == 'Thermal_Anomaly':
        dur = max(2, int(np.random.normal(8, 4)))
        dur = min(dur, 20)
        for j in range(i, min(i+dur, N_WINDOWS)):
            ft_smooth[j] = 'Thermal_Anomaly'
        i += dur
    elif ft_smooth[i] == 'Mechanical_Stop':
        dur = max(1, int(np.random.exponential(1.5)))
        dur = min(dur, 5)
        for j in range(i, min(i+dur, N_WINDOWS)):
            ft_smooth[j] = 'Mechanical_Stop'
        i += dur
    else:
        i += 1
df['failure_type'] = ft_smooth
def gen_severity(ft, dur):
    if ft == 'No_Failure':
        return 'NONE'
    if ft == 'Thermal_Anomaly':
        if dur > 60:   return 'HIGH'
        if dur > 10:   return 'MEDIUM'
        return 'LOW'
    if ft == 'Mechanical_Stop':
        if dur > 60:   return 'HIGH'
        if dur > 10:   return 'MEDIUM'
        return 'LOW'
    return 'NONE'
def gen_duration(ft):
    if ft == 'No_Failure':
        return 0.0
    if ft == 'Thermal_Anomaly':
        d = np.random.normal(DUR_THERMAL['mean'], DUR_THERMAL['std'])
        return round(float(np.clip(d, DUR_THERMAL['min'], DUR_THERMAL['max'])), 1)
    if ft == 'Mechanical_Stop':
        d = np.random.exponential(DUR_MECHANICAL['mean'])
        return round(float(np.clip(d, DUR_MECHANICAL['min'], DUR_MECHANICAL['max'])), 1)
    return 0.0
durations = [gen_duration(ft) for ft in df['failure_type']]
df['alarm_duration_min'] = durations
df['severity'] = [gen_severity(ft, dur) for ft, dur in zip(df['failure_type'], durations)]
print("[3] Génération des features temporelles...")
df['hour']        = df['window_start'].dt.hour
df['day_of_week'] = df['window_start'].dt.dayofweek
df['month']       = df['window_start'].dt.month
df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
def get_shift(h):
    if 6 <= h < 14:  return 'Morning'
    if 14 <= h < 22: return 'Afternoon'
    return 'Night'
df['shift']            = df['hour'].apply(get_shift)
df['shift_enc']        = df['shift'].map({'Morning':0,'Afternoon':1,'Night':2})
df['severity_enc']     = df['severity'].map({'NONE':0,'LOW':1,'MEDIUM':2,'HIGH':3})
df['is_night_shift']   = ((df['hour']>=22)|(df['hour']<6)).astype(int)
df['is_morning_shift'] = ((df['hour']>=6)&(df['hour']<14)).astype(int)
df['hour_sin']         = np.sin(2*np.pi*df['hour']/24).round(4)
df['hour_cos']         = np.cos(2*np.pi*df['hour']/24).round(4)
df['dow_sin']          = np.sin(2*np.pi*df['day_of_week']/7).round(4)
df['dow_cos']          = np.cos(2*np.pi*df['day_of_week']/7).round(4)
print("[4] Génération des features rolling alarmes...")
is_alarm = (df['failure_type'] != 'No_Failure').astype(int)
is_thermal  = (df['failure_type'] == 'Thermal_Anomaly').astype(int)
is_mech     = (df['failure_type'] == 'Mechanical_Stop').astype(int)
is_high     = (df['severity'] == 'HIGH').astype(int)
is_medium   = (df['severity'] == 'MEDIUM').astype(int)
df['n_alarms_in_window']   = is_alarm
df['failure_in_next_30min'] = is_alarm
df['alarms_last_60min']    = is_alarm.rolling(12, min_periods=1).sum().astype(int)
df['alarms_last_1h']       = (is_alarm.rolling(12, min_periods=1).sum() - is_alarm).clip(lower=0).astype(int)
df['alarms_last_6h']       = (is_alarm.rolling(72, min_periods=1).sum() - is_alarm).clip(lower=0).astype(int)
df['alarms_last_24h']      = (is_alarm.rolling(288,min_periods=1).sum() - is_alarm).clip(lower=0).astype(int)
df['past_thermal_6h']      = (is_thermal.rolling(72, min_periods=1).sum() - is_thermal).clip(lower=0).astype(int)
df['past_mechanical_6h']   = (is_mech.rolling(72, min_periods=1).sum() - is_mech).clip(lower=0).astype(int)
df['past_high_6h']         = is_high.rolling(72, min_periods=1).sum().astype(int)
df['past_medium_6h']       = is_medium.rolling(72, min_periods=1).sum().astype(int)
last_thermal_time  = None
last_mech_time     = None
times_any  = []
times_th   = []
times_mech = []
for i, row in df.iterrows():
    ws = row['window_start']
    if i == 0:
        times_any.append(0.0)
    else:
        times_any.append(5.0)
    if row['failure_type'] == 'Thermal_Anomaly':
        last_thermal_time = ws
    if last_thermal_time is None:
        times_th.append(9999.0)
    else:
        delta = (ws - last_thermal_time).total_seconds() / 60
        times_th.append(min(delta, 9999.0))
    if row['failure_type'] == 'Mechanical_Stop':
        last_mech_time = ws
    if last_mech_time is None:
        times_mech.append(9999.0)
    else:
        delta = (ws - last_mech_time).total_seconds() / 60
        times_mech.append(min(delta, 9999.0))
df['time_since_last_any']        = times_any
df['time_since_last_thermal']    = times_th
df['time_since_last_mechanical'] = times_mech
df['mean_duration_last_20'] = df['alarm_duration_min'].shift(1).rolling(20, min_periods=1).mean().fillna(0).round(3)
df['max_duration_last_20']  = df['alarm_duration_min'].shift(1).rolling(20, min_periods=1).max().fillna(0).round(3)
df['std_duration_last_20']  = df['alarm_duration_min'].shift(1).rolling(20, min_periods=1).std().fillna(0).round(3)
df['alarm_acceleration']    = (df['alarms_last_1h'] / (df['alarms_last_6h'] + 1)).round(4)
print("[5] Génération des pressostats (distributions réelles)...")
for name, p in PRESSOSTAT_PARAMS.items():
    signal = np.zeros(N_WINDOWS)
    spike_mask = np.random.random(N_WINDOWS) < p['spike_prob']
    n_sp = spike_mask.sum()
    if n_sp > 0:
        vals = np.abs(np.random.normal(p['mean_nz'], p['std_nz'], n_sp))
        vals = np.clip(vals, 0, p['max'])
        signal[spike_mask] = vals
    thermal_idx = df[df['failure_type'] == 'Thermal_Anomaly'].index.tolist()
    for idx in thermal_idx:
        for pre in range(max(0, idx-6), idx):
            if np.random.random() < 0.4:
                signal[pre] = min(
                    signal[pre] + abs(np.random.normal(p['mean_nz']*2, p['std_nz'])),
                    p['max']
                )
    df[name] = signal.round(3)
for name in ['EAU11', 'EAU12', 'EAU21', 'EAU22']:
    for w, lbl in [(6,'30min'), (12,'1h'), (72,'6h')]:
        df[f'{name}_mean_{lbl}'] = df[name].rolling(w, min_periods=1).mean().round(4)
        df[f'{name}_std_{lbl}']  = df[name].rolling(w, min_periods=1).std().fillna(0).round(4)
        df[f'{name}_max_{lbl}']  = df[name].rolling(w, min_periods=1).max().round(4)
    df[f'{name}_diff1']  = df[name].diff(1).fillna(0).round(4)
    df[f'{name}_diff3']  = df[name].diff(3).fillna(0).round(4)
    mu = df[name].mean(); sigma = df[name].std() + 1e-8
    df[f'{name}_zscore'] = ((df[name] - mu) / sigma).round(4)
df['EAU_delta_11_22'] = (df['EAU22'] - df['EAU11']).round(4)
df['EAU_delta_12_21'] = (df['EAU21'] - df['EAU12']).round(4)
df['EAU_mean_all']    = df[['EAU11','EAU12','EAU21','EAU22']].mean(axis=1).round(4)
df['EAU_max_all']     = df[['EAU11','EAU12','EAU21','EAU22']].max(axis=1).round(4)
COLS = [
    'window_start','window_end','severity','failure_type',
    'failure_in_next_30min','alarm_duration_min','n_alarms_in_window',
    'hour','day_of_week','month','is_weekend','shift',
    'alarms_last_60min','shift_enc','severity_enc',
    'alarms_last_1h','alarms_last_6h','alarms_last_24h',
    'past_thermal_6h','past_mechanical_6h','past_high_6h','past_medium_6h',
    'time_since_last_any','time_since_last_thermal','time_since_last_mechanical',
    'mean_duration_last_20','max_duration_last_20','std_duration_last_20',
    'alarm_acceleration',
    'EAU11','EAU12','EAU21','EAU22',
    'EAU11_mean_30min','EAU11_std_30min','EAU11_max_30min',
    'EAU11_mean_1h','EAU11_std_1h','EAU11_max_1h',
    'EAU11_mean_6h','EAU11_std_6h','EAU11_max_6h',
    'EAU11_diff1','EAU11_diff3','EAU11_zscore',
    'EAU12_mean_30min','EAU12_std_30min','EAU12_max_30min',
    'EAU12_mean_1h','EAU12_std_1h','EAU12_max_1h',
    'EAU12_mean_6h','EAU12_std_6h','EAU12_max_6h',
    'EAU12_diff1','EAU12_diff3','EAU12_zscore',
    'EAU21_mean_30min','EAU21_std_30min','EAU21_max_30min',
    'EAU21_mean_1h','EAU21_std_1h','EAU21_max_1h',
    'EAU21_mean_6h','EAU21_std_6h','EAU21_max_6h',
    'EAU21_diff1','EAU21_diff3','EAU21_zscore',
    'EAU22_mean_30min','EAU22_std_30min','EAU22_max_30min',
    'EAU22_mean_1h','EAU22_std_1h','EAU22_max_1h',
    'EAU22_mean_6h','EAU22_std_6h','EAU22_max_6h',
    'EAU22_diff1','EAU22_diff3','EAU22_zscore',
    'EAU_delta_11_22','EAU_delta_12_21','EAU_mean_all','EAU_max_all',
    'is_night_shift','is_morning_shift',
    'hour_sin','hour_cos','dow_sin','dow_cos',
]
df = df[COLS].fillna(0)
print("\n[6] Vérification finale...")
assert df.isna().sum().sum() == 0, "NULLS DÉTECTÉS !"
assert len(df) == N_WINDOWS
print(f"  Lignes     : {len(df)}")
print(f"  Colonnes   : {len(df.columns)}")
print(f"  Nulls      : {df.isna().sum().sum()}")
print(f"\n  failure_type:")
for ft, n in df['failure_type'].value_counts().items():
    print(f"    {ft:22s} : {n:>4} ({n/len(df)*100:.1f}%)")
print(f"\n  severity:")
for s, n in df['severity'].value_counts().items():
    print(f"    {s:8s} : {n:>4} ({n/len(df)*100:.1f}%)")
df.to_csv(OUT_FILE, index=False)
print(f"\n{'='*60}")
print(f"  ✅ Fichier généré : {OUT_FILE.name}")
print(f"  📅 Date           : {TODAY.strftime('%d/%m/%Y')}")
print(f"  📊 {len(df)} fenêtres de 5 min (24h complètes)")
print(f"{'='*60}")

