import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

np.random.seed(42)

TEMP_PARAMS = {
    'Temp_Module1':  {'normal': 85,  'std': 5,  'setpoint': 90},
    'Temp_Module2':  {'normal': 105, 'std': 6,  'setpoint': 110},
    'Temp_Module3':  {'normal': 130, 'std': 8,  'setpoint': 135},
    'Temp_Bruleur3': {'normal': 145, 'std': 10, 'setpoint': 150},
    'Pression_EAU':  {'normal': 0.3, 'std': 0.1, 'setpoint': 0.5},
    'Vitesse_Tapis': {'normal': 1.2, 'std': 0.1, 'setpoint': 1.3},
}

ALARM_RATES = {
    'burner_failure':   0.10 / (24 * 60),
    'temp_deviation':   0.07 / (24 * 60),
    'gradient_anomaly': 0.08 / (24 * 60),
}

ALARM_DURATIONS = {
    'burner_failure':   {'mean': 6,  'std': 8},
    'temp_deviation':   {'mean': 25, 'std': 20},
    'gradient_anomaly': {'mean': 20, 'std': 25},
}

START_DATE = datetime(2026, 3, 1)
END_DATE   = datetime(2026, 5, 5)
FREQ_SEC   = 60

OUT_PATH = (Path(__file__).parent.parent
            / "cjo-maintenance-data-cleaned"
            / "sechoir_capteurs_IF.csv")

COLS = [
    'timestamp', 'hour', 'day_of_week', 'month', 'is_night', 'is_morning',
    'hour_sin', 'hour_cos',
    'Temp_Module1', 'Temp_Module2', 'Temp_Module3', 'Temp_Bruleur3',
    'Pression_EAU', 'Vitesse_Tapis',
    'Temp_Module1_mean10',  'Temp_Module1_std10',  'Temp_Module1_diff1',  'Temp_Module1_diff5',
    'Temp_Module2_mean10',  'Temp_Module2_std10',  'Temp_Module2_diff1',  'Temp_Module2_diff5',
    'Temp_Module3_mean10',  'Temp_Module3_std10',  'Temp_Module3_diff1',  'Temp_Module3_diff5',
    'Temp_Bruleur3_mean10', 'Temp_Bruleur3_std10', 'Temp_Bruleur3_diff1', 'Temp_Bruleur3_diff5',
    'Temp_Module1_dev', 'Temp_Module2_dev', 'Temp_Module3_dev', 'Temp_Bruleur3_dev',
    'temp_mean_all', 'temp_max_all', 'temp_std_all', 'temp_range',
    'Pression_EAU_mean10', 'Vitesse_Tapis_mean10',
    'burner_manuel',
    'failure_type', 'alarm_active', 'anomaly_label',
]


def inject_failure(df, failure_type, rate, dur_mean, dur_std, sensor_affected, amplitude):
    n = len(df)
    sensor_cols = [s for s in sensor_affected if s in df.columns]
    col_idx = {s: df.columns.get_loc(s) for s in sensor_cols}
    ft_col   = df.columns.get_loc('failure_type')
    alm_col  = df.columns.get_loc('alarm_active')
    man_col  = df.columns.get_loc('burner_manuel')

    i = 0
    while i < n:
        if np.random.random() < rate * 60:
            dur   = int(np.clip(np.random.normal(dur_mean, dur_std), 1, 300))
            end_i = min(i + dur, n)

            pre_start = max(0, i - 30)
            n_pre = i - pre_start
            if n_pre > 0:
                progress = np.linspace(0, 1, n_pre) * amplitude * 0.5
                for sensor in sensor_cols:
                    df.iloc[pre_start:i, col_idx[sensor]] += progress

            df.iloc[i:end_i, ft_col]  = failure_type
            df.iloc[i:end_i, alm_col] = 1
            for sensor in sensor_cols:
                noise = amplitude * np.random.normal(1, 0.2, end_i - i)
                df.iloc[i:end_i, col_idx[sensor]] += noise

            if failure_type == 'Burner_Failure':
                manuel_start = max(0, i - 15)
                df.iloc[manuel_start:i, man_col] = 1

            i = end_i + np.random.randint(30, 120)
        else:
            i += 1
    return df


def main():
    print("=" * 60)
    print("  Génération capteurs synthétiques séchoir")
    print("  Calibré sur données réelles SCADA CJO")
    print("=" * 60)

    timestamps = []
    t = START_DATE
    while t < END_DATE:
        timestamps.append(t)
        t += timedelta(seconds=FREQ_SEC)
    N = len(timestamps)
    print(f"\nPériode : {START_DATE.date()} → {END_DATE.date()}")
    print(f"Nombre de points : {N} ({N // (60 * 24)} jours)")

    df = pd.DataFrame({'timestamp': pd.to_datetime(timestamps)})
    df['hour']          = df['timestamp'].dt.hour
    df['day_of_week']   = df['timestamp'].dt.dayofweek
    df['month']         = df['timestamp'].dt.month
    df['minute_of_day'] = df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute

    print("\n[1] Génération des capteurs de base...")
    for name, p in TEMP_PARAMS.items():
        night_effect = np.where(
            (df['hour'] < 6) | (df['hour'] >= 22),
            -3, np.where(df['hour'] < 14, 2, 0)
        )
        noise = np.random.normal(0, p['std'], N)
        drift = 2 * np.sin(2 * np.pi * np.arange(N) / (7 * 24 * 60))
        df[name] = (p['normal'] + night_effect + noise + drift).clip(0, p['setpoint'] * 1.5)

    print("[2] Génération des événements de pannes...")
    df['failure_type'] = 'Normal'
    df['alarm_active'] = 0
    df['burner_manuel'] = 0

    df = inject_failure(df, 'Burner_Failure',
                        rate=ALARM_RATES['burner_failure'],
                        dur_mean=ALARM_DURATIONS['burner_failure']['mean'],
                        dur_std=ALARM_DURATIONS['burner_failure']['std'],
                        sensor_affected=['Temp_Bruleur3', 'Temp_Module3'],
                        amplitude=20)

    df = inject_failure(df, 'Temp_Deviation',
                        rate=ALARM_RATES['temp_deviation'],
                        dur_mean=ALARM_DURATIONS['temp_deviation']['mean'],
                        dur_std=ALARM_DURATIONS['temp_deviation']['std'],
                        sensor_affected=['Temp_Module1', 'Temp_Module2', 'Temp_Module3'],
                        amplitude=15)

    df = inject_failure(df, 'Gradient_Anomaly',
                        rate=ALARM_RATES['gradient_anomaly'],
                        dur_mean=ALARM_DURATIONS['gradient_anomaly']['mean'],
                        dur_std=ALARM_DURATIONS['gradient_anomaly']['std'],
                        sensor_affected=['Temp_Module3', 'Temp_Bruleur3'],
                        amplitude=25)

    print("[3] Feature engineering...")
    temp_sensors = ['Temp_Module1', 'Temp_Module2', 'Temp_Module3', 'Temp_Bruleur3']

    for sensor in temp_sensors:
        df[f'{sensor}_mean10'] = df[sensor].rolling(10, min_periods=1).mean()
        df[f'{sensor}_std10']  = df[sensor].rolling(10, min_periods=1).std().fillna(0)
        df[f'{sensor}_diff1']  = df[sensor].diff(1).fillna(0)
        df[f'{sensor}_diff5']  = df[sensor].diff(5).fillna(0)

    for name, p in TEMP_PARAMS.items():
        if name in df.columns:
            df[f'{name}_dev'] = df[name] - p['setpoint']

    df['temp_mean_all'] = df[temp_sensors].mean(axis=1)
    df['temp_max_all']  = df[temp_sensors].max(axis=1)
    df['temp_std_all']  = df[temp_sensors].std(axis=1).fillna(0)
    df['temp_range']    = df[temp_sensors].max(axis=1) - df[temp_sensors].min(axis=1)

    df['is_night']   = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)
    df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 14)).astype(int)
    df['hour_sin']   = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']   = np.cos(2 * np.pi * df['hour'] / 24)

    df['Pression_EAU_mean10']  = df['Pression_EAU'].rolling(10, min_periods=1).mean()
    df['Vitesse_Tapis_mean10'] = df['Vitesse_Tapis'].rolling(10, min_periods=1).mean()

    df['anomaly_label'] = (
        df['alarm_active'].rolling(5, min_periods=1).max()
        .shift(-5).fillna(0).astype(int)
    )

    print("\n[4] Vérification...")
    print(f"  Total lignes : {len(df)}")
    print(f"  Nulls        : {df.isna().sum().sum()}")
    print("\n  failure_type :")
    for ft, cnt in df['failure_type'].value_counts().items():
        print(f"    {ft:25s} : {cnt:>6} ({cnt / len(df) * 100:.1f}%)")
    print("\n  anomaly_label :")
    print(f"    Normal   : {(df['anomaly_label']==0).sum()} ({(df['anomaly_label']==0).mean()*100:.1f}%)")
    print(f"    Anomalie : {(df['anomaly_label']==1).sum()} ({(df['anomaly_label']==1).mean()*100:.1f}%)")

    df_out = df[COLS].fillna(0)
    df_out.to_csv(OUT_PATH, index=False)
    print(f"\n✅ Fichier sauvegardé : {OUT_PATH.name}")
    print(f"   {len(df_out)} lignes | {len(df_out.columns)} colonnes")
    print("=" * 60)


if __name__ == "__main__":
    main()
