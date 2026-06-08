import datetime
import io
from pathlib import Path
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "cjo-maintenance-data-cleaned"
LIVE_CSV_DIR = DATA_DIR / "daily"
def _inject_css():
    st.markdown("""
<style>
.kpi-card {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 12px; padding: 1rem 1.2rem;
    text-align: center; margin-bottom: 0.5rem;
}
.kpi-label { color: #94a3b8; font-size: 0.75rem;
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.3rem; }
.kpi-value { color: #f1f5f9; font-size: 1.8rem; font-weight: 700; }
.kpi-sub   { color: #64748b; font-size: 0.72rem; margin-top: 0.2rem; }
.badge { display: inline-block; padding: 2px 10px;
    border-radius: 999px; font-size: 0.72rem; font-weight: 600; margin-right: 4px; }
.badge-high   { background: #ef444422; color: #ef4444; }
.badge-medium { background: #f9731622; color: #f97316; }
.badge-low    { background: #22c55e22; color: #22c55e; }
</style>
""", unsafe_allow_html=True)
def kpi(label, value, sub="", color=None):
    value_style = f' style="color:{color};"' if color else ""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value"{value_style}>{value}</div>
        {"<div class='kpi-sub'>"+sub+"</div>" if sub else ""}
    </div>""", unsafe_allow_html=True)
TEMPERATURE_FILE = "TREND_Temperature.csv"
VELOCITA_FILE    = "TREND_Velocita.csv"
ALARMS_FILE      = "Allarmi_TRAIN_READY.csv"
LIVE_REFRESH_SECONDS = 60
HORIZON_MAP = {
    "15 min": 15, "1 heure": 60, "4 heures": 240,
    "12 heures": 720, "Journée complète": 1440,
}
PAGES = ["📊 Vue Générale", "🚨 Alarmes", "🔴 Anomalies", "🔮 Prédiction Temps Réel", "🤖 AI Predictions"]
MODEL_FEATURE_NAMES = [
    "hour", "dayofweek", "is_weekend", "is_night", "is_peak_hours",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "alarms_last_1h", "alarms_last_6h", "alarms_last_24h",
    "high_last_6h", "medium_last_6h", "low_last_6h", "high_last_24h",
    "temp_fail_last_6h", "burner_fail_last_6h",
    "pression_fail_last_6h", "alim_fail_last_6h",
    "mean_duration_6h", "max_duration_6h", "unique_zones_24h",
    "minutes_since_last_alarm", "minutes_since_last_high",
]
@st.cache_data(show_spinner=False)
def load_temperature():
    df = pd.read_csv(DATA_DIR / TEMPERATURE_FILE, encoding="utf-8-sig", low_memory=False)
    df["Data"] = pd.to_datetime(df["Data"], dayfirst=True, errors="coerce")
    df["timestamp"] = df["Data"] + pd.to_timedelta(df["Minuti"] - 1, unit="m")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)
@st.cache_data(show_spinner=False)
def load_vitesse():
    df = pd.read_csv(DATA_DIR / VELOCITA_FILE, encoding="utf-8-sig", low_memory=False)
    rename_map = {}
    for c in df.columns:
        low = c.lower()
        if low.startswith("vel") and low[3:].isdigit():
            rename_map[c] = "Vel" + low[3:]
        elif low.startswith("setv") and low[4:].isdigit():
            rename_map[c] = "SetV" + low[4:]
    if rename_map:
        df = df.rename(columns=rename_map)
    df["Data"] = pd.to_datetime(df["Data"], dayfirst=True, errors="coerce")
    df["timestamp"] = df["Data"] + pd.to_timedelta(df["Minuti"] - 1, unit="m")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)
@st.cache_data(show_spinner=False, ttl=1800)
def load_alarms_four():
    df = pd.read_csv(DATA_DIR / ALARMS_FILE, low_memory=False)
    df["Inizio"] = pd.to_datetime(df["Inizio"], errors="coerce")
    df["Fine"]   = pd.to_datetime(df["Fine"],   errors="coerce")
    df["urgency"]  = df["urgency"].astype(str).str.title()
    df["severity"] = df["urgency"].str.upper()
    df = df.dropna(subset=["Inizio"])
    df = df.sort_values("Inizio").reset_index(drop=True)
    shift = pd.Timestamp.now() - df["Inizio"].max() - pd.Timedelta(hours=1)
    df["Inizio"] = df["Inizio"] + shift
    df["Fine"]   = df["Fine"]   + shift
    df["timestamp"] = df["Inizio"]
    df["date"]      = df["Inizio"].dt.date
    return df
def find_live_csv(date: datetime.date = None) -> Path | None:
    if date is None:
        date = datetime.date.today()
    path = LIVE_CSV_DIR / date.strftime("%d-%m-%Y.csv")
    return path if path.exists() else None
def parse_daily_csv(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    section_markers = {"### ALARMS ###": "alarms",
                       "### TEMPERATURE ###": "temperature",
                       "### VELOCITA ###": "velocita"}
    sections: dict[str, list[str]] = {}
    current_key: str | None = None
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            marker = line.strip()
            if marker in section_markers:
                current_key = section_markers[marker]
                sections[current_key] = []
            elif current_key is not None and line.strip():
                sections[current_key].append(line)
    def _to_df(key: str) -> pd.DataFrame:
        rows = sections.get(key, [])
        if not rows:
            return pd.DataFrame()
        return pd.read_csv(io.StringIO("\n".join(rows)), low_memory=False)
    return _to_df("alarms"), _to_df("temperature"), _to_df("velocita")
def _attach_timestamp_temp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
        df["timestamp"] = df["Data"] + pd.to_timedelta(df["Minuti"] - 1, unit="m")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
def _attach_timestamp_vel(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
    df["timestamp"] = df["Data"] + pd.to_timedelta(df["Minuti"] - 1, unit="m")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
def load_live_data(date: datetime.date = None):
    path = find_live_csv(date)
    if path is None:
        return None, None, None, None
    df_al, df_temp, df_vel = parse_daily_csv(path)
    if not df_temp.empty:
        df_temp = _attach_timestamp_temp(df_temp)
    if not df_vel.empty:
        df_vel = _attach_timestamp_vel(df_vel)
    return df_al, df_temp, df_vel, path
def build_feature_matrix(timestamps, df_alarm: pd.DataFrame) -> pd.DataFrame:
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
    out = pd.DataFrame(index=ts)
    out["hour"]          = ts.hour
    out["dayofweek"]     = ts.dayofweek
    out["is_weekend"]    = (ts.dayofweek >= 5).astype(int)
    out["is_night"]      = ((ts.hour >= 22) | (ts.hour <= 5)).astype(int)
    out["is_peak_hours"] = ((ts.hour >= 8) & (ts.hour <= 18)).astype(int)
    out["hour_sin"]      = np.sin(2 * np.pi * ts.hour / 24)
    out["hour_cos"]      = np.cos(2 * np.pi * ts.hour / 24)
    out["dow_sin"]       = np.sin(2 * np.pi * ts.dayofweek / 7)
    out["dow_cos"]       = np.cos(2 * np.pi * ts.dayofweek / 7)
    al = df_alarm.sort_values("Inizio").reset_index(drop=True)
    al_starts   = al["Inizio"].values
    al_urgency  = al["urgency"].astype(str).values
    al_failtype = al["failure_type"].astype(str).values
    al_duration = al["duration_minutes"].astype(float).values
    zone_col    = "Descrizione" if "Descrizione" in al.columns else "Oggetto"
    al_zone     = al[zone_col].astype(str).values
    rows = []
    for t in ts:
        t64 = np.datetime64(t)
        h1  = t64 - np.timedelta64(1,  "h")
        h6  = t64 - np.timedelta64(6,  "h")
        h24 = t64 - np.timedelta64(24, "h")
        m1  = (al_starts > h1)  & (al_starts <= t64)
        m6  = (al_starts > h6)  & (al_starts <= t64)
        m24 = (al_starts > h24) & (al_starts <= t64)
        u6  = al_urgency[m6]
        u24 = al_urgency[m24]
        f6  = al_failtype[m6]
        d6  = al_duration[m6]
        z24 = al_zone[m24]
        past      = al_starts[al_starts <= t64]
        past_high = al_starts[(al_starts <= t64) & (al_urgency == "High")]
        mins_last = (t64 - past[-1])      / np.timedelta64(1, "m") if len(past)      else 9999.0
        mins_high = (t64 - past_high[-1]) / np.timedelta64(1, "m") if len(past_high) else 9999.0
        rows.append({
            "alarms_last_1h":           int(m1.sum()),
            "alarms_last_6h":           int(m6.sum()),
            "alarms_last_24h":          int(m24.sum()),
            "high_last_6h":             int((u6  == "High").sum()),
            "medium_last_6h":           int((u6  == "Medium").sum()),
            "low_last_6h":              int((u6  == "Low").sum()),
            "high_last_24h":            int((u24 == "High").sum()),
            "temp_fail_last_6h":        int((f6 == "temperature_failure").sum()),
            "burner_fail_last_6h":      int((f6 == "burner_failure").sum()),
            "pression_fail_last_6h":    int((f6 == "pression_failure").sum()),
            "alim_fail_last_6h":        int((f6 == "alimentation_failure").sum()),
            "mean_duration_6h":         float(d6.mean()) if len(d6) else 0.0,
            "max_duration_6h":          float(d6.max())  if len(d6) else 0.0,
            "unique_zones_24h":         int(pd.Series(z24).nunique()),
            "minutes_since_last_alarm": float(mins_last),
            "minutes_since_last_high":  float(mins_high),
        })
    rolling = pd.DataFrame(rows, index=ts)
    out = pd.concat([out, rolling], axis=1)
    return out[MODEL_FEATURE_NAMES]
def get_sensor_columns(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix) and c[len(prefix):].isdigit()]
    return sorted(cols, key=lambda c: int(c[len(prefix):]))
def filter_by_date(df, date):
    return df[df["timestamp"].dt.date == date].copy()
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🔥 Four Céramique")
        st.markdown("---")
        page = st.radio("Navigation", options=PAGES,
            key="nav_page_four", label_visibility="collapsed")
        st.markdown("---")
        if st.button("🏠 Accueil", use_container_width=True):
            st.session_state.machine = None
            st.rerun()
        if st.button("🚪 Déconnexion", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.machine = None
            st.rerun()
    return page
def show_main_page(df_temp, df_vel, all_dates, df_alarm):
    st.markdown("## 📊 Vue Générale — Four Céramique")
    if all_dates:
        period_str = f"{all_dates[0].strftime('%d/%m/%Y')} → {all_dates[-1].strftime('%d/%m/%Y')}"
    else:
        period_str = "—"
    st.markdown(
        f"<div style='color:#64748b;font-size:0.85rem;margin-bottom:1.5rem;'>"
        f"Période : {period_str} &nbsp;|&nbsp; Capteurs SCADA &nbsp;|&nbsp; Données réelles</div>",
        unsafe_allow_html=True,
    )
    total_al  = len(df_alarm)
    dur_h     = df_alarm["duration_minutes"].sum() / 60.0 if total_al else 0.0
    n_burn    = int((df_alarm["failure_type"] == "burner_failure").sum())
    n_temp    = int((df_alarm["failure_type"] == "temperature_failure").sum())
    n_pres    = int((df_alarm["failure_type"] == "pression_failure").sum())
    n_alim    = int((df_alarm["failure_type"] == "alimentation_failure").sum())
    n_high    = int((df_alarm["urgency"] == "High").sum())
    n_medium  = int((df_alarm["urgency"] == "Medium").sum())
    n_low     = int((df_alarm["urgency"] == "Low").sum())
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi("Total Alarmes",        f"{total_al:,}".replace(",", " "), "événements")
    with c2: kpi("Durée Totale",         f"{dur_h:.0f} h",                  f"moy {df_alarm['duration_minutes'].mean():.1f} min" if total_al else "")
    with c3: kpi("Anomalies Brûleurs",   f"{n_burn:,}".replace(",", " "),   "burner_failure")
    with c4: kpi("Anomalies Température",f"{n_temp:,}".replace(",", " "),   "temperature_failure")
    with c5: kpi("Anomalies Pression",   f"{n_pres:,}".replace(",", " "),   "pression_failure")
    st.markdown(
        f"<div style='margin:0.5rem 0 1.5rem;'>"
        f"<span class='badge badge-high'>HIGH : {n_high}</span>"
        f"<span class='badge badge-medium'>MEDIUM : {n_medium}</span>"
        f"<span class='badge badge-low'>LOW : {n_low}</span></div>",
        unsafe_allow_html=True,
    )
    col_l, col_r = st.columns(2)
    with col_l:
        ft_c = df_alarm["failure_type"].value_counts().reset_index()
        ft_c.columns = ["Type", "Count"]
        ft_label_map = {
            "burner_failure":       "🔥 Brûleurs",
            "temperature_failure":  "🌡️ Température",
            "pression_failure":     "💨 Pression",
            "alimentation_failure": "⚡ Alimentation",
        }
        ft_c["Type"] = ft_c["Type"].map(ft_label_map).fillna(ft_c["Type"])
        import plotly.express as px
        fig = px.pie(ft_c, values="Count", names="Type",
                     title="Répartition par type de défaut", hole=0.45,
                     color_discrete_sequence=["#EF4444","#F97316","#3B82F6","#8B5CF6"])
        fig.update_layout(plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font_color="#94a3b8")
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        urg_c = df_alarm[df_alarm["urgency"].isin(["High","Medium","Low"])]["urgency"] \
                    .value_counts().reindex(["High","Medium","Low"]).reset_index()
        urg_c.columns = ["Urgence", "Count"]
        urg_c = urg_c.dropna()
        import plotly.express as px
        fig2 = px.bar(urg_c, x="Urgence", y="Count", color="Urgence",
                      color_discrete_map={"High":"#EF4444","Medium":"#F97316","Low":"#22C55E"},
                      title="Distribution par urgence", text="Count")
        fig2.update_traces(textposition="outside")
        fig2.update_layout(plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                           font_color="#94a3b8", showlegend=False,
                           xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b"))
        st.plotly_chart(fig2, use_container_width=True)
    st.markdown("### 📡 Analyse des capteurs SCADA")
    TEMP_SENSORS = get_sensor_columns(df_temp, "Temp")
    SETP_SENSORS = get_sensor_columns(df_temp, "SetT")
    PERC_SENSORS = get_sensor_columns(df_temp, "PercT")
    VEL_SENSORS  = get_sensor_columns(df_vel,  "Vel")
    SETV_SENSORS = get_sensor_columns(df_vel,  "SetV")
    ALL_SENSORS = {k: v for k, v in {
        "🌡️ Température (Temp)":    TEMP_SENSORS,
        "🎯 Consignes Temp (SetT)":  SETP_SENSORS,
        "🔧 Ouverture Vannes (%)":   PERC_SENSORS,
        "⚡ Vitesse (Vel)":          VEL_SENSORS,
        "🎯 Consignes Vitesse":      SETV_SENSORS,
    }.items() if v}
    col_sel, col_date = st.columns([3, 1])
    with col_sel:
        sensor_type = st.selectbox("Type de capteur", options=list(ALL_SENSORS.keys()))
    with col_date:
        selected_date = st.date_input(
            "Date", value=all_dates[-1],
            min_value=all_dates[0], max_value=all_dates[-1],
            key="selected_date_four", label_visibility="collapsed",
        )
    family_cols = ALL_SENSORS[sensor_type]
    source_df = df_vel if family_cols and (family_cols[0].startswith("Vel") or family_cols[0].startswith("SetV")) else df_temp
    available = [s for s in family_cols if s in source_df.columns]
    selected  = st.multiselect("Capteurs", options=available, default=available[:5])
    if not selected:
        st.info("Sélectionnez au moins un capteur.")
        return
    day_df = filter_by_date(source_df, selected_date)
    if day_df.empty:
        st.warning("Aucune donnée pour cette date.")
        return
    chart_data = day_df[["timestamp"] + selected].melt(
        id_vars=["timestamp"], var_name="Capteur", value_name="Valeur"
    )
    chart = alt.Chart(chart_data).mark_line().encode(
        x="timestamp:T",
        y=alt.Y("Valeur:Q").scale(zero=False),
        color="Capteur:N",
    ).properties(height=380)
    st.altair_chart(chart, use_container_width=True)
FAILURE_LABELS = {
    "burner_failure":       "🔥 Brûleurs",
    "temperature_failure":  "🌡️ Température",
    "pression_failure":     "💨 Pression",
    "alimentation_failure": "⚡ Alimentation",
}
URGENCY_COLORS = {
    "High":   "#e74c3c",
    "Medium": "#f39c12",
    "Low":    "#27ae60",
}
def show_alarms_page(df_alarm, all_dates):
    st.markdown("# 🚨 Alarmes — Four")
    if df_alarm.empty:
        st.info("Aucune alarme enregistrée.")
        return
    alarm_dates = sorted(df_alarm["timestamp"].dt.date.unique())
    fmin, fmax = alarm_dates[0], alarm_dates[-1]
    f1, f2, f3 = st.columns([2, 1.2, 1.2])
    with f1:
        date_range = st.date_input(
            "📅 Période",
            value=(fmin, fmax),
            min_value=fmin,
            max_value=fmax,
        )
    with f2:
        failure_opts = sorted(df_alarm["failure_type"].dropna().unique().tolist())
        selected_failures = st.multiselect(
            "Type de défaut",
            options=failure_opts,
            default=failure_opts,
            format_func=lambda x: FAILURE_LABELS.get(x, x),
        )
    with f3:
        urgency_opts = ["High", "Medium", "Low"]
        selected_urgency = st.multiselect(
            "Urgence",
            options=urgency_opts,
            default=urgency_opts,
        )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range
    mask = (
        (df_alarm["timestamp"].dt.date >= start_date) &
        (df_alarm["timestamp"].dt.date <= end_date) &
        (df_alarm["failure_type"].isin(selected_failures)) &
        (df_alarm["urgency"].isin(selected_urgency))
    )
    filtered = df_alarm[mask].copy()
    k1, k2, k3, k4 = st.columns(4)
    total_events     = len(filtered)
    total_downtime_h = filtered["duration_minutes"].sum() / 60.0
    avg_duration     = filtered["duration_minutes"].mean() if total_events else 0.0
    high_urgency     = (filtered["urgency"] == "High").sum()
    with k1: kpi("Total événements",       f"{total_events:,}".replace(",", " "))
    with k2: kpi("Indisponibilité totale", f"{total_downtime_h:.1f} h")
    with k3: kpi("Durée moyenne",          f"{avg_duration:.1f} min")
    with k4: kpi("⚠️ Urgence haute",       f"{high_urgency:,}".replace(",", " "))
    if filtered.empty:
        st.info("Aucune alarme ne correspond aux filtres.")
        return
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("##### Événements par jour (par type de défaut)")
        daily = (
            filtered
            .groupby([filtered["timestamp"].dt.date, "failure_type"])
            .size()
            .reset_index(name="count")
            .rename(columns={"timestamp": "date"})
        )
        daily["failure_label"] = daily["failure_type"].map(FAILURE_LABELS).fillna(daily["failure_type"])
        bar = alt.Chart(daily).mark_bar().encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("count:Q", title="Événements"),
            color=alt.Color("failure_label:N", title="Type", scale=alt.Scale(scheme="set2")),
            tooltip=["date:T", "failure_label:N", "count:Q"],
        ).properties(height=300)
        st.altair_chart(bar, use_container_width=True)
    with c2:
        st.markdown("##### Répartition par urgence")
        urg = filtered["urgency"].value_counts().reset_index()
        urg.columns = ["urgency", "count"]
        donut = alt.Chart(urg).mark_arc(innerRadius=55).encode(
            theta="count:Q",
            color=alt.Color(
                "urgency:N",
                scale=alt.Scale(
                    domain=list(URGENCY_COLORS.keys()),
                    range=list(URGENCY_COLORS.values()),
                ),
                legend=alt.Legend(title="Urgence"),
            ),
            tooltip=["urgency:N", "count:Q"],
        ).properties(height=300)
        st.altair_chart(donut, use_container_width=True)
    st.markdown("##### Détail des événements")
    display_cols = {
        "Inizio":           "Début",
        "Fine":             "Fin",
        "duration_minutes": "Durée (min)",
        "Oggetto":          "Équipement",
        "Descrizione":      "Description",
        "failure_type":     "Type de défaut",
        "urgency":          "Urgence",
    }
    available = [c for c in display_cols if c in filtered.columns]
    table = filtered[available].rename(columns=display_cols).copy()
    if "Durée (min)" in table.columns:
        table["Durée (min)"] = table["Durée (min)"].round(2)
    if "Type de défaut" in table.columns:
        table["Type de défaut"] = table["Type de défaut"].map(FAILURE_LABELS).fillna(table["Type de défaut"])
    st.dataframe(table, use_container_width=True, hide_index=True, height=420)
def show_ai_predictions_page():
    try:
        from prediction_rf import show_prediction_rf
        show_prediction_rf()
    except ModuleNotFoundError as e:
        st.markdown("# 🤖 AI Predictions")
        st.warning(f"Module introuvable : `{e.name}`")
    except FileNotFoundError as e:
        st.markdown("# 🤖 AI Predictions")
        st.error(f"**Fichier introuvable :** `{e.filename}`\n\n"
                 "Vérifiez :\n"
                 "- `models/kiln_xgb_model.pkl`\n"
                 "- `data/cleaned/Allarmi_TRAIN_READY.csv`")
    except Exception as exc:
        st.markdown("# 🤖 AI Predictions")
        st.error(f"Erreur : {exc}")
def show_live_alarms_page(df_al: pd.DataFrame, csv_path: Path):
    st.markdown("# 🚨 Alarmes — Four  🟢 Live")
    st.caption(f"Source : `{csv_path.name}`")
    if df_al is None or df_al.empty:
        st.info("Aucune alarme dans le fichier du jour.")
        return
    df = df_al.copy()
    display_cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=display_cols)
    k1, k2 = st.columns(2)
    with k1: kpi("Alarmes aujourd'hui", str(len(df)))
    with k2: kpi("Types distincts", str(df["ID_alarme"].nunique()) if "ID_alarme" in df.columns else "—")
    st.markdown("##### Détail des alarmes du jour")
    st.dataframe(df, use_container_width=True, hide_index=True, height=420)
    if "ID_alarme" in df.columns:
        st.markdown("##### Alarmes les plus fréquentes")
        top = (
            df.groupby(["ID_alarme", "Description"])["nbre_occurence"]
            .sum()
            .reset_index()
            .sort_values("nbre_occurence", ascending=False)
            .head(10)
        )
        bar = alt.Chart(top).mark_bar().encode(
            x=alt.X("nbre_occurence:Q", title="Occurrences"),
            y=alt.Y("Description:N", sort="-x", title=None),
            tooltip=["ID_alarme:Q", "Description:N", "nbre_occurence:Q"],
        ).properties(height=300)
        st.altair_chart(bar, use_container_width=True)
def show_anomalies_page_wrapper():
    try:
        from anomalies_four import show_anomalies_page
        show_anomalies_page()
    except ModuleNotFoundError as e:
        st.markdown("# 🔴 Anomalies")
        st.warning(f"Module introuvable : `{e.name}`")
    except FileNotFoundError as e:
        st.markdown("# 🔴 Anomalies")
        st.error(f"**Fichier introuvable :** `{e.filename}`\n\n"
                 "Vérifiez `cjo-maintenance-data-cleaned/four_anomalies_results.csv`.")
    except Exception as exc:
        st.markdown("# 🔴 Anomalies")
        st.error(f"Erreur : {exc}")
_LIVE_BADGE = {
    "GREEN":  ("✅", "#d4edda", "#28a745"),
    "YELLOW": ("⚠️", "#fff3cd", "#ffc107"),
    "ORANGE": ("🟠", "#ffe5d0", "#fd7e14"),
    "RED":    ("🚨", "#f8d7da", "#dc3545"),
}
def show_live_inference_page():
    st.markdown("# ⚡ Live Inference")
    st.caption("Horizon: **6 h** · Cadence: **1 h** — prediction for the current hour")
    result = st.session_state.get("_ai_result")
    error  = st.session_state.get("_ai_error")
    if result is None:
        st.error(f"Modèle non disponible : {error or 'résultat absent'}")
        if st.button("🔄 Actualiser"):
            st.cache_data.clear()
            st.rerun()
        return
    rec   = result["recommendation"]
    feats = result["features_used"]
    t     = result["timestamp"]
    level = rec.get("alert_level", "GREEN")
    icon, bg, border = _LIVE_BADGE.get(level, _LIVE_BADGE["GREEN"])
    st.markdown(
        f"""<div style="background:{bg};border:2px solid {border};
            border-radius:10px;padding:14px 20px;margin-bottom:12px">
          <span style="font-size:2rem">{icon}</span>
          <span style="font-size:1.4rem;font-weight:700;color:#111;
                       margin-left:10px">{level}</span>
          <span style="float:right;font-size:1.1rem;color:#333;margin-top:4px">
            {rec.get('probability', result.get('probability', 0))*100:.1f}% risk
          </span>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown(f"**{rec.get('headline', '')}**")
    cov_end   = t + datetime.timedelta(hours=6)
    next_pred = t + datetime.timedelta(hours=1)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted at",    t.strftime("%Y-%m-%d %H:%M"))
    m2.metric("Coverage window", f"{t.strftime('%H:%M')} → {cov_end.strftime('%H:%M')}")
    m3.metric("Prediction",      "ALERT" if result["prediction"] else "SAFE")
    m4.metric("Next update",     next_pred.strftime("%H:%M"))
    fiche = rec.get("fiche")
    if fiche and level != "GREEN":
        with st.expander("📋 Maintenance Fiche", expanded=True):
            fc1, fc2 = st.columns(2)
            with fc1:
                st.markdown(f"**{fiche.get('description', '')}**")
                st.markdown(f"- **Type:** {fiche.get('type', '')}")
                st.markdown(f"- **Urgence:** {fiche.get('urgence', '')}")
                st.markdown(f"- **Catégorie:** {fiche.get('category', '')}")
                latest = rec.get("latest_alarm", {})
                if latest:
                    st.markdown(
                        f"- **Dernière alarme:** {latest.get('description', '')} "
                        f"*(codice: {latest.get('codice', '')})*"
                    )
            with fc2:
                st.markdown("**Symptômes détectés**")
                for s in fiche.get("symptomes", []):
                    st.markdown(f"- {s}")
            st.markdown("**Actions immédiates**")
            for a in fiche.get("actions", []):
                st.markdown(f"- **{a}**")
            st.markdown("**Actions préventives**")
            for p in fiche.get("preventives", []):
                st.markdown(f"- {p}")
    with st.expander("🔔 Contexte de la dernière alarme"):
        st.json({
            "description":  feats.get("latest_alarm_description", ""),
            "codice":       feats.get("latest_alarm_codice", ""),
            "failure_type": feats.get("latest_alarm_failure_type", ""),
            "urgency":      feats.get("latest_alarm_urgency", ""),
        })
    with st.expander("📊 Vecteur de features (25 variables ML)", expanded=False):
        ml = {k: v for k, v in feats.items() if not k.startswith("latest_alarm_")}
        st.dataframe(
            pd.DataFrame.from_dict(ml, orient="index", columns=["value"])
              .reset_index().rename(columns={"index": "feature"}),
            hide_index=True,
            use_container_width=True,
        )
    if st.button("🔄 Actualiser"):
        st.cache_data.clear()
        st.rerun()
def show_feature_builder_page():
    st.markdown("# 🧪 Feature Builder — Validation")
    st.markdown(
        "Valide `build_features_for_timestamp()` sur `Allarmi_FEATURES.csv` "
        "et vérifie l'accord exact sur les 25 features ML."
    )
    TRAIN_PATH    = str(DATA_DIR / "Allarmi_TRAIN_READY.csv")
    FEATURES_PATH = str(BASE_DIR / "data" / "features" / "Allarmi_FEATURES.csv")
    RTOL = 1e-6
    if st.button("▶ Lancer la validation"):
        try:
            from scripts.feature_builder import build_features_for_timestamp, FEATURE_COLUMNS
        except ImportError:
            st.error("feature_builder non trouvé — vérifiez scripts/feature_builder.py.")
            return
        ML_COLS = [c for c in FEATURE_COLUMNS if not c.startswith("latest_alarm_")]
        with st.spinner("Chargement des données…"):
            alarm_hist = pd.read_csv(TRAIN_PATH)
            alarm_hist["Inizio"] = pd.to_datetime(alarm_hist["Inizio"], errors="coerce")
            alarm_hist = (
                alarm_hist.dropna(subset=["Inizio"])
                .sort_values("Inizio")
                .reset_index(drop=True)
            )
            feat_ref = pd.read_csv(FEATURES_PATH)
            feat_ref["timestamp"] = pd.to_datetime(feat_ref["timestamp"])
        n = len(feat_ref)
        test_rows = feat_ref.iloc[[0, n // 2, n - 1]].reset_index(drop=True)
        all_pass = True
        for _, ref_row in test_rows.iterrows():
            t = ref_row["timestamp"]
            computed = build_features_for_timestamp(t, alarm_hist)
            failures = []
            for col in ML_COLS:
                rv, cv = ref_row[col], computed[col]
                ok = (abs(float(cv) - float(rv)) <= RTOL * max(1.0, abs(float(rv)))
                      if isinstance(rv, float) or isinstance(cv, float)
                      else cv == rv)
                if not ok:
                    failures.append({"Feature": col, "Expected": rv, "Got": cv})
            if failures:
                all_pass = False
                st.error(f"FAIL — {t}")
                st.dataframe(pd.DataFrame(failures), use_container_width=True)
            else:
                st.success(f"PASS — {t}")
        st.markdown("---")
        if all_pass:
            st.success("Résultat global : PASS — tous les timestamps concordent.")
        else:
            st.error("Résultat global : FAIL — voir les écarts ci-dessus.")
def show_dashboard_four():
    _inject_css()
    page = render_sidebar()
    try:
        df_temp  = load_temperature()
        df_vel   = load_vitesse()
        df_alarm = load_alarms_four()
    except FileNotFoundError as e:
        st.error(f"**Fichier introuvable :** `{e.filename}`")
        st.stop()
    except Exception as e:
        st.error(f"Erreur :\n\n```\n{e}\n```")
        st.stop()
    all_dates = sorted(df_temp["timestamp"].dt.date.unique())
    if page == "📊 Vue Générale":
        show_main_page(df_temp, df_vel, all_dates, df_alarm)
    elif page == "🚨 Alarmes":
        show_alarms_page(df_alarm, all_dates)
    elif page == "🔴 Anomalies":
        show_anomalies_page_wrapper()
    elif page == "🔮 Prédiction Temps Réel":
        from page_prediction_four import show_prediction_four
        show_prediction_four()
    elif page == "🤖 AI Predictions":
        show_ai_predictions_page()

