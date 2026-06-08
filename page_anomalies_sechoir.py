from pathlib import Path
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
DATA_DIR      = Path(__file__).parent.parent / "cjo-maintenance-data-cleaned"
RESULTS_FILE  = DATA_DIR / "sechoir_anomalies_results.csv"
COLOR_NORMAL  = "#3aa1ff"
COLOR_ANOMALY = "#e74c3c"
COLOR_OK      = "#27ae60"
TEMP_SENSORS  = ["Temp_Module1", "Temp_Module2", "Temp_Module3", "Temp_Bruleur3"]
FAILURE_LABELS = {
    "Normal":           "✅ Normal",
    "Burner_Failure":   "🔥 Brûleurs",
    "Temp_Deviation":   "🌡️ Déviation Temp.",
    "Gradient_Anomaly": "📈 Gradient",
}
@st.cache_data(show_spinner=False)
def load_anomalies():
    df = pd.read_csv(RESULTS_FILE, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    for col in ("anomaly_pred", "true_label"):
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    avail = [c for c in TEMP_SENSORS if c in df.columns]
    if avail:
        df["temp_mean"] = df[avail].mean(axis=1)
    return df
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy  = (tp + tn) / max(len(y_true), 1)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=precision, recall=recall, f1=f1, accuracy=accuracy)
def show_anomalies_sechoir():
    from dashboard_sechoir import kpi
    st.markdown("# 🔍 Détection d'anomalies — Séchoir")
    st.caption("Modèle : **Isolation Forest** appliqué aux capteurs thermiques du séchoir céramique.")
    try:
        df = load_anomalies()
    except FileNotFoundError as e:
        st.error(f"**Fichier introuvable :** `{e.filename}`\n\n"
                 "Vérifiez `cjo-maintenance-data-cleaned/sechoir_anomalies_results.csv`.")
        return
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        return
    if df.empty:
        st.warning("Le fichier de résultats est vide.")
        return
    all_dates = sorted(df["timestamp"].dt.date.unique())
    f1, f2 = st.columns([2, 2])
    with f1:
        date_range = st.date_input(
            "📅 Période d'analyse",
            value=(all_dates[0], all_dates[-1]),
            min_value=all_dates[0],
            max_value=all_dates[-1],
        )
    with f2:
        view_mode = st.radio(
            "Affichage des points",
            options=["Tous", "Anomalies seulement", "Normaux seulement"],
            horizontal=True,
            index=0,
        )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range
    mask     = ((df["timestamp"].dt.date >= start_date) &
                (df["timestamp"].dt.date <= end_date))
    filtered = df[mask].copy()
    if filtered.empty:
        st.info("Aucune donnée dans cette période.")
        return
    total     = len(filtered)
    n_pred    = int(filtered["anomaly_pred"].sum())
    pred_rate = n_pred / total * 100 if total else 0.0
    last_anom = filtered.loc[filtered["anomaly_pred"] == 1, "timestamp"].max()
    last_anom_s = last_anom.strftime("%d/%m %H:%M") if pd.notna(last_anom) else "—"
    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi("Points analysés",    f"{total:,}".replace(",", " "))
    with k2: kpi("Anomalies prédites", f"{n_pred:,}".replace(",", " "))
    with k3: kpi("Taux d'anomalie",    f"{pred_rate:.1f} %")
    with k4: kpi("Dernière anomalie",  last_anom_s)
    st.markdown("---")
    if "true_label" in filtered.columns:
        st.markdown("### 📐 Performance du modèle (vs vérité terrain)")
        m = compute_metrics(filtered["true_label"].values, filtered["anomaly_pred"].values)
        e1, e2, e3, e4 = st.columns(4)
        with e1: kpi("Accuracy",  f"{m['accuracy']*100:.1f} %")
        with e2: kpi("Précision", f"{m['precision']*100:.1f} %")
        with e3: kpi("Rappel",    f"{m['recall']*100:.1f} %")
        with e4: kpi("F1-score",  f"{m['f1']*100:.1f} %")
        cm_col, leg_col = st.columns([1.4, 1])
        with cm_col:
            st.markdown("##### Matrice de confusion")
            cm_df = pd.DataFrame({
                "Réel":   ["Normal",   "Normal",   "Anomalie", "Anomalie"],
                "Prédit": ["Normal",   "Anomalie", "Normal",   "Anomalie"],
                "count":  [m["tn"],    m["fp"],    m["fn"],    m["tp"]],
                "kind":   ["TN",       "FP",       "FN",       "TP"],
            })
            cm_chart = alt.Chart(cm_df).mark_rect().encode(
                x=alt.X("Prédit:N", title="Prédiction"),
                y=alt.Y("Réel:N",   title="Réalité"),
                color=alt.Color("count:Q", scale=alt.Scale(scheme="blues"), legend=None),
                tooltip=["Réel:N", "Prédit:N", "count:Q", "kind:N"],
            )
            cm_text = alt.Chart(cm_df).mark_text(
                fontSize=18, fontWeight="bold", color="white"
            ).encode(x="Prédit:N", y="Réel:N", text="count:Q")
            st.altair_chart((cm_chart + cm_text).properties(height=240),
                            use_container_width=True)
        with leg_col:
            st.markdown("##### Lecture")
            st.markdown(
                f"- **TP** (anomalies bien détectées) : `{m['tp']}`  \n"
                f"- **FP** (fausses alertes) : `{m['fp']}`  \n"
                f"- **FN** (anomalies manquées) : `{m['fn']}`  \n"
                f"- **TN** (normaux bien classés) : `{m['tn']}`"
            )
        st.markdown("---")
    st.markdown("### 📈 Température moyenne et anomalies détectées")
    if view_mode == "Anomalies seulement":
        plot_df = filtered[filtered["anomaly_pred"] == 1]
    elif view_mode == "Normaux seulement":
        plot_df = filtered[filtered["anomaly_pred"] == 0]
    else:
        plot_df = filtered
    y_col = "temp_mean" if "temp_mean" in filtered.columns else "anomaly_score"
    y_title = "Température moyenne (°C)" if y_col == "temp_mean" else "Score d'anomalie"
    base = alt.Chart(filtered).mark_line(
        color=COLOR_NORMAL, opacity=0.55, strokeWidth=1.2,
    ).encode(
        x=alt.X("timestamp:T", title=None),
        y=alt.Y(f"{y_col}:Q", title=y_title).scale(zero=False),
    )
    anom_pts = alt.Chart(plot_df[plot_df["anomaly_pred"] == 1]).mark_point(
        size=45, filled=True, color=COLOR_ANOMALY, opacity=0.85,
    ).encode(
        x="timestamp:T",
        y=f"{y_col}:Q",
        tooltip=[
            alt.Tooltip("timestamp:T",       title="Date"),
            alt.Tooltip(f"{y_col}:Q",        title=y_title,  format=".2f"),
            alt.Tooltip("anomaly_score:Q",   title="Score",  format=".3f"),
            alt.Tooltip("failure_type:N",    title="Type"),
        ],
    )
    st.altair_chart((base + anom_pts).properties(height=380), use_container_width=True)
    st.markdown("### 📊 Évolution du score d'anomalie")
    score_line = alt.Chart(filtered).mark_area(
        opacity=0.55, color=COLOR_ANOMALY,
    ).encode(
        x=alt.X("timestamp:T", title=None),
        y=alt.Y("anomaly_score:Q", title="Score d'anomalie"),
        tooltip=[
            alt.Tooltip("timestamp:T",     title="Date"),
            alt.Tooltip("anomaly_score:Q", format=".3f"),
        ],
    )
    st.altair_chart(score_line.properties(height=220), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Distribution du score d'anomalie")
        hist_norm = alt.Chart(
            filtered[filtered["anomaly_pred"] == 0]
        ).mark_bar(color=COLOR_OK, opacity=0.75).encode(
            x=alt.X("anomaly_score:Q", bin=alt.Bin(maxbins=40), title="Score"),
            y=alt.Y("count():Q", title="Fréquence"),
        )
        hist_anom = alt.Chart(
            filtered[filtered["anomaly_pred"] == 1]
        ).mark_bar(color=COLOR_ANOMALY, opacity=0.75).encode(
            x=alt.X("anomaly_score:Q", bin=alt.Bin(maxbins=40)),
            y="count():Q",
        )
        st.altair_chart((hist_norm + hist_anom).properties(height=280),
                        use_container_width=True)
    with c2:
        st.markdown("##### Anomalies par jour")
        daily = (
            filtered[filtered["anomaly_pred"] == 1]
            .groupby(filtered["timestamp"].dt.date)
            .size()
            .reset_index(name="count")
            .rename(columns={"timestamp": "date"})
        )
        if daily.empty:
            st.info("Aucune anomalie détectée sur la période.")
        else:
            bar = alt.Chart(daily).mark_bar(color=COLOR_ANOMALY).encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("count:Q", title="Nb anomalies"),
                tooltip=["date:T", "count:Q"],
            ).properties(height=280)
            st.altair_chart(bar, use_container_width=True)
    if "true_label" in filtered.columns:
        st.markdown("### 🔗 Corrélation anomalies prédites ↔ vérité terrain")
        co = (
            filtered.groupby(["anomaly_pred", "true_label"])
            .size().reset_index(name="count")
        )
        co["Anomalie prédite"] = co["anomaly_pred"].map({0: "Non", 1: "Oui"})
        co["Anomalie réelle"]  = co["true_label"].map({0: "Non", 1: "Oui"})
        heat = alt.Chart(co).mark_rect().encode(
            x=alt.X("Anomalie réelle:N"),
            y=alt.Y("Anomalie prédite:N"),
            color=alt.Color("count:Q", scale=alt.Scale(scheme="oranges"), legend=None),
            tooltip=["Anomalie prédite:N", "Anomalie réelle:N", "count:Q"],
        )
        heat_text = alt.Chart(co).mark_text(
            fontSize=16, fontWeight="bold", color="black"
        ).encode(x="Anomalie réelle:N", y="Anomalie prédite:N", text="count:Q")
        st.altair_chart((heat + heat_text).properties(height=220), use_container_width=True)
    st.markdown("### 📋 Détail des anomalies détectées")
    anom_only = filtered[filtered["anomaly_pred"] == 1].copy()
    if anom_only.empty:
        st.info("Aucune anomalie à afficher.")
        return
    rename_map = {
        "timestamp":     "Horodatage",
        "anomaly_score": "Score",
        "failure_type":  "Type de panne",
        "true_label":    "Vérité terrain",
    }
    if "temp_mean" in anom_only.columns:
        rename_map["temp_mean"] = "Température moy. (°C)"
    cols_show = [c for c in rename_map if c in anom_only.columns]
    table = anom_only[cols_show].rename(columns=rename_map).copy()
    table["Score"] = table["Score"].round(4)
    if "Température moy. (°C)" in table.columns:
        table["Température moy. (°C)"] = table["Température moy. (°C)"].round(2)
    if "Vérité terrain" in table.columns:
        table["Vérité terrain"] = table["Vérité terrain"].map({1: "Anomalie", 0: "Normal"})
    if "Type de panne" in table.columns:
        table["Type de panne"] = table["Type de panne"].map(FAILURE_LABELS).fillna(table["Type de panne"])
    st.dataframe(
        table.sort_values("Horodatage", ascending=False),
        use_container_width=True,
        hide_index=True,
        height=420,
    )

