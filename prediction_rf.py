import pickle
from pathlib import Path
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from dashboard_four import (
    MODEL_FEATURE_NAMES,
    build_feature_matrix,
    load_alarms_four,
    kpi,
)
BASE_DIR   = Path(__file__).parent
MODEL_FILE = BASE_DIR / "models" / "kiln_xgb_model.pkl"
MODEL_METRICS  = {"accuracy": 0.98, "precision": 0.99, "recall": 0.97, "f1": 0.98}
URGENCY_COLORS = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#27ae60"}
@st.cache_resource
def load_model():
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)
@st.cache_data(show_spinner="Génération des prédictions…")
def generate_predictions(_model):
    df_alarm = load_alarms_four()
    t_start = df_alarm["Inizio"].min().floor("h")
    t_end   = df_alarm["Inizio"].max().ceil("h")
    grid    = pd.date_range(start=t_start, end=t_end, freq="1h")
    X     = build_feature_matrix(grid, df_alarm)
    proba = _model.predict_proba(X)[:, 1]
    out = pd.DataFrame({
        "timestamp":              grid,
        "proba":                  proba,
        "prediction":             (proba >= 0.5).astype(int),
        "alarms_last_6h":         X["alarms_last_6h"].values,
        "high_last_6h":           X["high_last_6h"].values,
        "minutes_since_last_alarm": X["minutes_since_last_alarm"].values,
    })
    out["urgency"] = np.where(proba >= 0.70, "High",
                       np.where(proba >= 0.40, "Medium", "Low"))
    return out, X
def show_prediction_rf():
    st.markdown("# 🤖 AI Predictions — Historic Mode")
    st.caption("Prédiction d'alarmes critiques à un horizon de 6 heures (modèle XGBoost).")
    if not MODEL_FILE.exists():
        st.error(f"Modèle introuvable : `{MODEL_FILE}`")
        return
    model = load_model()
    df, X = generate_predictions(model)
    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi("Model Accuracy",    f"{MODEL_METRICS['accuracy']*100:.0f} %", f"F1 = {MODEL_METRICS['f1']:.2f}")
    with k2: kpi("Total Predictions", f"{len(df):,}".replace(",", " "))
    with k3: kpi("Critical Alerts",   f"{(df['urgency']=='High').sum():,}".replace(",", " "))
    with k4: kpi("Last Prediction",   df["timestamp"].max().strftime("%d/%m %H:%M"))
    st.markdown("---")
    st.markdown("### 📋 Historique des prédictions")
    urgency_filter = st.multiselect(
        "Filtrer par urgence",
        options=["High", "Medium", "Low"],
        default=["High", "Medium"],
    )
    filtered = df[df["urgency"].isin(urgency_filter)].copy()
    table = filtered[[
        "timestamp", "proba", "urgency", "alarms_last_6h", "high_last_6h"
    ]].rename(columns={
        "timestamp":      "Horodatage",
        "proba":          "Probabilité",
        "urgency":        "Urgence",
        "alarms_last_6h": "Alarmes 6h",
        "high_last_6h":   "High 6h",
    })
    table["Probabilité"] = table["Probabilité"].round(3)
    st.dataframe(
        table.sort_values("Horodatage", ascending=False),
        use_container_width=True,
        hide_index=True,
        height=360,
        column_config={
            "Probabilité": st.column_config.ProgressColumn(
                "Probabilité", format="%.3f", min_value=0.0, max_value=1.0,
            ),
        },
    )
    csv = table.to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 Exporter en CSV", data=csv,
                       file_name="predictions_four.csv", mime="text/csv")
    st.markdown("---")
    st.markdown("### 🧠 Pourquoi cette prédiction ?")
    if filtered.empty:
        st.info("Aucune prédiction à expliquer (élargissez les filtres).")
        return
    top_preds = filtered.sort_values("proba", ascending=False).head(50)
    labels = top_preds.apply(
        lambda r: f"{r['timestamp'].strftime('%d/%m %H:%M')}  —  "
                  f"P={r['proba']:.3f}  ({r['urgency']})",
        axis=1,
    ).tolist()
    selected_label = st.selectbox("Sélectionner une prédiction", options=labels)
    idx = labels.index(selected_label)
    selected_ts = top_preds.iloc[idx]["timestamp"]
    selected_X  = X.loc[selected_ts]
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(selected_X.values.reshape(1, -1))[0]
        shap_df = pd.DataFrame({
            "Feature": MODEL_FEATURE_NAMES,
            "Valeur":  selected_X.values,
            "SHAP":    shap_vals,
        })
        shap_df["Abs"] = shap_df["SHAP"].abs()
        shap_df = shap_df.sort_values("Abs", ascending=False).head(8)
        chart = alt.Chart(shap_df).mark_bar().encode(
            x=alt.X("SHAP:Q", title="Contribution (SHAP)"),
            y=alt.Y("Feature:N", sort="-x", title=None),
            color=alt.condition(
                alt.datum.SHAP > 0,
                alt.value(URGENCY_COLORS["High"]),
                alt.value(URGENCY_COLORS["Low"]),
            ),
            tooltip=["Feature", "Valeur", "SHAP"],
        ).properties(height=280)
        st.altair_chart(chart, use_container_width=True)
        st.caption("🔴 Rouge = augmente le risque  •  🟢 Vert = diminue le risque")
    except ImportError:
        imp = pd.DataFrame({
            "Feature":    MODEL_FEATURE_NAMES,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=False).head(8)
        chart = alt.Chart(imp).mark_bar(color="#3aa1ff").encode(
            x=alt.X("Importance:Q"),
            y=alt.Y("Feature:N", sort="-x"),
        ).properties(height=280)
        st.altair_chart(chart, use_container_width=True)
        st.caption("ℹ️ Installez `shap` pour des explications par prédiction.")
    st.markdown("---")
    st.markdown("### 📈 Évolution du risque dans le temps")
    line = alt.Chart(df).mark_area(opacity=0.55, color=URGENCY_COLORS["High"]).encode(
        x=alt.X("timestamp:T", title=None),
        y=alt.Y("proba:Q", title="P(High dans 6h)", scale=alt.Scale(domain=[0, 1])),
        tooltip=[
            alt.Tooltip("timestamp:T", title="Date"),
            alt.Tooltip("proba:Q", format=".3f"),
            alt.Tooltip("urgency:N"),
        ],
    )
    threshold = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(
        strokeDash=[5, 5], color="white", opacity=0.5,
    ).encode(y="y:Q")
    st.altair_chart((line + threshold).properties(height=300), use_container_width=True)

