import pickle
import smtplib
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from dashboard_four import (
    MODEL_FEATURE_NAMES,
    build_feature_matrix,
    load_alarms_four,
    load_temperature,
    kpi,
)
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR.parent / "cjo-maintenance-data-cleaned"
MODEL_FILE = BASE_DIR / "models" / "kiln_xgb_model.pkl"
HIST_FILE  = DATA_DIR / "prediction_history_four.csv"
REFRESH_INTERVAL = 30
DARK  = '#0f172a'
CARD  = '#1e293b'
BORD  = '#334155'
TEXT  = '#94a3b8'
WHITE = '#f1f5f9'
FAILURE_LABELS = {
    "temperature_failure":  "🌡️ Panne Thermique",
    "burner_failure":       "🔥 Panne Brûleurs",
    "pression_failure":     "💨 Panne Pression",
    "alimentation_failure": "⚡ Panne Alimentation",
    "No_Failure":           "✅ Aucune panne",
}
MAIL_CONFIG = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port":   587,
    "sender":      "cjoadmin@gmail.com",
    "password":    "cxhlmuudblbtxozj",
    "recipient":   "ahmed.naffeti10@gmail.com",
}
def _css():
    st.markdown("""
    <style>
    .kpi-card {
        background:#1e293b; border:1px solid #334155;
        border-radius:12px; padding:1.1rem 1.4rem;
        text-align:center; margin-bottom:0.5rem;
    }
    .kpi-label { color:#94a3b8; font-size:0.72rem; text-transform:uppercase;
        letter-spacing:0.08em; margin-bottom:0.3rem; }
    .kpi-value { font-size:1.8rem; font-weight:700; }
    .kpi-sub   { color:#64748b; font-size:0.70rem; margin-top:0.25rem; }
    .status-box {
        border-radius:16px; padding:1.6rem 2rem;
        text-align:center; margin-bottom:1rem;
    }
    .status-normal  { background:#052e1622; border:2px solid #22c55e; }
    .status-anormal { background:#45031722; border:2px solid #ef4444; }
    .status-icon  { font-size:3.5rem; line-height:1; margin-bottom:0.4rem; }
    .status-label { font-size:2rem; font-weight:800; letter-spacing:0.05em; }
    .status-sub   { color:#94a3b8; font-size:0.82rem; margin-top:0.4rem; }
    </style>
    """, unsafe_allow_html=True)
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None
@st.cache_data(show_spinner=False)
def load_history():
    if HIST_FILE.exists():
        return pd.read_csv(HIST_FILE, parse_dates=['timestamp'])
    return pd.DataFrame(columns=[
        'timestamp', 'prediction', 'probability', 'failure_type_pred', 'mail_sent',
    ])
def save_history(hist_df):
    hist_df.tail(500).to_csv(HIST_FILE, index=False)
def infer_failure_type(features_row: dict) -> str:
    """Déduit le type de panne le plus probable depuis les features."""
    counts = {
        "temperature_failure":  features_row.get("temp_fail_last_6h",    0),
        "burner_failure":       features_row.get("burner_fail_last_6h",  0),
        "pression_failure":     features_row.get("pression_fail_last_6h",0),
        "alimentation_failure": features_row.get("alim_fail_last_6h",    0),
    }
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else "temperature_failure"
MAINTENANCE_FICHES = {
    "temperature_failure": {
        "description": "Anomalie détectée sur le système de régulation thermique du four.",
        "actions": [
            "Vérifier les sondes de température (thermocouple zones 1 à 4)",
            "Contrôler le régulateur PID et les consignes de chauffe",
            "Inspecter les résistances chauffantes (continuité, isolement)",
            "Vérifier les câbles et bornes de raccordement thermocouple",
            "Comparer températures mesurées vs consignes SCADA",
        ],
        "composants": "Thermocouples · Régulateur PID · Résistances · Câblage",
        "priorite": "🔴 URGENTE — Intervention dans les 2 heures",
        "delai": "≤ 2 h",
    },
    "burner_failure": {
        "description": "Anomalie détectée sur le système de brûleurs du four céramique.",
        "actions": [
            "Vérifier l'alimentation gaz (pression, vanne principale)",
            "Contrôler l'état des électrovannes de brûleurs",
            "Inspecter les détecteurs de flamme (ionisation / UV)",
            "Vérifier le séquenceur de brûleur et les temporisations",
            "Nettoyer ou remplacer les buses d'injection si encrassées",
        ],
        "composants": "Brûleurs · Électrovannes gaz · Détecteurs de flamme · Séquenceur",
        "priorite": "🔴 URGENTE — Intervention dans les 2 heures",
        "delai": "≤ 2 h",
    },
    "pression_failure": {
        "description": "Anomalie détectée sur le circuit de pression du four.",
        "actions": [
            "Vérifier les pressostats et capteurs de pression",
            "Contrôler l'étanchéité des circuits (fuites gaz/air)",
            "Inspecter le compresseur et le réservoir tampon",
            "Vérifier les filtres à air (colmatage)",
            "Contrôler les vannes de régulation pression",
        ],
        "composants": "Pressostats · Compresseur · Filtres · Vannes de régulation",
        "priorite": "🟠 HAUTE — Intervention dans les 4 heures",
        "delai": "≤ 4 h",
    },
    "alimentation_failure": {
        "description": "Anomalie détectée sur le système d'alimentation électrique du four.",
        "actions": [
            "Vérifier le tableau électrique principal (disjoncteurs, fusibles)",
            "Contrôler les variateurs de fréquence (alarmes actives)",
            "Inspecter les contacteurs et relais de puissance",
            "Vérifier la tension secteur et le neutre (déséquilibre phases)",
            "Contrôler l'onduleur / alimentation de secours si présent",
        ],
        "composants": "Tableau électrique · Variateurs · Contacteurs · Alimentation secteur",
        "priorite": "🔴 URGENTE — Intervention dans les 2 heures",
        "delai": "≤ 2 h",
    },
    "No_Failure": {
        "description": "Aucune panne détectée — prédiction préventive.",
        "actions": ["Surveillance continue recommandée."],
        "composants": "—",
        "priorite": "🟢 NORMALE",
        "delai": "—",
    },
}
def _build_fiche_html(failure_type: str) -> str:
    fiche = MAINTENANCE_FICHES.get(failure_type, MAINTENANCE_FICHES["No_Failure"])
    actions_html = "".join(
        f'<li style="margin:6px 0;color:#f1f5f9;">{a}</li>'
        for a in fiche["actions"]
    )
    return f"""
    <div style="margin-top:20px;background:#0f172a;border-radius:10px;
                padding:18px 20px;border-left:4px solid #f97316;">
        <h3 style="color:#f97316;margin:0 0 10px 0;font-size:1rem;">
            📋 Fiche de Maintenance Recommandée
        </h3>
        <p style="color:#94a3b8;margin:0 0 12px 0;font-size:0.88rem;">{fiche['description']}</p>
        <table style="width:100%;border-collapse:collapse;margin-bottom:14px;">
            <tr>
                <td style="padding:8px;color:#64748b;font-size:0.82rem;width:38%;">Priorité d'intervention</td>
                <td style="padding:8px;color:#f1f5f9;font-weight:bold;font-size:0.88rem;">{fiche['priorite']}</td>
            </tr>
            <tr style="background:#1e293b;">
                <td style="padding:8px;color:#64748b;font-size:0.82rem;">Délai max</td>
                <td style="padding:8px;color:#ef4444;font-weight:bold;font-size:0.88rem;">{fiche['delai']}</td>
            </tr>
            <tr>
                <td style="padding:8px;color:#64748b;font-size:0.82rem;">Composants concernés</td>
                <td style="padding:8px;color:#94a3b8;font-size:0.82rem;">{fiche['composants']}</td>
            </tr>
        </table>
        <p style="color:#64748b;font-size:0.82rem;margin:0 0 8px 0;font-weight:600;text-transform:uppercase;
                  letter-spacing:0.05em;">Actions à effectuer :</p>
        <ol style="margin:0;padding-left:20px;">
            {actions_html}
        </ol>
    </div>
    """
_MAIL_STYLE = {
    "temperature_failure":  {"color": "#ef4444", "emoji": "🌡️", "intro": "Une <strong>panne thermique</strong> a été prédite sur le four céramique. Le système de régulation de température nécessite une vérification immédiate."},
    "burner_failure":       {"color": "#f97316", "emoji": "🔥", "intro": "Une <strong>panne brûleurs</strong> a été prédite sur le four céramique. Le système de combustion doit être inspecté rapidement."},
    "pression_failure":     {"color": "#3b82f6", "emoji": "💨", "intro": "Une <strong>panne de pression</strong> a été prédite sur le four céramique. Le circuit de pression gaz/air doit être contrôlé."},
    "alimentation_failure": {"color": "#8b5cf6", "emoji": "⚡", "intro": "Une <strong>panne d'alimentation électrique</strong> a été prédite sur le four céramique. Le tableau électrique doit être vérifié en urgence."},
    "No_Failure":           {"color": "#22c55e", "emoji": "✅", "intro": "Aucune panne détectée — prédiction préventive."},
}
def send_alert_mail(prob: float, failure_type: str, timestamp: str) -> bool:
    try:
        ft_label   = FAILURE_LABELS.get(failure_type, failure_type)
        fiche_html = _build_fiche_html(failure_type)
        style      = _MAIL_STYLE.get(failure_type, _MAIL_STYLE["temperature_failure"])
        color      = style["color"]
        emoji      = style["emoji"]
        intro      = style["intro"]
        ft_label_clean = ft_label.replace("🌡️","").replace("🔥","").replace("💨","").replace("⚡","").replace("✅","").strip()
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[CJO ALERTE] Four Céramique — {ft_label_clean} prédite ({prob*100:.0f}%)"
        msg["From"]    = MAIL_CONFIG["sender"]
        msg["To"]      = MAIL_CONFIG["recipient"]
        html = f"""
        <html><body style="font-family:Arial,sans-serif;background:#0f172a;color:#f1f5f9;padding:20px;">
        <div style="max-width:640px;margin:auto;background:#1e293b;border-radius:12px;
                    padding:24px;border:2px solid {color};">
            <h2 style="color:{color};margin-top:0;">{emoji} ALERTE {ft_label_clean.upper()} — Four Céramique</h2>
            <p style="color:#94a3b8;">{intro}<br>
            Niveau de risque : <strong style="color:{color}">HIGH</strong> &nbsp;|&nbsp;
            Probabilité : <strong style="color:{color}">{prob*100:.1f}%</strong></p>
            <table style="width:100%;border-collapse:collapse;margin:16px 0;">
                <tr style="background:#0f172a;">
                    <td style="padding:10px;color:#94a3b8;">Heure de prédiction</td>
                    <td style="padding:10px;color:#f1f5f9;font-weight:bold;">{timestamp}</td>
                </tr>
                <tr>
                    <td style="padding:10px;color:#94a3b8;">Type de panne prédit</td>
                    <td style="padding:10px;font-weight:bold;color:{color};">{ft_label}</td>
                </tr>
                <tr style="background:#0f172a;">
                    <td style="padding:10px;color:#94a3b8;">Probabilité de panne</td>
                    <td style="padding:10px;color:{color};font-weight:bold;">{prob*100:.1f}%</td>
                </tr>
                <tr>
                    <td style="padding:10px;color:#94a3b8;">Horizon de prédiction</td>
                    <td style="padding:10px;color:#f1f5f9;">6 heures</td>
                </tr>
                <tr style="background:#0f172a;">
                    <td style="padding:10px;color:#94a3b8;">Heure d'alerte</td>
                    <td style="padding:10px;color:#f1f5f9;">{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</td>
                </tr>
            </table>
            {fiche_html}
            <p style="color:#64748b;font-size:12px;margin-top:20px;margin-bottom:0;">
            CJO Poulina — Système de maintenance prédictive · Message automatique</p>
        </div></body></html>
        """
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP(MAIL_CONFIG["smtp_server"], MAIL_CONFIG["smtp_port"]) as server:
            server.starttls()
            server.login(MAIL_CONFIG["sender"], MAIL_CONFIG["password"])
            server.sendmail(MAIL_CONFIG["sender"], MAIL_CONFIG["recipient"], msg.as_string())
        return True
    except Exception:
        return False
_ACTIVE_ZONES = list(range(1, 41))
def show_live_temps(df_temp: pd.DataFrame):
    """Affiche les températures actuelles vs consignes depuis TREND_Temperature.csv."""
    if df_temp is None or df_temp.empty:
        st.info("Données de température non disponibles.")
        return
    last   = df_temp.iloc[-1]
    ts_str = last["timestamp"].strftime("%d/%m/%Y %H:%M") if pd.notna(last.get("timestamp")) else "—"
    rows = []
    for i in _ACTIVE_ZONES:
        tc, sc = f"Temp{i}", f"SetT{i}"
        if tc not in last.index or sc not in last.index:
            continue
        t_val = float(last[tc])
        s_val = float(last[sc])
        if s_val <= 0 and t_val <= 0:
            continue
        ecart   = t_val - s_val
        pct_dev = (ecart / s_val * 100) if s_val > 0 else 0.0
        rows.append({"Zone": f"Z{i}", "Temp": t_val, "Consigne": s_val,
                     "Écart": ecart, "Écart %": pct_dev})
    if not rows:
        st.info("Aucune zone active détectée.")
        return
    df_z = pd.DataFrame(rows)
    st.markdown(f"### 🌡️ Capteurs Température — Dernière mesure : `{ts_str}`")
    temp_vals = [r["Temp"]    for r in rows if r["Temp"] > 0]
    ecarts_ab = [abs(r["Écart"]) for r in rows]
    n_warn    = sum(1 for r in rows if abs(r["Écart %"]) > 10)
    tc1, tc2, tc3, tc4 = st.columns(4)
    with tc1: kpi("Temp max",          f"{max(temp_vals):.0f} °C" if temp_vals else "—", "zone active")
    with tc2: kpi("Temp moyenne",      f"{sum(temp_vals)/len(temp_vals):.0f} °C" if temp_vals else "—")
    with tc3: kpi("Écart max / cible", f"{max(ecarts_ab):.0f} °C" if ecarts_ab else "—")
    with tc4: kpi("Zones hors ±10%",   str(n_warn), "vs consigne",
                  "#EF4444" if n_warn > 0 else "#22C55E")
    colors = ["#EF4444" if abs(r["Écart %"]) > 10 else "#22C55E" for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_z["Zone"], y=df_z["Consigne"],
        name="Consigne (SetT)", marker_color="#3B82F6", opacity=0.65,
    ))
    fig.add_trace(go.Bar(
        x=df_z["Zone"], y=df_z["Temp"],
        name="Mesure (Temp)", marker_color=colors, opacity=0.88,
    ))
    fig.update_layout(
        barmode="overlay",
        plot_bgcolor=DARK, paper_bgcolor=DARK, font_color=TEXT,
        xaxis=dict(gridcolor=CARD, title="Zone"),
        yaxis=dict(gridcolor=CARD, title="Température (°C)"),
        legend=dict(bgcolor=CARD, bordercolor=BORD),
        height=300, margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    warn_df = df_z[abs(df_z["Écart %"]) > 5].sort_values("Écart %", key=abs, ascending=False)
    if not warn_df.empty:
        with st.expander(f"⚠️ {len(warn_df)} zone(s) avec écart > 5%", expanded=n_warn > 0):
            disp = warn_df.copy()
            disp["Temp"]    = disp["Temp"].round(1).astype(str)    + " °C"
            disp["Consigne"]= disp["Consigne"].round(1).astype(str)+ " °C"
            disp["Écart"]   = disp["Écart"].round(1).astype(str)   + " °C"
            disp["Écart %"] = disp["Écart %"].round(1).astype(str) + " %"
            st.dataframe(disp[["Zone","Temp","Consigne","Écart","Écart %"]],
                         use_container_width=True, hide_index=True)
    st.markdown("#### 📈 Tendance récente — zones clés")
    KEY_ZONES = ["Temp9", "Temp17", "Temp25", "Temp29", "Temp33"]
    key_avail = [c for c in KEY_ZONES if c in df_temp.columns]
    if key_avail:
        cutoff   = df_temp["timestamp"].max() - pd.Timedelta(hours=4)
        recent   = df_temp[df_temp["timestamp"] >= cutoff][["timestamp"] + key_avail].copy()
        if not recent.empty:
            melted = recent.melt(id_vars="timestamp", var_name="Capteur", value_name="°C")
            trend  = go.Figure()
            palette = ["#3B82F6","#F97316","#22C55E","#EF4444","#8B5CF6"]
            for idx, cap in enumerate(key_avail):
                sub = melted[melted["Capteur"] == cap]
                trend.add_trace(go.Scatter(
                    x=sub["timestamp"], y=sub["°C"],
                    mode="lines", name=cap,
                    line=dict(color=palette[idx % len(palette)], width=1.8),
                ))
            trend.update_layout(
                plot_bgcolor=DARK, paper_bgcolor=DARK, font_color=TEXT,
                xaxis=dict(gridcolor=CARD, title=None),
                yaxis=dict(gridcolor=CARD, title="°C"),
                legend=dict(bgcolor=CARD, bordercolor=BORD, orientation="h",
                            yanchor="bottom", y=1.02),
                height=280, margin=dict(t=30, b=10),
            )
            st.plotly_chart(trend, use_container_width=True)
    st.markdown("---")
def run_prediction(model):
    """Prédit pour le timestamp courant à partir de l'historique des alarmes."""
    df_alarm = load_alarms_four()
    now      = pd.Timestamp.now()
    X = build_feature_matrix([now], df_alarm)
    pred  = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])
    features_row = X.iloc[0].to_dict()
    failure_type = "No_Failure"
    if pred == 1:
        failure_type = infer_failure_type(features_row)
    return pred, proba, failure_type, features_row
def plot_history(hist_df):
    if len(hist_df) < 2:
        st.info("Historique insuffisant — lancez au moins 2 prédictions.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_df['timestamp'], y=hist_df['probability'],
        mode='lines+markers', name='Probabilité anomalie',
        line=dict(color='#3B82F6', width=2),
        marker=dict(
            color=np.where(hist_df['prediction'] == 1, '#EF4444', '#22C55E'),
            size=8, line=dict(color=DARK, width=1),
        ),
        hovertemplate='%{x|%H:%M}<br>Prob: %{y:.1%}<extra></extra>',
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#F97316",
                  line_width=1.5, annotation_text="Seuil 50%",
                  annotation_font_color="#F97316")
    anom = hist_df[hist_df['prediction'] == 1]
    if not anom.empty:
        fig.add_trace(go.Scatter(
            x=anom['timestamp'], y=anom['probability'],
            mode='markers', name='ANORMAL détecté',
            marker=dict(color='#EF4444', size=12, symbol='x',
                        line=dict(color='#EF4444', width=2)),
            hovertemplate='%{x|%H:%M} — ANORMAL<br>Prob: %{y:.1%}<extra></extra>',
        ))
    fig.update_layout(
        title="Historique des probabilités d'anomalie",
        plot_bgcolor=DARK, paper_bgcolor=DARK, font_color=TEXT,
        xaxis=dict(gridcolor=CARD, title='Heure'),
        yaxis=dict(gridcolor=CARD, title='Probabilité', range=[0, 1],
                   tickformat='.0%'),
        legend=dict(bgcolor=CARD, bordercolor=BORD),
        hovermode='x unified', height=380,
    )
    st.plotly_chart(fig, use_container_width=True)
def show_prediction_four():
    _css()
    st.markdown("## 🔮 Prédiction Temps Réel — Four Céramique")
    st.markdown(
        f"<div style='color:{TEXT};font-size:0.85rem;margin-bottom:1.5rem;'>"
        "XGBoost &nbsp;|&nbsp; Horizon 6h &nbsp;|&nbsp;"
        f" Actualisation auto toutes les {REFRESH_INTERVAL} min</div>",
        unsafe_allow_html=True,
    )
    model = load_model()
    if model is None:
        st.error(f"Modèle introuvable : `{MODEL_FILE}`")
        return
    if 'pred_history_four'   not in st.session_state:
        st.session_state.pred_history_four   = load_history()
    if 'last_pred_time_four' not in st.session_state:
        st.session_state.last_pred_time_four = 0.0
    if 'last_result_four'    not in st.session_state:
        st.session_state.last_result_four    = None
    if 'mail_enabled_four'   not in st.session_state:
        st.session_state.mail_enabled_four   = True
    if 'last_mail_ts_four'   not in st.session_state:
        st.session_state.last_mail_ts_four   = None
    col_btn, col_mail, col_next = st.columns([2, 2, 3])
    with col_btn:
        force_pred = st.button("🔄 Lancer la prédiction",
                               use_container_width=True, type="primary")
    with col_mail:
        st.session_state.mail_enabled_four = st.toggle(
            "📧 Alertes mail", value=st.session_state.mail_enabled_four)
    with col_next:
        elapsed_now = time.time() - st.session_state.last_pred_time_four
        remaining   = max(0, int(REFRESH_INTERVAL * 60 - elapsed_now))
        components.html(f"""
        <div style='color:#94a3b8;font-size:0.8rem;padding-top:0.6rem;font-family:sans-serif;'>
            Prochaine auto :
            <strong style='color:#f1f5f9;' id='cd'>
                {remaining//60:02d}:{remaining%60:02d}
            </strong>
        </div>
        <script>
        var el = document.getElementById('cd');
        var secs = {remaining};
        setInterval(function() {{
            if (secs > 0) secs--;
            var m = Math.floor(secs / 60);
            var s = secs % 60;
            el.textContent = (m<10?'0':'')+m+':'+(s<10?'0':'')+s;
        }}, 1000);
        </script>
        """, height=40)
    st.markdown("---")
    elapsed        = time.time() - st.session_state.last_pred_time_four
    auto_trigger   = elapsed >= (REFRESH_INTERVAL * 60)
    first_load     = st.session_state.last_result_four is None
    should_predict = force_pred or (auto_trigger and not first_load) or first_load
    if should_predict:
        try:
            pred, proba, failure_type, features_row = run_prediction(model)
            ts_now = datetime.now()
            ts_str = ts_now.strftime('%d/%m/%Y %H:%M:%S')
            mail_sent      = False
            mail_attempted = False
            is_new_ts = (st.session_state.last_mail_ts_four != ts_str)
            if pred == 1 and st.session_state.mail_enabled_four and is_new_ts:
                mail_attempted = True
                mail_sent = send_alert_mail(proba, failure_type, ts_str)
                if mail_sent:
                    st.session_state.last_mail_ts_four = ts_str
            result = {
                'prediction':        pred,
                'probability':       proba,
                'failure_type_pred': failure_type,
                'features_row':      features_row,
                'timestamp':         ts_now,
                'mail_sent':         mail_sent,
                'mail_attempted':    mail_attempted,
            }
            st.session_state.last_result_four    = result
            st.session_state.last_pred_time_four = time.time()
            new_row = pd.DataFrame([{
                'timestamp':         ts_now,
                'prediction':        pred,
                'probability':       proba,
                'failure_type_pred': failure_type,
                'mail_sent':         mail_sent,
            }])
            st.session_state.pred_history_four = pd.concat(
                [st.session_state.pred_history_four, new_row], ignore_index=True)
            save_history(st.session_state.pred_history_four)
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
            return
    res = st.session_state.last_result_four
    if res is None:
        return
    pred  = res['prediction']
    proba = res['probability']
    ft    = res['failure_type_pred']
    ts    = res['timestamp'].strftime('%H:%M:%S')
    if pred == 0:
        st.markdown(f"""
        <div class="status-box status-normal">
            <div class="status-icon">🟢</div>
            <div class="status-label" style="color:#22c55e;">NORMAL</div>
            <div class="status-sub">Aucune anomalie prévue dans les 6 prochaines heures</div>
        </div>""", unsafe_allow_html=True)
    else:
        ft_label = FAILURE_LABELS.get(ft, ft)
        st.markdown(f"""
        <div class="status-box status-anormal">
            <div class="status-icon">🔴</div>
            <div class="status-label" style="color:#ef4444;">ANORMAL — HIGH</div>
            <div class="status-sub">Anomalie probable dans les 6 prochaines heures · {ft_label}</div>
        </div>""", unsafe_allow_html=True)
        if res.get('mail_sent'):
            st.success("📧 Alerte mail envoyée au responsable maintenance.")
        elif res.get('mail_attempted'):
            st.warning("📧 Mail non envoyé (vérifiez la configuration SMTP).")
    c1, c2, c3, c4 = st.columns(4)
    hist = st.session_state.pred_history_four
    n_anom = int((hist['prediction'] == 1).sum()) if len(hist) > 0 else 0
    with c1:
        kpi("Probabilité anomalie", f"{proba*100:.1f}%",
            "HIGH dans 6h", "#EF4444" if proba >= 0.5 else "#22C55E")
    with c2:
        kpi("Type de panne prédit",
            FAILURE_LABELS.get(ft, ft).replace("🌡️","").replace("🔥","")
              .replace("💨","").replace("⚡","").replace("✅","").strip(),
            "", "#F97316" if pred == 1 else TEXT)
    with c3:
        kpi("Anomalies détectées", str(n_anom),
            f"sur {len(hist)} prédictions", "#EF4444" if n_anom > 0 else "#22C55E")
    with c4:
        kpi("Dernière mise à jour", ts, datetime.now().strftime('%d/%m/%Y'))
    st.markdown("---")
    try:
        df_temp = load_temperature()
        show_live_temps(df_temp)
    except Exception as _e:
        st.warning(f"Capteurs température indisponibles : {_e}")
    with st.expander("🔎 Détail des 25 features utilisées", expanded=False):
        feat_df = pd.DataFrame(
            list(res['features_row'].items()),
            columns=['Feature', 'Valeur']
        )
        col_l, col_r = st.columns(2)
        mid = len(feat_df) // 2
        with col_l:
            st.dataframe(feat_df.iloc[:mid], use_container_width=True, hide_index=True)
        with col_r:
            st.dataframe(feat_df.iloc[mid:], use_container_width=True, hide_index=True)
    st.markdown("### 📈 Historique des prédictions")
    plot_history(hist.copy())
    if not hist.empty:
        st.markdown("### 📋 Journal des prédictions")
        disp = hist.copy().sort_values('timestamp', ascending=False).head(50)
        disp['état']  = disp['prediction'].map({0: '🟢 Normal', 1: '🔴 Anormal'})
        disp['prob']  = (disp['probability'] * 100).round(1).astype(str) + '%'
        disp['type']  = disp['failure_type_pred'].map(FAILURE_LABELS).fillna(disp['failure_type_pred'])
        disp['mail']  = disp['mail_sent'].map({True: '✅', False: '—'})
        def niveau(p):
            if p >= 0.80: return '🔴 HIGH'
            if p >= 0.50: return '🟠 MEDIUM'
            return '🟢 LOW'
        disp['niveau'] = disp['probability'].apply(niveau)
        st.dataframe(
            disp[['timestamp', 'état', 'niveau', 'prob', 'type', 'mail']].reset_index(drop=True),
            use_container_width=True,
            column_config={
                'timestamp': st.column_config.DatetimeColumn("Horodatage", format="DD/MM HH:mm:ss"),
                'état':      st.column_config.TextColumn("État"),
                'niveau':    st.column_config.TextColumn("Niveau"),
                'prob':      st.column_config.TextColumn("Probabilité"),
                'type':      st.column_config.TextColumn("Type prédit"),
                'mail':      st.column_config.TextColumn("Mail"),
            },
        )
    with st.expander("⚙️ Configuration des alertes mail", expanded=False):
        st.markdown("Renseignez vos paramètres SMTP dans `page_prediction_four.py` → `MAIL_CONFIG`.")
        st.code("""MAIL_CONFIG = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port":   587,
    "sender":      "votre.email@gmail.com",
    "password":    "votre_app_password",   # mot de passe d'application Gmail
    "recipient":   "responsable@cjo.com.tn",
}""", language="python")
        st.info("Pour Gmail : activez l'authentification à 2 facteurs → "
                "Sécurité → Mots de passe des applications → générer un mot de passe.")

