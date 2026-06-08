import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from pathlib import Path
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "cjo-maintenance-data-cleaned"
MODEL_FILE        = BASE_DIR / "xgboost_sechoir_binary.pkl"
FEATURES_FILE     = BASE_DIR / "feature_cols_sechoir.pkl"
RULES_FILE        = BASE_DIR / "rule_mapping_sechoir.pkl"
FT_MODEL_FILE     = BASE_DIR / "xgboost_failure_type_sechoir.pkl"
LE_FT_FILE        = BASE_DIR / "le_failure_type_sechoir.pkl"
DATA_FILE         = DATA_DIR / "Alarme_sechoir_ML.csv"
HIST_FILE         = DATA_DIR / "prediction_history_sechoir.csv"
FAILURE_LABELS = {
    "Thermal_Anomaly":  "🌡️ Panne Thermique",
    "Mechanical_Stop":  "⚙️ Arrêt Mécanique",
    "No_Failure":       "✅ Aucune panne",
}
FT_FEATURES = [
    'hour', 'day_of_week', 'month', 'is_weekend', 'shift_enc',
    'is_night_shift', 'is_morning_shift',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'time_since_last_any', 'time_since_last_thermal',
    'time_since_last_mechanical', 'mean_duration_last_20',
    'max_duration_last_20', 'std_duration_last_20', 'alarm_acceleration',
]
MAIL_CONFIG = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port":   587,
    "sender":      "cjoadmin@gmail.com",
    "password":    "cxhlmuudblbtxozj",
    "recipient":   "ahmed.naffeti10@gmail.com",
}
def predict_failure_type(row):
    """Rule mapping : déduit le type de panne à partir des features."""
    try:
        thermal = row.get('past_thermal_6h',   0) if hasattr(row, 'get') else float(row['past_thermal_6h'])
        mech    = row.get('past_mechanical_6h', 0) if hasattr(row, 'get') else float(row['past_mechanical_6h'])
        eau     = row.get('EAU_max_all',        0) if hasattr(row, 'get') else float(row['EAU_max_all'])
    except Exception:
        thermal, mech, eau = 0, 0, 0
    if thermal > mech or eau > 1.5:
        return "Thermal_Anomaly"
    return "Mechanical_Stop"
REFRESH_INTERVAL = 30
DARK  = '#0f172a'
CARD  = '#1e293b'
BORD  = '#334155'
TEXT  = '#94a3b8'
WHITE = '#f1f5f9'
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
def kpi(label, value, sub="", color=WHITE):
    sub_html = f"<div class='kpi-sub'>{sub}</div>" if sub else ""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{color}">{value}</div>
        {sub_html}
    </div>""", unsafe_allow_html=True)
@st.cache_resource(show_spinner=False)
def load_model_assets():
    try:
        model    = joblib.load(MODEL_FILE)
        features = joblib.load(FEATURES_FILE)
    except Exception:
        return None, None, None, None
    try:
        ft_model = joblib.load(FT_MODEL_FILE)
        le_ft    = joblib.load(LE_FT_FILE)
    except Exception:
        ft_model, le_ft = None, None
    return model, features, ft_model, le_ft
@st.cache_data(ttl=60, show_spinner=False)
def load_live_data():
    df = pd.read_csv(DATA_FILE)
    df['window_start'] = pd.to_datetime(df['window_start'])
    return df
@st.cache_data(ttl=60, show_spinner=False)
def load_history():
    if HIST_FILE.exists():
        return pd.read_csv(HIST_FILE, parse_dates=['timestamp'])
    return pd.DataFrame(columns=[
        'timestamp', 'prediction', 'probability',
        'failure_type_pred', 'window_start', 'mail_sent',
    ])
def save_history(hist_df):
    hist_df.tail(500).to_csv(HIST_FILE, index=False)
def apply_rule_mapping(row: pd.Series, rules) -> str:
    """Applique le rule_mapping pour déterminer le type de panne prédit."""
    if rules is None:
        if row.get('past_thermal_6h', 0) > row.get('past_mechanical_6h', 0):
            return "Thermal_Anomaly"
        return "Mechanical_Stop"
    if isinstance(rules, dict):
        for panne_type, conditions in rules.items():
            match = all(
                row.get(feat, 0) >= thresh
                for feat, thresh in conditions.items()
            )
            if match:
                return panne_type
        return "Mechanical_Stop"
    if callable(rules):
        try:
            return rules(row)
        except Exception:
            pass
    return "Inconnu"
_MAIL_STYLE = {
    "Thermal_Anomaly": {
        "color": "#ef4444", "emoji": "🌡️",
        "intro": "Une <strong>panne thermique</strong> a été prédite sur le séchoir céramique. Le système de chauffage et les capteurs de température doivent être vérifiés immédiatement.",
        "actions": ["Vérifier les résistances chauffantes et thermocouples", "Contrôler la régulation de température (zones H200/H202/H204)", "Inspecter les câblages et bornes de connexion", "Vérifier les alarmes thermiques actives dans le SCADA"],
        "composants": "Résistances · Thermocouples · Régulateurs · Câblage SCADA",
        "priorite": "🔴 URGENTE — Intervention dans les 2 heures", "delai": "≤ 2 h",
    },
    "Mechanical_Stop": {
        "color": "#3b82f6", "emoji": "⚙️",
        "intro": "Un <strong>arrêt mécanique</strong> a été prédit sur le séchoir céramique. Le système de convoyage et les organes mécaniques doivent être inspectés.",
        "actions": ["Inspecter le convoyeur et les rouleaux de transport", "Vérifier les moteurs d'entraînement et les réducteurs", "Contrôler les capteurs de position et fins de course", "Inspecter les courroies, chaînes et accouplements"],
        "composants": "Convoyeur · Moteurs · Réducteurs · Capteurs position",
        "priorite": "🟠 HAUTE — Intervention dans les 4 heures", "delai": "≤ 4 h",
    },
    "No_Failure": {
        "color": "#22c55e", "emoji": "✅",
        "intro": "Aucune panne détectée — surveillance préventive.",
        "actions": ["Surveillance continue recommandée."],
        "composants": "—", "priorite": "🟢 NORMALE", "delai": "—",
    },
}
def _build_fiche_sechoir(failure_type: str) -> str:
    s = _MAIL_STYLE.get(failure_type, _MAIL_STYLE["Thermal_Anomaly"])
    actions_html = "".join(f'<li style="margin:6px 0;color:#f1f5f9;">{a}</li>' for a in s["actions"])
    return f"""
    <div style="margin-top:20px;background:#0f172a;border-radius:10px;
                padding:18px 20px;border-left:4px solid #f97316;">
        <h3 style="color:#f97316;margin:0 0 10px 0;font-size:1rem;">📋 Fiche de Maintenance Recommandée</h3>
        <table style="width:100%;border-collapse:collapse;margin-bottom:14px;">
            <tr>
                <td style="padding:8px;color:#64748b;font-size:0.82rem;width:38%;">Priorité</td>
                <td style="padding:8px;color:#f1f5f9;font-weight:bold;">{s['priorite']}</td>
            </tr>
            <tr style="background:#1e293b;">
                <td style="padding:8px;color:#64748b;font-size:0.82rem;">Délai max</td>
                <td style="padding:8px;color:#ef4444;font-weight:bold;">{s['delai']}</td>
            </tr>
            <tr>
                <td style="padding:8px;color:#64748b;font-size:0.82rem;">Composants</td>
                <td style="padding:8px;color:#94a3b8;font-size:0.82rem;">{s['composants']}</td>
            </tr>
        </table>
        <p style="color:#64748b;font-size:0.82rem;margin:0 0 8px 0;font-weight:600;
                  text-transform:uppercase;letter-spacing:0.05em;">Actions à effectuer :</p>
        <ol style="margin:0;padding-left:20px;">{actions_html}</ol>
    </div>
    """
def send_alert_mail(prob: float, failure_type: str, window_start: str) -> bool:
    try:
        ft_label   = FAILURE_LABELS.get(failure_type, failure_type)
        style      = _MAIL_STYLE.get(failure_type, _MAIL_STYLE["Thermal_Anomaly"])
        color      = style["color"]
        emoji      = style["emoji"]
        intro      = style["intro"]
        fiche_html = _build_fiche_sechoir(failure_type)
        ft_clean   = ft_label.replace("🌡️","").replace("⚙️","").replace("✅","").strip()
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[CJO ALERTE] Séchoir — {ft_clean} prédite ({prob*100:.0f}%)"
        msg["From"]    = MAIL_CONFIG["sender"]
        msg["To"]      = MAIL_CONFIG["recipient"]
        html = f"""
        <html><body style="font-family:Arial,sans-serif;background:#0f172a;color:#f1f5f9;padding:20px;">
        <div style="max-width:640px;margin:auto;background:#1e293b;border-radius:12px;
                    padding:24px;border:2px solid {color};">
            <h2 style="color:{color};margin-top:0;">{emoji} ALERTE {ft_clean.upper()} — Séchoir Céramique</h2>
            <p style="color:#94a3b8;">{intro}<br>
            Niveau de risque : <strong style="color:{color}">HIGH</strong> &nbsp;|&nbsp;
            Probabilité : <strong style="color:{color}">{prob*100:.1f}%</strong></p>
            <table style="width:100%;border-collapse:collapse;margin:16px 0;">
                <tr style="background:#0f172a;">
                    <td style="padding:10px;color:#94a3b8;">Fenêtre temporelle</td>
                    <td style="padding:10px;color:#f1f5f9;font-weight:bold;">{window_start}</td>
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
                    <td style="padding:10px;color:#f1f5f9;">2 heures</td>
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
def run_prediction(model, features, ft_model, le_ft, df_live):
    """Prédit sur la fenêtre la plus proche de l'heure actuelle."""
    now = pd.Timestamp.now().floor('5min')
    time_delta = (df_live['window_start'] - now).abs()
    idx        = time_delta.idxmin()
    last_row = df_live.loc[idx]
    window_start = str(last_row['window_start'])
    available = [f for f in features if f in df_live.columns]
    missing   = [f for f in features if f not in df_live.columns]
    X = pd.DataFrame([last_row[available].values], columns=available)
    for col in missing:
        X[col] = 0.0
    X = X[features]
    pred  = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])
    if ft_model is not None and le_ft is not None:
        ft_avail   = [f for f in FT_FEATURES if f in df_live.columns]
        X_ft = pd.DataFrame([last_row[ft_avail].values], columns=ft_avail)
        for col in FT_FEATURES:
            if col not in X_ft.columns:
                X_ft[col] = 0.0
        X_ft = X_ft[FT_FEATURES]
        ft_class = int(ft_model.predict(X_ft)[0])
        failure_type = le_ft.inverse_transform([ft_class])[0]
        if pred == 0:
            failure_type = "No_Failure"
    else:
        failure_type = "No_Failure"
        if pred == 1:
            failure_type = apply_rule_mapping(last_row, None)
    return pred, proba, failure_type, window_start
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
def show_prediction_sechoir():
    _css()
    st.markdown("## 🤖 Prédiction Temps Réel — Séchoir Céramique")
    st.markdown(
        f"<div style='color:{TEXT};font-size:0.85rem;margin-bottom:1.5rem;'>"
        "XGBoost Binaire &nbsp;|&nbsp; Fenêtres 5 min &nbsp;|&nbsp;"
        f" Actualisation auto toutes les {REFRESH_INTERVAL} min</div>",
        unsafe_allow_html=True,
    )
    model, features, ft_model, le_ft = load_model_assets()
    if model is None:
        st.error("Modèle non trouvé. Placez `xgboost_sechoir_binary.pkl` et "
                 "`feature_cols_sechoir.pkl` dans `src/`.")
        return
    if 'pred_history'        not in st.session_state:
        st.session_state.pred_history        = load_history()
    if 'last_pred_time'      not in st.session_state:
        st.session_state.last_pred_time      = 0.0
    if 'last_result'         not in st.session_state:
        st.session_state.last_result         = None
    if 'mail_enabled'        not in st.session_state:
        st.session_state.mail_enabled        = True
    if 'last_window_start' not in st.session_state:
        hist = st.session_state.pred_history
        if not hist.empty:
            st.session_state.last_window_start = str(hist.iloc[0]['window_start'])
        else:
            st.session_state.last_window_start = None
    col_btn, col_mail, col_next = st.columns([2, 2, 3])
    with col_btn:
        force_pred = st.button("🔄 Lancer la prédiction", use_container_width=True, type="primary")
    with col_mail:
        st.session_state.mail_enabled = st.toggle(
            "📧 Alertes mail", value=st.session_state.mail_enabled)
    with col_next:
        elapsed_now = time.time() - st.session_state.last_pred_time
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
    elapsed      = time.time() - st.session_state.last_pred_time
    auto_trigger = elapsed >= (REFRESH_INTERVAL * 60)
    first_load   = st.session_state.last_result is None
    should_predict = force_pred or (auto_trigger and not first_load) or first_load
    if should_predict:
        try:
            df_live = load_live_data()
            pred, proba, failure_type, window_start = run_prediction(
                model, features, ft_model, le_ft, df_live)
            is_new_window = window_start != st.session_state.last_window_start
            mail_sent     = False
            mail_attempted = False
            if pred == 1 and st.session_state.mail_enabled and is_new_window:
                mail_attempted = True
                mail_sent = send_alert_mail(proba, failure_type, window_start)
            result = {
                'prediction':        pred,
                'probability':       proba,
                'failure_type_pred': failure_type,
                'window_start':      window_start,
                'mail_sent':         mail_sent,
                'mail_attempted':    mail_attempted,
                'timestamp':         datetime.now(),
            }
            st.session_state.last_result    = result
            st.session_state.last_pred_time = time.time()
            if is_new_window:
                st.session_state.last_window_start = window_start
                new_row = pd.DataFrame([{
                    'timestamp':         result['timestamp'],
                    'prediction':        pred,
                    'probability':       proba,
                    'failure_type_pred': failure_type,
                    'window_start':      window_start,
                    'mail_sent':         mail_sent,
                }])
                st.session_state.pred_history = pd.concat(
                    [st.session_state.pred_history, new_row], ignore_index=True)
                save_history(st.session_state.pred_history)
        except FileNotFoundError:
            st.error(f"Fichier de données introuvable : `{DATA_FILE.name}`. "
                     "Lancez d'abord `generate_sechoir_daily.py`.")
            return
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
            return
    res = st.session_state.last_result
    if res is None:
        return
    pred     = res['prediction']
    proba    = res['probability']
    ft       = res['failure_type_pred']
    ts       = res['timestamp'].strftime('%H:%M:%S') if hasattr(res['timestamp'], 'strftime') else str(res['timestamp'])
    if pred == 0:
        st.markdown(f"""
        <div class="status-box status-normal">
            <div class="status-icon">🟢</div>
            <div class="status-label" style="color:#22c55e;">NORMAL</div>
            <div class="status-sub">Aucune anomalie prévue dans les 2 prochaines heures</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="status-box status-anormal">
            <div class="status-icon">🔴</div>
            <div class="status-label" style="color:#ef4444;">ANORMAL — HIGH</div>
            <div class="status-sub">Anomalie probable dans les 2 prochaines heures · Type : {ft}</div>
        </div>""", unsafe_allow_html=True)
        if res.get('mail_sent'):
            st.success("📧 Alerte mail envoyée au responsable maintenance.")
        elif res.get('mail_attempted'):
            st.warning("📧 Mail non envoyé (vérifiez la configuration SMTP).")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Probabilité anomalie", f"{proba*100:.1f}%",
            "HIGH dans 2h", "#EF4444" if proba >= 0.5 else "#22C55E")
    with c2:
        kpi("Type de panne prédit", FAILURE_LABELS.get(ft, ft.replace("_", " ")),
            "", "#F97316" if pred == 1 else TEXT)
    with c3:
        hist = st.session_state.pred_history
        n_anom = int((hist['prediction'] == 1).sum()) if len(hist) > 0 else 0
        kpi("Anomalies détectées", str(n_anom),
            f"sur {len(hist)} prédictions", "#EF4444" if n_anom > 0 else "#22C55E")
    with c4:
        kpi("Dernière mise à jour", ts, res['window_start'][:16] if res['window_start'] else "")
    st.markdown("---")
    with st.expander("🔎 Détail des features utilisées", expanded=False):
        try:
            df_live = load_live_data()
            last_row = df_live.iloc[-1]
            available = [f for f in features if f in df_live.columns]
            feat_vals = last_row[available].to_frame(name='Valeur').reset_index()
            feat_vals.columns = ['Feature', 'Valeur']
            col_l, col_r = st.columns(2)
            mid = len(feat_vals) // 2
            with col_l:
                st.dataframe(feat_vals.iloc[:mid], use_container_width=True, hide_index=True)
            with col_r:
                st.dataframe(feat_vals.iloc[mid:], use_container_width=True, hide_index=True)
        except Exception:
            st.info("Données live non disponibles.")
    st.markdown("### 📈 Historique des prédictions")
    hist_df = st.session_state.pred_history.copy()
    plot_history(hist_df)
    if not hist_df.empty:
        st.markdown("### 📋 Journal des prédictions")
        disp = hist_df.copy().sort_values('timestamp', ascending=False).head(50)
        disp['état'] = disp['prediction'].map({0: '🟢 Normal', 1: '🔴 Anormal'})
        disp['prob'] = (disp['probability'] * 100).round(1).astype(str) + '%'
        disp['mail'] = disp['mail_sent'].map({True: '✅', False: '—'})
        disp['failure_type_label'] = disp['failure_type_pred'].map(
            lambda x: FAILURE_LABELS.get(x, x.replace("_", " ") if isinstance(x, str) else x)
        )
        def niveau(p):
            if p >= 0.80: return '🔴 HIGH'
            if p >= 0.50: return '🟠 MEDIUM'
            return '🟢 LOW'
        disp['niveau'] = disp['probability'].apply(niveau)
        st.dataframe(
            disp[['timestamp', 'état', 'niveau', 'prob', 'failure_type_label',
                  'window_start', 'mail']].reset_index(drop=True),
            use_container_width=True,
            column_config={
                'timestamp':           st.column_config.DatetimeColumn("Horodatage", format="DD/MM HH:mm:ss"),
                'état':                st.column_config.TextColumn("État"),
                'niveau':              st.column_config.TextColumn("Niveau"),
                'prob':                st.column_config.TextColumn("Probabilité"),
                'failure_type_label':  st.column_config.TextColumn("Type prédit"),
                'window_start':        st.column_config.TextColumn("Fenêtre"),
                'mail':                st.column_config.TextColumn("Mail"),
            },
        )
    with st.expander("⚙️ Configuration des alertes mail", expanded=False):
        st.markdown("Renseignez vos paramètres SMTP directement dans `page_prediction_sechoir.py` → `MAIL_CONFIG`.")
        st.code("""MAIL_CONFIG = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port":   587,
    "sender":      "votre.email@gmail.com",
    "password":    "votre_app_password",   # mot de passe d'application Gmail
    "recipient":   "responsable@cjo.com.tn",
}""", language="python")
        st.info("Pour Gmail : activez l'authentification à 2 facteurs → "
                "Sécurité → Mots de passe des applications → générer un mot de passe.")
    elapsed_now = time.time() - st.session_state.last_pred_time
    remaining = int(max(0, REFRESH_INTERVAL * 60 - elapsed_now))
    if remaining > 0:
        components.html(f"""
        <p style='color:#64748b;font-size:0.8rem;font-family:sans-serif;margin:0;'>
            Prochaine actualisation automatique dans
            <strong id='cd2' style='color:#94a3b8;'>{remaining//60:02d}:{remaining%60:02d}</strong>
            min — ou cliquez sur 🔄 pour forcer.
        </p>
        <script>
        var el = document.getElementById('cd2');
        var secs = {remaining};
        setInterval(function() {{
            if (secs > 0) secs--;
            var m = Math.floor(secs / 60);
            var s = secs % 60;
            el.textContent = (m<10?'0':'')+m+':'+(s<10?'0':'')+s;
        }}, 1000);
        </script>
        """, height=30)

