import pandas as pd
import joblib
import smtplib
import json
import sys
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from pathlib import Path
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
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "cjo-maintenance-data-cleaned"
MODEL_FILE    = BASE_DIR / "xgboost_sechoir_binary.pkl"
FEATURES_FILE = BASE_DIR / "feature_cols_sechoir.pkl"
RULES_FILE    = BASE_DIR / "rule_mapping_sechoir.pkl"
DATA_FILE     = DATA_DIR / "Alarme_sechoir_ML.csv"
LOG_FILE      = DATA_DIR / "alerter_log.json"
MAIL_CONFIG = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port":   587,
    "sender":      "cjoadmin@gmail.com",
    "password":    "cxhlmuudblbtxozj",
    "recipient":   "ahmed.naffeti10@gmail.com",
}
SEUIL_ALERTE = 0.50
def log(message: str, level: str = "INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {message}")
def save_log(entry: dict):
    """Ajoute une entrée au journal JSON."""
    logs = []
    if LOG_FILE.exists():
        try:
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        except Exception:
            logs = []
    logs.append(entry)
    logs = logs[-200:]
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=2, default=str)
def load_assets():
    try:
        model    = joblib.load(MODEL_FILE)
        features = joblib.load(FEATURES_FILE)
    except Exception as e:
        log(f"Erreur chargement modèle/features : {e}", "ERROR")
        sys.exit(1)
    try:
        rules = joblib.load(RULES_FILE)
        log("Modèle + règles chargés avec succès.")
    except Exception as e:
        log(f"rule_mapping_sechoir.pkl non chargé ({e}) — utilisation des règles intégrées.", "WARN")
        rules = predict_failure_type
    return model, features, rules
def load_data():
    try:
        df = pd.read_csv(DATA_FILE)
        df['window_start'] = pd.to_datetime(df['window_start'])
        log(f"Données chargées : {len(df)} fenêtres | dernière : {df['window_start'].iloc[-1]}")
        return df
    except FileNotFoundError:
        log(f"Fichier introuvable : {DATA_FILE.name} — lancez generate_sechoir_daily.py", "ERROR")
        sys.exit(1)
def apply_rule_mapping(row: pd.Series, rules) -> str:
    if rules is None:
        if row.get('past_thermal_6h', 0) > row.get('past_mechanical_6h', 0):
            return "Thermal_Anomaly"
        return "Mechanical_Stop"
    if isinstance(rules, dict):
        for panne_type, conditions in rules.items():
            if all(row.get(feat, 0) >= thresh for feat, thresh in conditions.items()):
                return panne_type
        return "Mechanical_Stop"
    if callable(rules):
        try:
            return rules(row)
        except Exception:
            pass
    return "Inconnu"
def predict(model, features, rules, df):
    now        = pd.Timestamp.now().floor('5min')
    time_delta = (df['window_start'] - now).abs()
    idx        = time_delta.idxmin()
    last_row     = df.loc[idx]
    window_start = str(last_row['window_start'])
    available = [f for f in features if f in df.columns]
    missing   = [f for f in features if f not in df.columns]
    if missing:
        log(f"{len(missing)} features manquantes, complétées à 0 : {missing[:5]}...", "WARN")
    X = pd.DataFrame([last_row[available].values], columns=available)
    for col in missing:
        X[col] = 0.0
    X = X[features]
    pred  = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])
    failure_type = "No_Failure"
    if pred == 1:
        failure_type = apply_rule_mapping(last_row, rules)
    log(f"Prédiction : {'ANORMAL' if pred==1 else 'NORMAL'} | "
        f"Prob : {proba*100:.1f}% | Type : {failure_type} | Fenêtre : {window_start}")
    return pred, proba, failure_type, window_start
def send_mail(proba: float, failure_type: str, window_start: str) -> bool:
    ft_color = {
        "Thermal_Anomaly":  "#f97316",
        "Mechanical_Stop":  "#3b82f6",
    }.get(failure_type, "#ef4444")
    html = f"""
    <html><body style="font-family:Arial,sans-serif;background:#0f172a;
                       color:#f1f5f9;padding:20px;margin:0;">
    <div style="max-width:580px;margin:auto;background:#1e293b;
                border-radius:12px;padding:28px;border:2px solid #ef4444;">
        <h2 style="color:#ef4444;margin-top:0;font-size:1.3rem;">
            🔴 ALERTE MAINTENANCE — Séchoir Céramique CJO
        </h2>
        <p style="color:#94a3b8;margin-bottom:20px;">
            Le système de maintenance prédictive a détecté une anomalie imminente.
            Une intervention est recommandée dans les <strong style="color:#f1f5f9;">2 prochaines heures</strong>.
        </p>
        <table style="width:100%;border-collapse:collapse;border-radius:8px;overflow:hidden;">
            <tr style="background:#0f172a;">
                <td style="padding:12px 16px;color:#94a3b8;width:45%;">Fenêtre analysée</td>
                <td style="padding:12px 16px;color:#f1f5f9;font-weight:600;">{window_start[:16]}</td>
            </tr>
            <tr>
                <td style="padding:12px 16px;color:#94a3b8;">Probabilité d'anomalie</td>
                <td style="padding:12px 16px;color:#ef4444;font-weight:700;font-size:1.1rem;">{proba*100:.1f}%</td>
            </tr>
            <tr style="background:#0f172a;">
                <td style="padding:12px 16px;color:#94a3b8;">Type de panne prédit</td>
                <td style="padding:12px 16px;color:{ft_color};font-weight:600;">{failure_type.replace("_"," ")}</td>
            </tr>
            <tr>
                <td style="padding:12px 16px;color:#94a3b8;">Heure d'alerte</td>
                <td style="padding:12px 16px;color:#f1f5f9;">{datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}</td>
            </tr>
        </table>
        <div style="margin-top:20px;padding:14px;background:#0f172a;
                    border-radius:8px;border-left:3px solid #f97316;">
            <p style="margin:0;color:#94a3b8;font-size:0.85rem;">
                ⚠️ Vérifiez les capteurs thermiques (H196/H198/H200/H204)
                et les pressostats EAU11/EAU12/EAU21/EAU22.
            </p>
        </div>
        <p style="color:#334155;font-size:11px;margin-top:20px;margin-bottom:0;text-align:center;">
            CJO Poulina · Système de maintenance prédictive · Message automatique<br>
            Ne pas répondre à cet email.
        </p>
    </div>
    </body></html>
    """
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = (f"[CJO ALERTE] Séchoir — Anomalie HIGH "
                          f"{proba*100:.0f}% · {failure_type.replace('_',' ')}")
        msg["From"]    = MAIL_CONFIG["sender"]
        msg["To"]      = MAIL_CONFIG["recipient"]
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP(MAIL_CONFIG["smtp_server"], MAIL_CONFIG["smtp_port"]) as server:
            server.starttls()
            server.login(MAIL_CONFIG["sender"], MAIL_CONFIG["password"])
            server.sendmail(
                MAIL_CONFIG["sender"],
                MAIL_CONFIG["recipient"],
                msg.as_string(),
            )
        log(f"Mail envoyé à {MAIL_CONFIG['recipient']}")
        return True
    except smtplib.SMTPAuthenticationError:
        log("Échec authentification SMTP — vérifiez sender/password dans MAIL_CONFIG", "ERROR")
        return False
    except smtplib.SMTPException as e:
        log(f"Erreur SMTP : {e}", "ERROR")
        return False
    except Exception as e:
        log(f"Erreur envoi mail : {e}", "ERROR")
        return False
def main():
    log("=" * 50)
    log("Alerter Séchoir CJO — démarrage")
    log("=" * 50)
    model, features, rules = load_assets()
    df = load_data()
    pred, proba, failure_type, window_start = predict(model, features, rules, df)
    mail_sent = False
    if pred == 1 and proba >= SEUIL_ALERTE:
        log(f"ANORMAL détecté ({proba*100:.1f}%) — envoi mail...", "WARN")
        mail_sent = send_mail(proba, failure_type, window_start)
    else:
        log("État NORMAL — aucune alerte envoyée.")
    entry = {
        "timestamp":    datetime.now().isoformat(),
        "prediction":   pred,
        "probability":  round(proba, 4),
        "failure_type": failure_type,
        "window_start": window_start,
        "mail_sent":    mail_sent,
    }
    save_log(entry)
    log(f"Résultat journalisé dans {LOG_FILE.name}")
    log("=" * 50)
if __name__ == "__main__":
    main()

