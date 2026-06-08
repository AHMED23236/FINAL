import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
MAIL_CONFIG = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port":   587,
    "sender":      "cjoadmin@gmail.com",
    "password":    "cxhlmuudblbtxozj",
    "recipient":   "ahmed.naffeti10@gmail.com",
}
FAILURE_LABELS = {
    "Thermal_Anomaly":  "🌡️ Panne Thermique",
    "Mechanical_Stop":  "⚙️ Arrêt Mécanique",
    "No_Failure":       "✅ Aucune panne",
}
_MAIL_STYLE = {
    "Thermal_Anomaly": {
        "color": "#ef4444", "emoji": "🌡️",
        "intro": "Une <strong>panne thermique</strong> a été prédite sur le séchoir céramique. Le système de chauffage et les capteurs de température doivent être vérifiés immédiatement.",
        "actions": [
            "Vérifier les résistances chauffantes et thermocouples",
            "Contrôler la régulation de température (zones H200/H202/H204)",
            "Inspecter les câblages et bornes de connexion",
            "Vérifier les alarmes thermiques actives dans le SCADA",
        ],
        "composants": "Résistances · Thermocouples · Régulateurs · Câblage SCADA",
        "priorite": "🔴 URGENTE — Intervention dans les 2 heures",
        "delai": "≤ 2 h",
    },
    "Mechanical_Stop": {
        "color": "#3b82f6", "emoji": "⚙️",
        "intro": "Un <strong>arrêt mécanique</strong> a été prédit sur le séchoir céramique. Le système de convoyage et les organes mécaniques doivent être inspectés.",
        "actions": [
            "Inspecter le convoyeur et les rouleaux de transport",
            "Vérifier les moteurs d'entraînement et les réducteurs",
            "Contrôler les capteurs de position et fins de course",
            "Inspecter les courroies, chaînes et accouplements",
        ],
        "composants": "Convoyeur · Moteurs · Réducteurs · Capteurs position",
        "priorite": "🟠 HAUTE — Intervention dans les 4 heures",
        "delai": "≤ 4 h",
    },
}
def _build_fiche_sechoir(failure_type: str) -> str:
    s = _MAIL_STYLE.get(failure_type, _MAIL_STYLE["Thermal_Anomaly"])
    actions_html = "".join(
        f'<li style="margin:6px 0;color:#f1f5f9;">{a}</li>'
        for a in s["actions"]
    )
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
def send_test_mail(failure_type: str, prob: float = 0.95):
    ft_label   = FAILURE_LABELS.get(failure_type, failure_type)
    style      = _MAIL_STYLE.get(failure_type, _MAIL_STYLE["Thermal_Anomaly"])
    color      = style["color"]
    emoji      = style["emoji"]
    intro      = style["intro"]
    fiche_html = _build_fiche_sechoir(failure_type)
    ft_clean   = ft_label.replace("🌡️","").replace("⚙️","").replace("✅","").strip()
    window_start = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[TEST CJO] Séchoir — {ft_clean} prédite ({prob*100:.0f}%)"
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
        CJO Poulina — Système de maintenance prédictive · Message automatique [TEST]</p>
    </div></body></html>
    """
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP(MAIL_CONFIG["smtp_server"], MAIL_CONFIG["smtp_port"]) as server:
        server.starttls()
        server.login(MAIL_CONFIG["sender"], MAIL_CONFIG["password"])
        server.sendmail(MAIL_CONFIG["sender"], MAIL_CONFIG["recipient"], msg.as_string())
    print(f"✅ Mail envoyé — type: {failure_type} ({ft_label})")
if __name__ == "__main__":
    print("Envoi mail TEST Séchoir...")
    send_test_mail("Thermal_Anomaly", prob=0.944)
    send_test_mail("Mechanical_Stop", prob=0.761)
    print("Done.")

