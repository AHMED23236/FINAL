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
    "temperature_failure":  "🌡️ Panne Thermique",
    "burner_failure":       "🔥 Panne Brûleurs",
    "pression_failure":     "💨 Panne Pression",
    "alimentation_failure": "⚡ Panne Alimentation",
    "No_Failure":           "✅ Aucune panne",
}
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
}
_MAIL_STYLE = {
    "temperature_failure":  {"color": "#ef4444", "emoji": "🌡️", "intro": "Une <strong>panne thermique</strong> a été prédite sur le four céramique. Le système de régulation de température nécessite une vérification immédiate."},
    "burner_failure":       {"color": "#f97316", "emoji": "🔥", "intro": "Une <strong>panne brûleurs</strong> a été prédite sur le four céramique. Le système de combustion doit être inspecté rapidement."},
    "pression_failure":     {"color": "#3b82f6", "emoji": "💨", "intro": "Une <strong>panne de pression</strong> a été prédite sur le four céramique. Le circuit de pression gaz/air doit être contrôlé."},
    "alimentation_failure": {"color": "#8b5cf6", "emoji": "⚡", "intro": "Une <strong>panne d'alimentation électrique</strong> a été prédite sur le four céramique. Le tableau électrique doit être vérifié en urgence."},
}
def _build_fiche_html(failure_type: str) -> str:
    fiche = MAINTENANCE_FICHES.get(failure_type, MAINTENANCE_FICHES["temperature_failure"])
    actions_html = "".join(
        f'<li style="margin:6px 0;color:#f1f5f9;">{a}</li>'
        for a in fiche["actions"]
    )
    return f"""
    <div style="margin-top:20px;background:#0f172a;border-radius:10px;
                padding:18px 20px;border-left:4px solid #f97316;">
        <h3 style="color:#f97316;margin:0 0 10px 0;font-size:1rem;">📋 Fiche de Maintenance Recommandée</h3>
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
        <p style="color:#64748b;font-size:0.82rem;margin:0 0 8px 0;font-weight:600;
                  text-transform:uppercase;letter-spacing:0.05em;">Actions à effectuer :</p>
        <ol style="margin:0;padding-left:20px;">{actions_html}</ol>
    </div>
    """
def send_test_mail(failure_type: str, prob: float):
    ft_label   = FAILURE_LABELS.get(failure_type, failure_type)
    style      = _MAIL_STYLE.get(failure_type, _MAIL_STYLE["temperature_failure"])
    color      = style["color"]
    emoji      = style["emoji"]
    intro      = style["intro"]
    fiche_html = _build_fiche_html(failure_type)
    ft_clean   = ft_label.replace("🌡️","").replace("🔥","").replace("💨","").replace("⚡","").replace("✅","").strip()
    timestamp  = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[TEST CJO] Four Céramique — {ft_clean} prédite ({prob*100:.0f}%)"
    msg["From"]    = MAIL_CONFIG["sender"]
    msg["To"]      = MAIL_CONFIG["recipient"]
    html = f"""
    <html><body style="font-family:Arial,sans-serif;background:#0f172a;color:#f1f5f9;padding:20px;">
    <div style="max-width:640px;margin:auto;background:#1e293b;border-radius:12px;
                padding:24px;border:2px solid {color};">
        <h2 style="color:{color};margin-top:0;">{emoji} ALERTE {ft_clean.upper()} — Four Céramique</h2>
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
        CJO Poulina — Système de maintenance prédictive · Message automatique [TEST]</p>
    </div></body></html>
    """
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP(MAIL_CONFIG["smtp_server"], MAIL_CONFIG["smtp_port"]) as server:
        server.starttls()
        server.login(MAIL_CONFIG["sender"], MAIL_CONFIG["password"])
        server.sendmail(MAIL_CONFIG["sender"], MAIL_CONFIG["recipient"], msg.as_string())
    print(f"OK Mail envoye — {failure_type} ({ft_label})")
if __name__ == "__main__":
    print("Envoi mails TEST Four Ceramique...")
    send_test_mail("temperature_failure",  prob=0.999)
    send_test_mail("burner_failure",       prob=0.872)
    send_test_mail("pression_failure",     prob=0.654)
    send_test_mail("alimentation_failure", prob=0.731)
    print("Done — 4 mails envoyes.")

