"""
================================================================================
RULE ENGINE — Maintenance Recommendations Based on Real Fiche Database
================================================================================
This module looks up the right maintenance fiche for an alarm using a
three-level matching strategy:
  Level 1 — DIRECT MATCH:    alarm description == fiche description
  Level 2 — PATTERN MATCH:   "Ecart min/max de Tempér." family fiches
  Level 3 — TYPE FALLBACK:   generic fiche per failure_type
Coverage on the training data: ~100% of all alarm events.
================================================================================
"""
import json
import os
import re
_HERE = os.path.dirname(os.path.abspath(__file__))
_FICHES_PATH = os.path.join(_HERE, "rules", "fiches_database.json")
with open(_FICHES_PATH, "r", encoding="utf-8") as f:
    FICHES = json.load(f)
PATTERN_FICHES = {
    "ecart_max_temp": {
        "description": "Écart maximum de température (générique)",
        "type":        "Temperature Failure",
        "urgence":     "Low",
        "category":    "Non-Bloquante",
        "symptomes":   ["Température de la zone supérieure à la consigne max",
                        "Écart positif détecté",
                        "Alarme de surveillance déclenchée"],
        "actions":     ["Vérifier le thermocouple de la zone concernée",
                        "Contrôler le régulateur PID associé",
                        "Réduire légèrement la consigne si nécessaire",
                        "Surveiller l'évolution de l'écart"],
        "preventives": ["Étalonnage trimestriel des thermocouples",
                        "Vérification mensuelle des paramètres PID"],
    },
    "ecart_min_temp": {
        "description": "Écart minimum de température (générique)",
        "type":        "Temperature Failure",
        "urgence":     "Low",
        "category":    "Non-Bloquante",
        "symptomes":   ["Température de la zone inférieure à la consigne min",
                        "Écart négatif détecté",
                        "Alarme de surveillance déclenchée"],
        "actions":     ["Vérifier le thermocouple de la zone concernée",
                        "Contrôler les brûleurs de la zone",
                        "Augmenter légèrement la consigne",
                        "Surveiller l'évolution de l'écart"],
        "preventives": ["Étalonnage trimestriel des thermocouples",
                        "Vérification mensuelle des brûleurs"],
    },
}
GENERIC_FICHES = {
    "temperature_failure": {
        "description": "Anomalie de température (générique)",
        "type":        "Temperature Failure",
        "urgence":     "Medium",
        "category":    "Bloquante",
        "symptomes":   ["Anomalie de température détectée"],
        "actions":     ["Vérifier les thermocouples de la zone concernée",
                        "Contrôler le système de régulation",
                        "Inspecter les brûleurs"],
        "preventives": ["Étalonnage périodique des capteurs"],
    },
    "burner_failure": {
        "description": "Anomalie système brûleurs (générique)",
        "type":        "Burner Failure",
        "urgence":     "High",
        "category":    "Bloquante",
        "symptomes":   ["Anomalie sur le système des brûleurs"],
        "actions":     ["Vérifier l'alimentation des brûleurs",
                        "Contrôler la pression du gaz",
                        "Inspecter les électrodes d'allumage"],
        "preventives": ["Inspection mensuelle des brûleurs",
                        "Contrôle de la pression gaz"],
    },
    "pression_failure": {
        "description": "Anomalie système pression (générique)",
        "type":        "Pression Failure",
        "urgence":     "High",
        "category":    "Bloquante",
        "symptomes":   ["Anomalie sur le système de pression"],
        "actions":     ["Vérifier les ventilateurs",
                        "Contrôler les pressostats",
                        "Inspecter les filtres"],
        "preventives": ["Nettoyage mensuel des filtres",
                        "Étalonnage des pressostats"],
    },
    "alimentation_failure": {
        "description": "Anomalie système alimentation (générique)",
        "type":        "Alimentation Failure",
        "urgence":     "High",
        "category":    "Bloquante",
        "symptomes":   ["Anomalie sur le système d'alimentation"],
        "actions":     ["Vérifier les disjoncteurs et fusibles",
                        "Contrôler l'alimentation 24 VDC",
                        "Inspecter les câbles électriques"],
        "preventives": ["Inspection mensuelle de l'armoire électrique",
                        "Test trimestriel des protections"],
    },
}
def lookup_fiche(description, failure_type):
    """
    Find the best fiche for a given alarm.
    Returns the fiche dict and the matching level used.
    """
    desc = str(description).strip()
    if desc in FICHES:
        return FICHES[desc], "direct"
    if re.match(r"Ecart\s+max", desc, re.IGNORECASE):
        return PATTERN_FICHES["ecart_max_temp"], "pattern_max_temp"
    if re.match(r"Ecart\s+min", desc, re.IGNORECASE):
        return PATTERN_FICHES["ecart_min_temp"], "pattern_min_temp"
    ftype = str(failure_type).strip().lower()
    if ftype in GENERIC_FICHES:
        return GENERIC_FICHES[ftype], "generic_by_type"
    return GENERIC_FICHES["temperature_failure"], "default"
def get_recommendation(features_row, model_prediction, model_probability):
    """
    Build a complete recommendation for an operator.
    Args:
        features_row       : a row from the features DataFrame (Series or dict)
        model_prediction   : 0 or 1 (model.predict output)
        model_probability  : probability of class 1 (model.predict_proba output)
    Returns: dict with alert_level, fiche, etc.
    """
    if model_prediction == 0:
        return {
            "alert_level": "GREEN",
            "icon":        "✅",
            "probability": float(model_probability),
            "headline":    "Operations look normal for the next 6 hours.",
            "fiche":       None,
            "match_level": None,
        }
    last_desc = str(features_row.get("latest_alarm_description", "")).strip()
    if last_desc == "" or last_desc == "nan":
        return {
            "alert_level": "YELLOW",
            "icon":        "⚠️",
            "probability": float(model_probability),
            "headline":    "Risk predicted but no recent alarm to localize. Stay vigilant.",
            "fiche":       None,
            "match_level": None,
        }
    last_ftype = features_row.get("latest_alarm_failure_type", "")
    fiche, match_level = lookup_fiche(last_desc, last_ftype)
    p = float(model_probability)
    if   p >= 0.85: alert, icon = "RED",    "🚨"
    elif p >= 0.65: alert, icon = "ORANGE", "🟠"
    else:           alert, icon = "YELLOW", "⚠️"
    return {
        "alert_level":   alert,
        "icon":          icon,
        "probability":   p,
        "headline":      f"High alarm expected within 6h. Latest issue: {last_desc}",
        "fiche":         fiche,
        "match_level":   match_level,
        "latest_alarm":  {
            "description":  last_desc,
            "codice":       str(features_row.get("latest_alarm_codice", "")).strip(),
            "failure_type": last_ftype,
            "urgency":      features_row.get("latest_alarm_urgency", ""),
        },
    }
def print_recommendation(rec):
    print("=" * 70)
    print(f"  {rec['icon']}  ALERT LEVEL: {rec['alert_level']}   "
          f"(model confidence: {rec['probability']*100:.1f}%)")
    print("=" * 70)
    print(f"  {rec['headline']}")
    if rec["fiche"]:
        f = rec["fiche"]
        print()
        print(f"  📋 FICHE: {f['description']}")
        print(f"     Type    : {f['type']}")
        print(f"     Urgence : {f['urgence']}")
        print(f"     Match   : {rec['match_level']}")
        print()
        print("  🔍 Symptômes détectés:")
        for s in f["symptomes"]:
            print(f"     • {s}")
        print()
        print("  🔧 Actions immédiates:")
        for a in f["actions"]:
            print(f"     • {a}")
        print()
        print("  🛡️  Actions préventives:")
        for p in f["preventives"]:
            print(f"     • {p}")
    print("=" * 70)
    print()
if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("/mnt/user-data/outputs/Allarmi_FEATURES.csv")
    print("\n" + "█" * 70)
    print("  DEMO: 4 sample recommendations from real data")
    print("█" * 70 + "\n")
    samples = df.iloc[[100, 600, 1100, 1400]]
    for idx, row in samples.iterrows():
        pred  = int(row["y"])
        proba = 0.92 if pred == 1 else 0.08
        rec = get_recommendation(row, pred, proba)
        print(f"--- Row {idx} | {row['timestamp']} ---")
        print_recommendation(rec)

