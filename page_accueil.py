import streamlit as st
def show_accueil():
    st.markdown("""
    <style>
    .accueil-title {
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 0.3rem;
        margin-top: 3rem;
    }
    .accueil-sub {
        text-align: center;
        font-size: 1rem;
        color: #64748b;
        margin-bottom: 3rem;
    }
    .machine-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 20px;
        padding: 2.5rem 2rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s ease;
        margin: 0 1rem;
    }
    .machine-card:hover {
        border-color: #7a9bbf;
        background: #263447;
    }
    .machine-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        display: block;
    }
    .machine-name {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 0.5rem;
    }
    .machine-desc {
        font-size: 0.85rem;
        color: #94a3b8;
        line-height: 1.6;
    }
    .machine-badge {
        display: inline-block;
        margin-top: 1rem;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-four    { background: #ef444422; color: #ef4444; border: 1px solid #ef444444; }
    .badge-sechoir { background: #3b82f622; color: #3b82f6; border: 1px solid #3b82f644; }
    .divider {
        text-align: center;
        color: #334155;
        font-size: 1.5rem;
        margin: auto;
        padding-top: 2rem;
    }
    .footer-accueil {
        text-align: center;
        color: #334155;
        font-size: 0.75rem;
        margin-top: 4rem;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="accueil-title">🏭 CJO Poulina</div>
    <div class="accueil-sub">Système de Maintenance Prédictive — Sélectionnez une machine</div>
    """, unsafe_allow_html=True)
    col_left, col_mid, col_right = st.columns([2, 0.3, 2])
    with col_left:
        st.markdown("""
        <div class="machine-card">
            <span class="machine-icon">🔥</span>
            <div class="machine-name">Four Céramique</div>
            <div class="machine-desc">
                Analyse des capteurs de température et vitesse.<br>
                Détection d'anomalies par Isolation Forest.<br>
                Prédiction XGBoost en temps réel.
            </div>
            <span class="machine-badge badge-four">Données temps réel</span>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button("Accéder au Four", use_container_width=True, type="primary", key="btn_four"):
            st.session_state.machine = "four"
            st.rerun()
    with col_mid:
        st.markdown('<div class="divider">|</div>', unsafe_allow_html=True)
    with col_right:
        st.markdown("""
        <div class="machine-card">
            <span class="machine-icon">🌀</span>
            <div class="machine-name">Séchoir Céramique</div>
            <div class="machine-desc">
                Analyse des alarmes SCADA (Mars → Mai 2026).<br>
                Pressostats EAU11/12/21/22.<br>
                Prédiction XGBoost : severity + failure type.
            </div>
            <span class="machine-badge badge-sechoir">Données SCADA réelles</span>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button("Accéder au Séchoir", use_container_width=True, key="btn_sechoir"):
            st.session_state.machine = "sechoir"
            st.rerun()
    st.markdown("""
    <div class="footer-accueil">
        <span style="color:#22c55e;">●</span>&nbsp;
        Système opérationnel · CJO Poulina © 2026
    </div>
    """, unsafe_allow_html=True)

