import streamlit as st
from pathlib import Path
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "cjo2026"
def load_css():
    css_path = Path(__file__).parent / "login.css"
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
def show_login():
    load_css()
    logo = Path(__file__).parent.parent / "cjo-maintenance_assets" / "polina-logo.png"
    col1, col2, col3 = st.columns([1, 2, 1])
    with col3:
        st.image(str(logo), width=80)
    st.markdown("""
    <div class="logo-wrap">
        <div class="brand">CJO Poulina</div>
        <div class="brand-sub">Système de Maintenance Prédictive</div>
    </div>
    """, unsafe_allow_html=True)
    if st.session_state.login_error:
        st.error("Identifiants incorrects. Veuillez réessayer.")
    st.markdown("<span class='f-label'>Nom d'utilisateur</span>", unsafe_allow_html=True)
    username = st.text_input(
        "u", placeholder="Entrez votre identifiant",
        label_visibility="collapsed", key="inp_user"
    )
    st.markdown("<div class='gap'></div>", unsafe_allow_html=True)
    st.markdown("<span class='f-label'>Mot de passe</span>", unsafe_allow_html=True)
    password = st.text_input(
        "p", placeholder="••••••••",
        type="password",
        label_visibility="collapsed", key="inp_pass"
    )
    if st.button("Se connecter"):
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            st.session_state.logged_in = True
            st.session_state.login_error = False
            st.rerun()
        else:
            st.session_state.login_error = True
            st.rerun()
    st.markdown("""
    <div class="footer">
        <span class="live-dot"></span>
        Système opérationnel · CJO Poulina © 2026
    </div>
    """, unsafe_allow_html=True)

