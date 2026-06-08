import streamlit as st
st.set_page_config(
    page_title="CJO - Maintenance Prédictive",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed"
)
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "login_error" not in st.session_state:
    st.session_state.login_error = False
if "machine" not in st.session_state:
    st.session_state.machine = None
if not st.session_state.logged_in:
    from login import show_login
    show_login()
elif st.session_state.machine == "four":
    from dashboard_four import show_dashboard_four
    show_dashboard_four()
elif st.session_state.machine == "sechoir":
    from dashboard_sechoir import show_dashboard_sechoir
    show_dashboard_sechoir()
else:
    from page_accueil import show_accueil
    show_accueil()

