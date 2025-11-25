import streamlit as st
import requests

"""
Ceci n'est qu'un exemple de composant de feedback.
"""

def feedback_component(pattern_id: int, backend_url: str, alpha: float = 0.3, beta: float = 0.3, key: int = 0, method: str = "Importance Sampling"):
    """Composant de feedback pour un motif"""
    rating_key = f"rating_{pattern_id}_{st.session_state.get('feedback_epoch', 0)}"
    if rating_key not in st.session_state:
        st.session_state[rating_key] = 0  # 0 = none, 1 = like, -1 = dislike

    col1, col2 = st.columns(2)

    # Show the Like button only if current rating is not 1
    with col1:
        if st.session_state[rating_key] != 1:
            if st.button("👍", key=f"like_{key}"):
                try:
                    payload = {
                        "index": int(pattern_id),
                        "rating": 1,
                        "alpha": float(alpha),
                        "beta": float(beta),
                        "method": method
                    }
                    # backend attends des champs Form -> utiliser data=
                    response = requests.post(f"{backend_url}/api/feedback", data=payload, timeout=5)
                    if response.status_code == 200:
                        st.session_state[rating_key] = 1
                        st.rerun()
                    else:
                        st.error(f"Erreur serveur ({response.status_code}) : {response.text}")
                except Exception as e:
                    st.error(f"Erreur: {e}")

    # Show the Dislike button only if current rating is not -1
    with col2:
        if st.session_state[rating_key] != -1:
            if st.button("👎", key=f"dislike_{key}"):
                try:
                    payload = {
                        "index": int(pattern_id),
                        "rating": -1,
                        "alpha": float(alpha),
                        "beta": float(beta),
                        "method": method
                    }
                    response = requests.post(f"{backend_url}/api/feedback", data=payload, timeout=5)
                    if response.status_code == 200:
                        st.session_state[rating_key] = -1
                        st.rerun()
                    else:
                        st.error(f"Erreur serveur ({response.status_code}) : {response.text}")
                except Exception as e:
                    st.error(f"Erreur: {e}")