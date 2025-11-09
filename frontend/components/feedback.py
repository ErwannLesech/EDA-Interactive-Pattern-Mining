import streamlit as st
import requests

"""
Ceci n'est qu'un exemple de composant de feedback.
"""

def feedback_component(pattern_id: int, backend_url: str):
    """Composant de feedback pour un motif"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("👍", key=f"like_{pattern_id}"):
            response = requests.post(
                f"{backend_url}/api/feedback",
                json={"pattern_id": pattern_id, "rating": 1}
            )
            if response.status_code == 200:
                st.success("Feedback enregistré!")
    
    with col2:
        if st.button("👎", key=f"dislike_{pattern_id}"):
            response = requests.post(
                f"{backend_url}/api/feedback",
                json={"pattern_id": pattern_id, "rating": -1}
            )
            if response.status_code == 200:
                st.success("Feedback enregistré!")
