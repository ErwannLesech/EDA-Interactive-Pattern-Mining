import streamlit as st
import requests
import pandas as pd

def upload_component(backend_url: str):
    """Composant d'upload de fichier"""
    st.header("📤 Upload de Dataset")
    
    uploaded_file = st.file_uploader(
        "Choisissez un fichier CSV ou Excel",
        type=["csv", "xlsx", "xls"]
    )
    
    if uploaded_file is not None:
        # Affichage du nom
        st.success(f"Fichier chargé : {uploaded_file.name}")
        
        # Envoi au backend
        if st.button("Valider et traiter"):
            with st.spinner("Envoi au serveur..."):
                files = {"file": uploaded_file.getvalue()}
                
                try:
                    response = requests.post(
                        f"{backend_url}/api/upload",
                        files={"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        st.success("✅ Fichier traité avec succès!")
                        
                        # Affichage des stats
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Nombre de lignes", data["rows"])
                        with col2:
                            st.metric("Nombre de colonnes", len(data["columns"]))
                        
                        # Aperçu
                        st.subheader("Aperçu des données")
                        preview_df = pd.DataFrame(data["preview"])
                        st.dataframe(preview_df)
                        
                    else:
                        st.error(f"Erreur : {response.json()}")
                        
                except Exception as e:
                    st.error(f"Erreur de connexion : {str(e)}")
