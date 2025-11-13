import streamlit as st
import requests
import pandas as pd
from components.upload import upload_component
from components.visualizations import visualize_patterns
from components.feedback import feedback_component

# Configuration de la page
st.set_page_config(
    page_title="Pattern Mining Interactive",
    page_icon="🔍",
    layout="wide"
)

# URL du backend
BACKEND_URL = "http://backend:8000"

# Titre
st.title("🔍 Pattern Mining Interactive")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Paramètres de mining
    st.subheader("Extraction")
    min_support = st.slider("Minimum", 0.01, 0.5, 0.05)
    min_confidence = st.slider("Maximum", 0.1, 1.0, 0.5)
    max_length = st.slider("Autres paramètres", 2, 10, 5)
    
    st.markdown("---")
    
    # Paramètres d'échantillonnage
    st.subheader("Échantillonnage")
    k_samples = st.number_input("Nombre de motifs", 10, 500, 50)
    strategy = st.selectbox("Remise", ["avec", "sans"])
    
    st.markdown("---")
    st.info("Uploadez un fichier CSV, Excel, Json ou Txt sous les formats transactionnels, transactionnels inversés, séquentiels ou matriciels.")

# Corps principal
tab1, tab2, tab3 = st.tabs(["📤 Upload", "🔍 Motifs", "📊 Analyse"])

with tab1:
    upload_component(BACKEND_URL)

with tab2:
    st.header("Motifs Découverts")
    
    # Vérifier qu'un dataset est chargé
    if not st.session_state.get('active_dataset_id'):
        st.warning("⚠️ Aucun dataset chargé")
        st.info("👉 Allez dans l'onglet 'Upload' pour charger un dataset avant de lancer l'extraction")
    else:
        dataset_name = st.session_state.get('active_dataset_name', 'Dataset')
        st.success(f"✅ Dataset actif: **{dataset_name}**")
        
        if st.button("🚀 Lancer l'extraction", type="primary"):
            with st.spinner("Extraction en cours..."):
                # TODO: Appel API avec dataset_id
                st.success("✅ Extraction en cours de développement!")
        
        # Emplacement pour feedback
        # feedback_component()
        st.info("Les motifs extraits apparaîtront ici après l'extraction")
        st.info("Le composant de feedback sera intégré avec les motifs extraits")
    
with tab3:
    st.header("Analyse et Visualisations")
    
    if not st.session_state.get('active_dataset_id'):
        st.warning("⚠️ Aucun dataset chargé")
        st.info("👉 Allez dans l'onglet 'Upload' pour charger un dataset avant de lancer l'analyse")
    else:
        dataset_id = st.session_state['active_dataset_id']
        dataset_name = st.session_state.get('active_dataset_name', 'Dataset')
        
        st.success(f"✅ Dataset actif: **{dataset_name}**")
        
        # Récupérer les infos du dataset
        try:
            response = requests.get(f"{BACKEND_URL}/api/dataset/{dataset_id}", timeout=5)
            if response.status_code == 200:
                dataset_info = response.json()
                
                # Afficher les statistiques du dataset
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("� Lignes", f"{dataset_info['rows']:,}")
                with col2:
                    st.metric("📐 Colonnes", f"{len(dataset_info['columns'])}")
                with col3:
                    st.metric("🔍 Dataset ID", dataset_id[:8] + "...")
                
                st.markdown("---")
                
                # Bouton pour lancer l'analyse
                if st.button("🚀 Lancer l'analyse", type="primary", use_container_width=True):
                    with st.spinner("Analyse en cours..."):
                        # TODO: Appel API d'analyse avec dataset_id
                        # Par exemple: requests.post(f"{BACKEND_URL}/api/analyze/{dataset_id}", ...)
                        st.success("✅ Analyse lancée!")
                        st.info("📊 L'analyse utilisera automatiquement le dataset chargé")
                
                # Zone pour afficher les résultats d'analyse
                st.markdown("---")
                st.subheader("📊 Résultats d'analyse")
                st.info("Les résultats d'analyse et visualisations apparaîtront ici")
                
                # Aperçu du dataset
                with st.expander("👁️ Aperçu du dataset"):
                    preview_df = pd.DataFrame(dataset_info['preview'])
                    st.dataframe(preview_df, use_container_width=True)
                
            else:
                st.error("❌ Impossible de récupérer les informations du dataset")
                
        except Exception as e:
            st.error(f"❌ Erreur de connexion au backend: {str(e)}")
