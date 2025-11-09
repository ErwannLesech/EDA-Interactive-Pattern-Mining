import streamlit as st
import requests
import pandas as pd
from components.upload import upload_component
from components.visualizations import visualize_patterns
from components.feedback import feedback_component
from components.dataset_selector import dataset_selector_component, get_active_dataset_info

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
    st.info("📖 Uploadez un fichier CSV, Excel, Json ou Txt sous les formats transactionnels, transactionnels inversés, séquentiels ou matriciels.")

# Afficher le dataset actif dans la sidebar
with st.sidebar:
    st.markdown("---")
    st.subheader("📊 Dataset Actif")
    
    active_info = get_active_dataset_info(BACKEND_URL)
    if active_info:
        st.success("✅ Dataset chargé")
        st.metric("Lignes", f"{active_info['rows']:,}")
        st.metric("Colonnes", f"{len(active_info['columns'])}")
    else:
        st.warning("⚠️ Aucun dataset sélectionné")
        st.caption("Allez dans l'onglet 'Datasets' pour en sélectionner un")

# Corps principal
tab1, tab2, tab3 = st.tabs(["Upload", "🔍 Motifs", "📊 Analyse"])

with tab1:
    upload_component(BACKEND_URL)
    
    st.header("Gestion des Datasets")
    active_dataset_id = dataset_selector_component(BACKEND_URL)
    
    if active_dataset_id:
        st.divider()
        st.info(f"💡 **Tip**: Le dataset actif sera utilisé pour l'extraction de motifs dans l'onglet 'Motifs'")

with tab2:
    st.header("Motifs Découverts")
    
    # Vérifier qu'un dataset est sélectionné
    if not st.session_state.get('active_dataset_id'):
        st.warning("⚠️ Aucun dataset sélectionné")
        st.info("👉 Allez dans l'onglet 'Datasets' pour sélectionner un dataset avant de lancer l'extraction")
    else:
        st.success(f"✅ Dataset actif: `{st.session_state['active_dataset_id'][:8]}...`")
        
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
        st.warning("⚠️ Aucun dataset sélectionné")
        st.info("👉 Allez dans l'onglet 'Datasets' pour sélectionner un dataset")
    else:
        st.info("Les visualisations apparaîtront ici après l'extraction")
