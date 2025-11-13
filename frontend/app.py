import streamlit as st
import requests
import pandas as pd
import logging
from components.upload import upload_component
from components.visualizations import visualize_patterns
from components.feedback import feedback_component
from components.dataset_selector import dataset_selector_component, get_active_dataset_info
from components.evaluation import evaluation_component

# Logger
logger = logging.getLogger(__name__)

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
tab1, tab2, tab3, tab4 = st.tabs(["Upload", "🔍 Motifs", "📊 Analyse", "📈 Évaluation"])

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
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🚀 Lancer l'extraction", type="primary"):
                with st.spinner("Extraction en cours..."):
                    try:
                        # Appel API avec dataset_id
                        response = requests.post(
                            f"{BACKEND_URL}/api/mine",
                            json={
                                "dataset_id": st.session_state['active_dataset_id'],
                                "min_support": min_support,
                                "min_confidence": min_confidence
                            },
                            timeout=60
                        )
                        
                        if response.status_code == 200:
                            mining_results = response.json()
                            st.session_state['mining_results'] = mining_results
                            st.success(f"✅ {mining_results['message']}")
                            st.rerun()
                        else:
                            st.error(f"❌ Erreur: {response.status_code} - {response.text}")
                            
                    except requests.exceptions.Timeout:
                        st.error("❌ Délai d'attente dépassé. Le dataset est peut-être trop volumineux.")
                    except Exception as e:
                        st.error(f"❌ Erreur lors de l'extraction: {str(e)}")
        
        with col2:
            if st.button("🔄 Ré-échantillonner", help="Applique un nouvel échantillonnage basé sur vos feedbacks"):
                with st.spinner("Ré-échantillonnage en cours..."):
                    try:
                        response = requests.post(
                            f"{BACKEND_URL}/api/sample",
                            json={
                                "dataset_id": st.session_state['active_dataset_id'],
                                "k": k_samples,
                                "replacement": strategy == "avec",
                                "support_weight": 0.4,
                                "surprise_weight": 0.4,
                                "redundancy_weight": 0.2
                            },
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            sampling_results = response.json()
                            # Mettre à jour l'aperçu avec les motifs échantillonnés
                            if st.session_state.get('mining_results'):
                                st.session_state['mining_results']['patterns_preview'] = sampling_results['sampled_patterns']
                            st.success(f"✅ {sampling_results['message']}")
                            st.rerun()
                        else:
                            st.error(f"❌ Erreur: {response.status_code}")
                            
                    except Exception as e:
                        st.error(f"❌ Erreur: {str(e)}")
        
        with col3:
            if st.session_state.get('mining_results'):
                results = st.session_state['mining_results']
                st.metric("Temps d'extraction", f"{results['computation_time']:.2f}s")
        
        # Afficher les résultats si disponibles
        if st.session_state.get('mining_results'):
            results = st.session_state['mining_results']
            
            st.markdown("---")
            
            # Métriques globales
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Motifs Extraits", f"{results['num_patterns']:,}")
            with col2:
                st.metric("Règles Générées", f"{results['num_rules']:,}")
            with col3:
                st.metric("Temps", f"{results['computation_time']:.2f}s")
            
            st.markdown("---")
            
            # Tabs pour les résultats
            tab_patterns, tab_rules = st.tabs(["📋 Motifs Fréquents", "🔗 Règles d'Association"])
            
            with tab_patterns:
                st.subheader("Top 10 Motifs Fréquents")
                
                if results['patterns_preview']:
                    # Afficher les statistiques de feedback
                    try:
                        stats_response = requests.get(
                            f"{BACKEND_URL}/api/feedback/stats/{st.session_state['active_dataset_id']}"
                        )
                        if stats_response.status_code == 200:
                            feedback_stats = stats_response.json()['stats']
                            
                            # Afficher les stats en haut
                            st.markdown("### 📊 Statistiques de Feedback")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("👍 Likes", feedback_stats['likes'])
                            with col2:
                                st.metric("👎 Dislikes", feedback_stats['dislikes'])
                            with col3:
                                st.metric("Total", feedback_stats['total_feedbacks'])
                            with col4:
                                st.metric("Taux d'acceptation", f"{feedback_stats['acceptance_rate']}%")
                            
                            st.markdown("---")
                    except Exception as e:
                        logger.error(f"Erreur récupération stats: {str(e)}")
                    
                    # Convertir en DataFrame pour affichage
                    patterns_data = []
                    for idx, p in enumerate(results['patterns_preview']):
                        patterns_data.append({
                            "ID": idx,
                            "Items": ", ".join(p['itemset']),
                            "Support": f"{p['support']:.4f}",
                            "Longueur": p['length'],
                            "Couverture": f"{p['coverage']:.1f}"
                        })
                    
                    df_patterns = pd.DataFrame(patterns_data)
                    st.dataframe(df_patterns, use_container_width=True, hide_index=True)
                    
                    # Section de feedback interactif
                    st.markdown("### 💬 Feedback Interactif")
                    st.info("👉 Sélectionnez un motif et donnez votre avis pour améliorer les recommandations futures")
                    
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        pattern_id = st.selectbox(
                            "Sélectionner un motif",
                            options=range(len(results['patterns_preview'])),
                            format_func=lambda x: f"Motif {x}: {', '.join(results['patterns_preview'][x]['itemset'])}"
                        )
                    
                    with col2:
                        if st.button("👍 Like", type="primary", use_container_width=True):
                            try:
                                feedback_response = requests.post(
                                    f"{BACKEND_URL}/api/feedback",
                                    json={
                                        "dataset_id": st.session_state['active_dataset_id'],
                                        "pattern_id": pattern_id,
                                        "rating": 1,
                                        "comment": "Like"
                                    }
                                )
                                
                                if feedback_response.status_code == 200:
                                    result = feedback_response.json()
                                    if result.get('reweighted'):
                                        st.success("✅ Merci ! Les motifs ont été ré-pondérés.")
                                        st.rerun()
                                    else:
                                        st.success("✅ Feedback enregistré")
                                else:
                                    st.error(f"Erreur: {feedback_response.status_code}")
                                    
                            except Exception as e:
                                st.error(f"Erreur: {str(e)}")
                    
                    with col3:
                        if st.button("👎 Dislike", use_container_width=True):
                            try:
                                feedback_response = requests.post(
                                    f"{BACKEND_URL}/api/feedback",
                                    json={
                                        "dataset_id": st.session_state['active_dataset_id'],
                                        "pattern_id": pattern_id,
                                        "rating": -1,
                                        "comment": "Dislike"
                                    }
                                )
                                
                                if feedback_response.status_code == 200:
                                    result = feedback_response.json()
                                    if result.get('reweighted'):
                                        st.success("✅ Merci ! Les motifs ont été ré-pondérés.")
                                        st.rerun()
                                    else:
                                        st.success("✅ Feedback enregistré")
                                else:
                                    st.error(f"Erreur: {feedback_response.status_code}")
                                    
                            except Exception as e:
                                st.error(f"Erreur: {str(e)}")
                    
                    st.markdown("---")
                    
                    # Bouton pour voir tous les motifs
                    if st.button("📥 Voir tous les motifs"):
                        try:
                            response = requests.get(
                                f"{BACKEND_URL}/api/patterns/{st.session_state['active_dataset_id']}",
                                params={"limit": 1000}
                            )
                            if response.status_code == 200:
                                all_patterns = response.json()
                                st.session_state['all_patterns'] = all_patterns
                                st.info(f"Chargé {all_patterns['num_patterns']} motifs")
                        except Exception as e:
                            st.error(f"Erreur: {str(e)}")
                else:
                    st.info("Aucun motif trouvé avec ces paramètres")
            
            with tab_rules:
                st.subheader("Top 10 Règles d'Association")
                
                if results['rules_preview']:
                    # Convertir en DataFrame pour affichage
                    rules_data = []
                    for r in results['rules_preview']:
                        rules_data.append({
                            "Antécédents": ", ".join(r['antecedents']),
                            "Conséquents": ", ".join(r['consequents']),
                            "Support": f"{r['support']:.4f}",
                            "Confiance": f"{r['confidence']:.4f}",
                            "Lift": f"{r['lift']:.2f}"
                        })
                    
                    df_rules = pd.DataFrame(rules_data)
                    st.dataframe(df_rules, use_container_width=True, hide_index=True)
                    
                    # Bouton pour voir toutes les règles
                    if st.button("📥 Voir toutes les règles"):
                        try:
                            response = requests.get(
                                f"{BACKEND_URL}/api/rules/{st.session_state['active_dataset_id']}",
                                params={"limit": 1000}
                            )
                            if response.status_code == 200:
                                all_rules = response.json()
                                st.session_state['all_rules'] = all_rules
                                st.info(f"Chargé {all_rules['num_rules']} règles")
                        except Exception as e:
                            st.error(f"Erreur: {str(e)}")
                else:
                    st.info("Aucune règle trouvée avec ces paramètres")
        else:
            st.info("Cliquez sur 'Lancer l'extraction' pour commencer")
    
with tab3:
    st.header("Analyse et Visualisations")
    
    if not st.session_state.get('active_dataset_id'):
        st.warning("⚠️ Aucun dataset sélectionné")
        st.info("👉 Allez dans l'onglet 'Datasets' pour sélectionner un dataset")
    else:
        st.info("Les visualisations apparaîtront ici après l'extraction")

with tab4:
    # Partie 4 : Évaluation & Reproductibilité
    evaluation_component(
        backend_url=BACKEND_URL,
        dataset_id=st.session_state.get('active_dataset_id')
    )
