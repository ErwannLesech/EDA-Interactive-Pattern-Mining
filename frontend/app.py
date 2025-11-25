import streamlit as st
import requests
import pandas as pd
from components.upload import upload_component
from components.visualizations import visualize_patterns
from components.feedback import feedback_component
from components.sampling import sampling_tab

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

# Corps principal - Ajout de l'onglet Échantillonnage
tab1,tab2, tab3 = st.tabs(["📤 Upload", "🎲 Échantillonnage", "📊 Analyse"])

with tab1:
    upload_component(BACKEND_URL)


        
with tab2:
    st.header("🎲 Échantillonnage de Motifs")
    
    # Utiliser le composant d'échantillonnage
    dataset_id = st.session_state.get('active_dataset_id')
    sampling_tab(BACKEND_URL, dataset_id)
    
with tab3:
    st.header("📊 Évaluation & Reproductibilité")
    
    if not st.session_state.get('extraction_done'):
        st.warning("⚠️ Aucune extraction effectuée")
        st.info("👉 Allez dans l'onglet 'Motifs' pour lancer l'extraction avant d'évaluer")
    else:
        from components.evaluation import display_evaluation_metrics, display_evaluation_summary
        
        st.success("✅ Évaluation disponible pour les motifs extraits")
        
        # Bouton pour lancer l'évaluation
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("""
            Cette analyse évalue la qualité de l'échantillonnage selon plusieurs critères :
            - **Taux d'acceptation** : Pourcentage de motifs appréciés par l'utilisateur
            - **Diversité** : Variété des motifs échantillonnés (dissimilarité)
            - **Couverture** : Représentativité de l'échantillon par rapport au pool complet
            - **Stabilité** : Reproductibilité avec différentes seeds aléatoires
            - **Performance** : Temps de réponse de l'algorithme
            """)
        
        with col2:
            if st.button("🚀 Évaluer", type="primary", use_container_width=True):
                with st.spinner("Évaluation en cours..."):
                    try:
                        response = requests.get(f"{BACKEND_URL}/api/patterns/evaluate", timeout=30)
                        if response.status_code == 200:
                            evaluation_data = response.json()
                            st.session_state['evaluation_data'] = evaluation_data
                            st.success("✅ Évaluation terminée!")
                            st.rerun()
                        else:
                            st.error(f"❌ Erreur lors de l'évaluation (code {response.status_code})")
                    except Exception as e:
                        st.error(f"❌ Erreur: {str(e)}")
        
        st.markdown("---")
        
        # Afficher les résultats si disponibles
        if 'evaluation_data' in st.session_state:
            evaluation_data = st.session_state['evaluation_data']
            
            # Résumé compact
            st.subheader("📈 Résumé des Métriques")
            display_evaluation_summary(evaluation_data)
            
            st.markdown("---")
            
            # Métriques détaillées
            display_evaluation_metrics(evaluation_data)
            
            st.markdown("---")
            
            # Export des résultats
            st.subheader("💾 Export des Résultats")
            col1, col2 = st.columns(2)
            
            with col1:
                # Préparer les données pour export CSV
                eval_results = evaluation_data.get("evaluation", {})
                export_data = {
                    "Métrique": [],
                    "Valeur": []
                }
                
                # Taux d'acceptation
                acceptance = eval_results.get("acceptance", {})
                export_data["Métrique"].append("Taux d'acceptation")
                export_data["Valeur"].append(f"{acceptance.get('acceptance_rate', 0):.2%}")
                export_data["Métrique"].append("Total feedbacks")
                export_data["Valeur"].append(acceptance.get('total_feedbacks', 0))
                
                # Diversité
                diversity = eval_results.get("diversity", {})
                export_data["Métrique"].append("Score de diversité")
                export_data["Valeur"].append(f"{diversity.get('diversity_score', 0):.3f}")
                export_data["Métrique"].append("Items uniques")
                export_data["Valeur"].append(diversity.get('unique_items_count', 0))
                
                # Couverture
                coverage = eval_results.get("coverage", {})
                export_data["Métrique"].append("Couverture motifs")
                export_data["Valeur"].append(f"{coverage.get('pattern_coverage', 0):.2%}")
                export_data["Métrique"].append("Couverture items")
                export_data["Valeur"].append(f"{coverage.get('item_coverage', 0):.2%}")
                
                # Stabilité
                stability = eval_results.get("stability", {})
                if stability.get("stability_score"):
                    export_data["Métrique"].append("Score de stabilité")
                    export_data["Valeur"].append(f"{stability.get('stability_score', 0):.3f}")
                
                # Performance
                response_time = eval_results.get("response_time", {})
                if response_time.get("mean_time"):
                    export_data["Métrique"].append("Temps moyen (s)")
                    export_data["Valeur"].append(f"{response_time.get('mean_time', 0):.3f}")
                
                # Score global
                export_data["Métrique"].append("Score global")
                export_data["Valeur"].append(f"{eval_results.get('overall_score', 0):.2%}")
                
                export_df = pd.DataFrame(export_data)
                csv = export_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Télécharger les métriques (CSV)",
                    data=csv,
                    file_name="evaluation_metrics.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                import json
                json_str = json.dumps(evaluation_data, indent=2)
                st.download_button(
                    label="📥 Télécharger le rapport complet (JSON)",
                    data=json_str,
                    file_name="evaluation_report.json",
                    mime="application/json",
                    use_container_width=True
                )
        else:
            st.info("👆 Cliquez sur 'Évaluer' pour générer les métriques d'évaluation")
