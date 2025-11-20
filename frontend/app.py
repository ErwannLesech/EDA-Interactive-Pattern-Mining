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

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Paramètres de mining
    st.subheader("Extraction")
    min_support = st.slider("Minimum support", 0.01, 0.5, 0.05)
    support_weight = st.slider("Poids du support", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    surprise_weight = st.slider("Poids de la surprise", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
    redundancy_weight = st.slider("Poids de la redondance", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    st.write(f"La somme des poids est : {support_weight + surprise_weight + redundancy_weight:.2f}")
    
    st.markdown("---")
    
    # Paramètres d'échantillonnage
    st.subheader("Échantillonnage")
    k_samples = st.number_input("Nombre de motifs", 10, 500, 50)
    strategy = st.selectbox("Remise", ["avec", "sans"])
    
    st.markdown("---")

    st.subheader("Feedback")
    alpha = st.number_input("Alpha (like)", 0.001, 1.0, 0.03, 0.001)
    beta = st.number_input("Beta (dislike)", 0.001, 1.0, 0.03, 0.001)
    st.markdown("---")
    st.info("Uploadez un fichier CSV, Excel, Json ou Txt sous les formats transactionnels, transactionnels inversés, séquentiels ou matriciels.")

# Corps principal - Ajout de l'onglet Échantillonnage
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload", "🔍 Motifs", "🎲 Échantillonnage", "📊 Analyse"])

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
        if "motifs_df" not in st.session_state:
            st.session_state["motifs_df"] = pd.DataFrame()
        if "sampled_df" not in st.session_state:
            st.session_state["sampled_df"] = pd.DataFrame()
        if "extraction_done" not in st.session_state:
            st.session_state["extraction_done"] = False
        if not st.session_state["extraction_done"]:
            if st.button("🚀 Lancer l'extraction", type="primary"):
                if support_weight + surprise_weight + redundancy_weight != 1:
                    st.error(f"❌ La somme des poids doit être égale à 1. Elle est égale à {support_weight + surprise_weight + redundancy_weight:.2f}. Ajustez les curseurs.")
                else:
                    with st.spinner("Extraction en cours..."):
                        response = requests.post(
                        f"{BACKEND_URL}/api/patterns/mine",
                        data={  # Utilise 'data' pour envoyer les paramètres en Form
                            "min_support": min_support,
                        "support_weight": support_weight,
                        "surprise_weight": surprise_weight,
                        "redundancy_weight": redundancy_weight,
                        "k": k_samples,
                        "replacement": strategy == "avec"
                    },
                    timeout=30
                    )
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state["motifs_df"] = pd.DataFrame(result.get("frequent_itemsets", []))
                        st.session_state["sampled_df"] = pd.DataFrame(result.get("sampled_patterns", []))
                        # bump feedback epoch so feedback buttons reset for the new sample
                        st.session_state["feedback_epoch"] = st.session_state.get("feedback_epoch", 0) + 1
                        if not st.session_state["motifs_df"].empty:
                            if len(st.session_state["motifs_df"]) < k_samples:
                                st.session_state['warning'] = True
                                st.session_state['message'] = f"⚠️ Seuls {len(st.session_state['motifs_df'])} motifs ont été extraits, inférieur au nombre demandé ({k_samples})."
                            else:
                                st.session_state['warning'] = False
                                st.session_state['message'] = result.get("message", "Extraction terminée !")
                        st.session_state["extraction_done"] = True
                        st.rerun()
                    else:
                        st.error("❌ Extraction impossible"+ f" (Statut {response.status_code})")
        else:
            if st.button("🔄 Relancer l'extraction"):
                if support_weight + surprise_weight + redundancy_weight != 1:
                    st.error(f"❌ La somme des poids doit être égale à 1. Elle est égale à {support_weight + surprise_weight + redundancy_weight:.2f}. Ajustez les curseurs.")
                else:
                    with st.spinner("Extraction en cours..."):
                        response = requests.post(
                        f"{BACKEND_URL}/api/patterns/resample",
                        data={  # Utilise 'data' pour envoyer les paramètres en Form
                            "min_support": min_support,
                        "support_weight": support_weight,
                        "surprise_weight": surprise_weight,
                        "redundancy_weight": redundancy_weight,
                        "k": k_samples,
                        "replacement": strategy == "avec"
                    },
                    timeout=30
                    )
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state["motifs_df"] = pd.DataFrame(result.get("frequent_itemsets", []))
                        st.session_state["sampled_df"] = pd.DataFrame(result.get("sampled_patterns", []))
                        # bump feedback epoch so feedback buttons reset for the new resample
                        st.session_state["feedback_epoch"] = st.session_state.get("feedback_epoch", 0) + 1
                        if not st.session_state["motifs_df"].empty:
                            if len(st.session_state["motifs_df"]) < k_samples:
                                st.session_state['warning'] = True
                                st.session_state['message'] = f"⚠️ Seuls {len(st.session_state['motifs_df'])} motifs ont été extraits, inférieur au nombre demandé ({k_samples})."
                            else:
                                st.session_state['warning'] = False
                                st.session_state['message'] = result.get("message", "Extraction terminée !")
                        st.session_state["extraction_done"] = True
                        st.rerun()
                    else:
                        st.error("❌ Extraction impossible"+ f" (Statut {response.status_code})")
        if st.session_state.get('warning', False) and st.session_state.get('extraction_done', False):
            st.warning(st.session_state['message'])
        elif st.session_state.get('extraction_done', False):
            st.success(st.session_state['message'])
        if not st.session_state["motifs_df"].empty:
            visualize_patterns(st.session_state["motifs_df"])
        if not st.session_state["sampled_df"].empty:
            st.subheader("Feedback sur les motifs échantillonnés")
            n_cols = 5
            rows = [st.session_state["sampled_df"].iloc[i:i+n_cols] for i in range(0, len(st.session_state["sampled_df"]), n_cols)]
            for row_group in rows:
                cols = st.columns(len(row_group))
                for col, (_, row) in zip(cols, row_group.iterrows()):
                    with col:
                        # Affiche l'itemset (formaté si besoin)
                        if isinstance(row['itemset'], (list, set, tuple)):
                            itemset_str = ", ".join(map(str, row['itemset']))
                        else:
                            itemset_str = str(row['itemset'])
                        st.markdown(f"**{itemset_str}**")
                        # Boutons de feedback
                        feedback_component(
                            pattern_id=row.get("index", row.name),
                            backend_url=BACKEND_URL,
                            alpha=alpha,
                            beta=beta,
                            key= row.get("id", row.name)
                        )
        
with tab3:
    st.header("🎲 Échantillonnage de Motifs")
    
    # Utiliser le composant d'échantillonnage
    dataset_id = st.session_state.get('active_dataset_id')
    sampling_tab(BACKEND_URL, dataset_id)
    
with tab4:
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
