import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def sampling_component(backend_url: str, dataset_id: str):
    """
    Composant pour l'échantillonnage de motifs
    
    Args:
        backend_url: URL du backend
        dataset_id: ID du dataset actif
    """
    
    st.subheader("⚙️ Configuration de l'échantillonnage")
    
    # Sélection de la méthode
    method = st.selectbox(
        "Méthode d'échantillonnage",
        ["Importance Sampling", "TwoStep Sampling", "GDPS"],
        help="Choisissez la méthode d'échantillonnage à utiliser"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        k = st.number_input(
            "Nombre de motifs (k)", 
            min_value=10, 
            max_value=1000, 
            value=50,
            help="Nombre de motifs à échantillonner"
        )
    
    # Paramètres spécifiques à chaque méthode
    if method == "Importance Sampling":
        with col2:
            replacement = st.checkbox(
                "Avec remise", 
                value=True,
                help="Échantillonner avec ou sans remplacement"
            )
        
        st.markdown("### Poids des critères")
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            support_weight = st.slider(
                "Support", 
                0.0, 1.0, 0.33,
                help="Importance du support dans le score composite"
            )
        
        with col_b:
            surprise_weight = st.slider(
                "Surprise", 
                0.0, 1.0, 0.33,
                help="Importance de la surprise (écart au modèle d'indépendance)"
            )
        
        with col_c:
            redundancy_weight = st.slider(
                "Anti-redondance", 
                0.0, 1.0, 0.34,
                help="Importance de la diversité (pénalise les motifs similaires)"
            )
        
        # Normaliser les poids
        total = support_weight + surprise_weight + redundancy_weight
        if total > 0:
            support_weight /= total
            surprise_weight /= total
            redundancy_weight /= total
        
        params = {
            "support_weight": support_weight,
            "surprise_weight": surprise_weight,
            "redundancy_weight": redundancy_weight,
            "replacement": replacement
        }
    
    elif method == "TwoStep Sampling":
        st.info("🔹 TwoStep : Échantillonne une transaction puis un sous-ensemble de cette transaction")
        params = {}
    
    else:  # GDPS
        with col2:
            utility = st.selectbox(
                "Type d'utilité",
                ["freq", "area", "decay"],
                help="freq: uniforme, area: proportionnel à la taille, decay: décroissance exponentielle"
            )
        
        col_min, col_max = st.columns(2)
        
        with col_min:
            min_norm = st.number_input(
                "Taille min", 
                min_value=1, 
                max_value=10, 
                value=1,
                help="Taille minimale des motifs"
            )
        
        with col_max:
            max_norm = st.number_input(
                "Taille max", 
                min_value=1, 
                max_value=20, 
                value=10,
                help="Taille maximale des motifs"
            )
        
        params = {
            "min_norm": min_norm,
            "max_norm": max_norm,
            "utility": utility
        }
    
    # Bouton pour lancer l'échantillonnage
    if st.button("🚀 Lancer l'échantillonnage", type="primary", use_container_width=True):
        with st.spinner(f"Échantillonnage en cours avec {method}..."):
            try:
                # Préparer les données
                form_data = {
                    "dataset_id": dataset_id,
                    "k": k,
                    **params
                }
                
                # Appeler l'API appropriée
                if method == "Importance Sampling":
                    endpoint = f"{backend_url}/api/sample/importance"
                elif method == "TwoStep Sampling":
                    endpoint = f"{backend_url}/api/sample/twostep"
                else:  # GDPS
                    endpoint = f"{backend_url}/api/sample/gdps"
                
                response = requests.post(endpoint, data=form_data, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Sauvegarder dans la session
                    st.session_state['sampled_patterns'] = result
                    st.session_state['sampling_method'] = method
                    
                    st.success(f"✅ {k} motifs échantillonnés avec succès !")
                    
                    # Afficher les résultats
                    display_sampled_patterns(result, method)
                    
                else:
                    st.error(f"❌ Erreur: {response.status_code} - {response.text}")
            
            except requests.exceptions.Timeout:
                st.error("⏱️ Timeout: L'échantillonnage prend trop de temps")
            except Exception as e:
                st.error(f"❌ Erreur: {str(e)}")
    
    # Si des motifs ont déjà été échantillonnés, les afficher
    if 'sampled_patterns' in st.session_state:
        st.markdown("---")
        display_sampled_patterns(
            st.session_state['sampled_patterns'], 
            st.session_state.get('sampling_method', method)
        )


def display_sampled_patterns(result: dict, method: str):
    """Affiche les motifs échantillonnés"""
    
    st.subheader("📊 Motifs échantillonnés")
    
    # Informations générales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Méthode", result.get("method", method).replace("_", " ").title())
    
    with col2:
        st.metric("Échantillonnés", result.get("k", 0))
    
    with col3:
        if "total_patterns" in result:
            st.metric("Total motifs", result["total_patterns"])
    
    with col4:
        if "parameters" in result:
            params = result["parameters"]
            if "min_support" in params:
                st.metric("Min Support", f"{params['min_support']:.3f}")
    
    # Afficher les motifs
    patterns = result.get("sampled_patterns", [])
    
    if patterns:
        # Créer un DataFrame selon le format
        pattern_data = []
        
        # Détecter le format (nouveau format avec dict ou ancien format avec tuples)
        if patterns and isinstance(patterns[0], dict):
            # Nouveau format avec informations détaillées
            for pattern in patterns:
                pattern_data.append({
                    "ID": pattern.get("index", 0),
                    "Motif": ", ".join(sorted(pattern.get("itemset", []))),
                    "Taille": pattern.get("length", 0),
                    "Support": round(pattern.get("support", 0), 4),
                    "Surprise": round(pattern.get("surprise", 0), 4),
                    "Redondance": round(pattern.get("redundancy", 0), 4),
                    "Score": round(pattern.get("composite_score", 0), 6)
                })
        else:
            # Ancien format (tuples ou listes simples)
            for idx, pattern in enumerate(patterns):
                if isinstance(pattern, tuple):
                    pattern_data.append({
                        "ID": idx,
                        "Motif": str(pattern[1]) if len(pattern) > 1 else str(pattern),
                        "Taille": len(pattern[1]) if len(pattern) > 1 else len(pattern)
                    })
                else:
                    pattern_data.append({
                        "ID": idx,
                        "Motif": str(pattern),
                        "Taille": len(pattern) if isinstance(pattern, list) else 1
                    })
        
        df_patterns = pd.DataFrame(pattern_data)
        
        # Afficher le tableau
        st.dataframe(df_patterns, use_container_width=True, hide_index=True)
        
        # Visualisations
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            # Distribution des tailles
            fig_size = px.histogram(
                df_patterns, 
                x="Taille",
                title="Distribution des tailles de motifs",
                labels={"Taille": "Taille du motif", "count": "Fréquence"}
            )
            st.plotly_chart(fig_size, use_container_width=True)
        
        with col_viz2:
            # Si on a les scores, afficher la distribution
            if "Score" in df_patterns.columns:
                fig_score = px.histogram(
                    df_patterns,
                    x="Score",
                    title="Distribution des scores composites",
                    labels={"Score": "Score composite", "count": "Fréquence"}
                )
                st.plotly_chart(fig_score, use_container_width=True)
            elif "Support" in df_patterns.columns:
                fig_support = px.histogram(
                    df_patterns,
                    x="Support",
                    title="Distribution du support",
                    labels={"Support": "Support", "count": "Fréquence"}
                )
                st.plotly_chart(fig_support, use_container_width=True)
        
        # Graphique supplémentaire pour Importance Sampling
        if method == "Importance Sampling" and "Support" in df_patterns.columns and "Surprise" in df_patterns.columns:
            fig_scatter = px.scatter(
                df_patterns,
                x="Support",
                y="Surprise",
                size="Taille",
                color="Redondance",
                hover_data=["Motif"],
                title="Support vs Surprise (taille = taille du motif, couleur = redondance)",
                labels={"Support": "Support", "Surprise": "Surprise"}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Option pour télécharger
        csv = df_patterns.to_csv(index=False)
        # Créer une clé unique basée sur l'identifiant de l'objet result
        unique_id = id(result)
        # st.download_button(
        #     label="📥 Télécharger les motifs (CSV)",
        #     data=csv,
        #     file_name=f"motifs_{method.replace(' ', '_')}_{result.get('k', 0)}.csv",
        #     mime="text/csv",
        #     key=f"download_patterns_{method.replace(' ', '_')}_{result.get('k', 0)}_{unique_id}"
        # )



def feedback_component_with_sampling():
    """Composant pour donner du feedback sur les motifs échantillonnés"""
    
    if 'sampled_patterns' not in st.session_state:
        st.info("👆 Échantillonnez d'abord des motifs pour donner du feedback")
        return
    
    st.subheader("💬 Feedback sur les motifs")
    
    patterns = st.session_state['sampled_patterns'].get('sampled_patterns', [])
    
    if not patterns:
        return
    
    # Sélectionner un motif
    pattern_index = st.selectbox(
        "Sélectionnez un motif",
        range(len(patterns)),
        format_func=lambda x: f"Motif {x}: {patterns[x]}"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("👍 J'aime", use_container_width=True):
            st.success(f"✅ Feedback positif pour le motif {pattern_index}")
            # TODO: Appeler l'API de feedback
    
    with col2:
        if st.button("👎 Je n'aime pas", use_container_width=True):
            st.warning(f"⚠️ Feedback négatif pour le motif {pattern_index}")
            # TODO: Appeler l'API de feedback


# Fonction principale pour l'onglet d'échantillonnage
def sampling_tab(backend_url: str, dataset_id: str):
    """Onglet complet pour l'échantillonnage"""
    
    if not dataset_id:
        st.warning("⚠️ Aucun dataset chargé")
        st.info("👉 Allez dans l'onglet 'Upload' pour charger un dataset")
        return
    
    # Composant d'échantillonnage
    sampling_component(backend_url, dataset_id)
    
    st.markdown("---")
    
    # Composant de feedback
    feedback_component_with_sampling()
