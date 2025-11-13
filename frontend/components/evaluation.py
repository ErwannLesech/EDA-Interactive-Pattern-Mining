"""
Composant Streamlit pour l'évaluation et la visualisation des métriques.
Partie 4 : Évaluation & reproductibilité

Auteurs: Lesech Erwann, Le Riboter Aymeric, Aubron Abel, Claude Nathan
Date: Novembre 2025
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Any


def display_metric_card(title: str, value: Any, delta: str = None, help_text: str = None):
    """Affiche une carte de métrique stylisée."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.metric(label=title, value=value, delta=delta, help=help_text)


def plot_diversity_metrics(diversity_data: Dict[str, Any]):
    """
    Affiche les métriques de diversité avec graphiques comparatifs.
    
    Args:
        diversity_data: Dict contenant les métriques Jaccard, Cosine, Hamming
    """
    st.subheader("📊 Diversité des Motifs")
    
    # Vérifier que les données existent
    if not diversity_data or 'error' in diversity_data.get('jaccard', {}):
        st.warning("⚠️ Données de diversité non disponibles")
        return
    
    # Préparer les données pour la comparaison
    methods = []
    avg_scores = []
    min_scores = []
    max_scores = []
    std_scores = []
    
    for method, data in diversity_data.items():
        if 'error' not in data:
            methods.append(method.capitalize())
            avg_scores.append(data.get('average_diversity', 0))
            min_scores.append(data.get('min_diversity', 0))
            max_scores.append(data.get('max_diversity', 0))
            std_scores.append(data.get('std_diversity', 0))
    
    # Graphique comparatif des moyennes
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=methods,
            y=avg_scores,
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
            text=[f"{score:.3f}" for score in avg_scores],
            textposition='auto',
        ))
        fig.update_layout(
            title="Diversité Moyenne par Méthode",
            xaxis_title="Méthode",
            yaxis_title="Score de Diversité",
            height=400,
            yaxis_range=[0, 1]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Box plot pour montrer la distribution
        fig = go.Figure()
        for i, method in enumerate(methods):
            fig.add_trace(go.Box(
                y=[min_scores[i], avg_scores[i], max_scores[i]],
                name=method,
                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'][i]
            ))
        fig.update_layout(
            title="Distribution de la Diversité",
            yaxis_title="Score",
            height=400,
            yaxis_range=[0, 1]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Métriques détaillées
    st.markdown("---")
    cols = st.columns(len(methods))
    for i, method in enumerate(methods):
        with cols[i]:
            st.markdown(f"**{method}**")
            st.metric("Moyenne", f"{avg_scores[i]:.3f}")
            st.metric("Écart-type", f"{std_scores[i]:.3f}")
            st.caption(f"Min: {min_scores[i]:.3f} | Max: {max_scores[i]:.3f}")


def plot_coverage_metrics(coverage_data: Dict[str, Any]):
    """
    Affiche les métriques de couverture.
    
    Args:
        coverage_data: Dict avec les données de couverture
    """
    st.subheader("🎯 Couverture des Transactions")
    
    if 'error' in coverage_data:
        st.warning(f"⚠️ {coverage_data['error']}")
        return
    
    coverage_rate = coverage_data.get('coverage_rate', 0)
    covered = coverage_data.get('covered_transactions', 0)
    total = coverage_data.get('total_transactions', 0)
    
    # Gauge pour le taux de couverture
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=coverage_rate,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Taux de Couverture (%)"},
            delta={'reference': 70, 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#4ECDC4"},
                'steps': [
                    {'range': [0, 40], 'color': "#FFE5E5"},
                    {'range': [40, 70], 'color': "#FFF4E5"},
                    {'range': [70, 100], 'color': "#E5FFE5"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.metric("Transactions Couvertes", f"{covered:,}", help="Nombre de transactions couvertes par au moins un motif")
        st.metric("Total Transactions", f"{total:,}")
        avg_coverage = coverage_data.get('average_coverage_per_pattern', 0)
        st.metric("Moyenne/Motif", f"{avg_coverage:.1f}", help="Nombre moyen de transactions couvertes par motif")
    
    # Détails supplémentaires
    if 'min_coverage' in coverage_data and 'max_coverage' in coverage_data:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Min Couverture", coverage_data['min_coverage'])
        with col2:
            st.metric("Moyenne", f"{avg_coverage:.1f}")
        with col3:
            st.metric("Max Couverture", coverage_data['max_coverage'])


def plot_acceptance_metrics(acceptance_data: Dict[str, Any]):
    """
    Affiche les métriques d'acceptation utilisateur.
    
    Args:
        acceptance_data: Dict avec les données de feedback
    """
    st.subheader("👍 Taux d'Acceptation Utilisateur")
    
    total = acceptance_data.get('total_feedbacks', 0)
    
    if total == 0:
        st.info("ℹ️ Aucun feedback utilisateur enregistré. Commencez à évaluer les motifs pour voir les statistiques.")
        return
    
    acceptance_rate = acceptance_data.get('acceptance_rate', 0)
    likes = acceptance_data.get('likes', 0)
    dislikes = acceptance_data.get('dislikes', 0)
    neutral = acceptance_data.get('neutral', 0)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Métriques principales
        st.metric("Taux d'Acceptation", f"{acceptance_rate:.1f}%", 
                 delta=f"{acceptance_rate - 50:.1f}% vs 50%",
                 help="Pourcentage de likes par rapport au total de feedbacks")
        st.metric("Total Feedbacks", total)
    
    with col2:
        # Graphique en camembert
        labels = ['👍 Likes', '👎 Dislikes', '😐 Neutre']
        values = [likes, dislikes, neutral]
        colors = ['#28a745', '#dc3545', '#ffc107']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.4,
            marker_colors=colors,
            textinfo='label+percent+value',
            textposition='auto'
        )])
        fig.update_layout(
            title="Distribution des Feedbacks",
            height=300,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)


def plot_stability_metrics(stability_data: Dict[str, Any]):
    """
    Affiche les métriques de stabilité.
    
    Args:
        stability_data: Dict avec les données de stabilité
    """
    st.subheader("🔄 Stabilité de l'Échantillonnage")
    
    if 'error' in stability_data:
        st.info(f"ℹ️ {stability_data['error']}")
        return
    
    stability_score = stability_data.get('stability_score', 0)
    avg_similarity = stability_data.get('average_jaccard_similarity', 0)
    std_similarity = stability_data.get('std_jaccard_similarity', 0)
    num_runs = stability_data.get('num_runs', 0)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Gauge pour le score de stabilité
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=stability_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Score de Stabilité (%)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#45B7D1"},
                'steps': [
                    {'range': [0, 50], 'color': "#FFE5E5"},
                    {'range': [50, 75], 'color': "#FFF4E5"},
                    {'range': [75, 100], 'color': "#E5FFE5"}
                ],
                'threshold': {
                    'line': {'color': "green", 'width': 4},
                    'thickness': 0.75,
                    'value': 75
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.metric("Similarité Moyenne", f"{avg_similarity:.3f}", 
                 help="Similarité de Jaccard moyenne entre runs")
        st.metric("Écart-type", f"{std_similarity:.3f}",
                 help="Plus faible = plus stable")
        st.metric("Nombre de Runs", num_runs)
        
        if 'min_similarity' in stability_data and 'max_similarity' in stability_data:
            st.caption(f"Min: {stability_data['min_similarity']:.3f} | Max: {stability_data['max_similarity']:.3f}")


def plot_performance_metrics(performance_data: Dict[str, Any]):
    """
    Affiche les métriques de performance.
    
    Args:
        performance_data: Dict avec les données de temps de réponse
    """
    st.subheader("⚡ Performance & Temps de Réponse")
    
    if 'error' in performance_data:
        st.info(f"ℹ️ {performance_data['error']}")
        return
    
    avg_time = performance_data.get('average_time', 0)
    min_time = performance_data.get('min_time', 0)
    max_time = performance_data.get('max_time', 0)
    std_time = performance_data.get('std_time', 0)
    meets_target = performance_data.get('meets_target', False)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Indicateur de performance
        target_color = "green" if meets_target else "orange"
        st.metric("Temps Moyen", f"{avg_time:.3f}s", 
                 delta=f"{avg_time - 3.0:.3f}s vs objectif",
                 delta_color="inverse",
                 help="Objectif: < 3 secondes")
        
        if meets_target:
            st.success("✅ Objectif atteint (< 3s)")
        else:
            st.warning("⚠️ Performance à améliorer")
        
        st.metric("Écart-type", f"{std_time:.3f}s")
    
    with col2:
        # Graphique de distribution
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=[min_time, avg_time, max_time],
            name="Temps de réponse",
            marker_color='#FF6B6B'
        ))
        fig.add_hline(y=3.0, line_dash="dash", line_color="red", 
                     annotation_text="Objectif: 3s")
        fig.update_layout(
            title="Distribution du Temps de Réponse",
            yaxis_title="Temps (secondes)",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)


def evaluation_component(backend_url: str, dataset_id: str = None):
    """
    Composant principal pour l'évaluation complète.
    
    Args:
        backend_url: URL du backend API
        dataset_id: ID du dataset à évaluer
    """
    st.header("📈 Évaluation & Reproductibilité")
    st.markdown("Analyse de la qualité, diversité, couverture et stabilité des motifs extraits.")
    
    if not dataset_id:
        st.warning("⚠️ Veuillez sélectionner un dataset pour lancer l'évaluation")
        return
    
    # Bouton pour lancer l'évaluation
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.button("🚀 Lancer l'Évaluation Complète", type="primary", use_container_width=True):
            with st.spinner("Évaluation en cours..."):
                try:
                    # Appel API
                    response = requests.post(
                        f"{backend_url}/api/evaluate",
                        json={
                            "dataset_id": dataset_id,
                            "include_stability": True,
                            "include_performance": False,
                            "stability_runs": 5
                        }
                    )
                    
                    if response.status_code == 200:
                        evaluation_data = response.json()
                        st.session_state['evaluation_data'] = evaluation_data
                        st.success("✅ Évaluation terminée !")
                    else:
                        st.error(f"❌ Erreur: {response.status_code} - {response.text}")
                        return
                        
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'évaluation: {str(e)}")
                    return
    
    # Afficher les résultats si disponibles
    if 'evaluation_data' in st.session_state:
        data = st.session_state['evaluation_data']
        
        st.markdown("---")
        
        # Résumé global
        if 'summary' in data:
            summary = data['summary']
            st.subheader("📋 Résumé Global")
            
            quality = summary.get('overall_quality', 'N/A')
            quality_color = {
                'Excellente': '🟢',
                'Bonne': '🟡',
                'Moyenne': '🟠',
                'Faible': '🔴'
            }.get(quality, '⚪')
            
            st.markdown(f"### {quality_color} Qualité Globale: **{quality}**")
            
            if 'recommendations' in summary and summary['recommendations']:
                st.markdown("**Recommandations:**")
                for rec in summary['recommendations']:
                    st.markdown(f"- {rec}")
            
            st.markdown("---")
        
        # Tabs pour organiser les métriques
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Diversité",
            "🎯 Couverture",
            "👍 Acceptation",
            "🔄 Stabilité"
        ])
        
        with tab1:
            if 'diversity' in data:
                plot_diversity_metrics(data['diversity'])
        
        with tab2:
            if 'coverage' in data:
                plot_coverage_metrics(data['coverage'])
        
        with tab3:
            if 'acceptance' in data:
                plot_acceptance_metrics(data['acceptance'])
        
        with tab4:
            if 'stability' in data:
                plot_stability_metrics(data['stability'])
        
        # Option d'export
        st.markdown("---")
        st.subheader("💾 Export des Résultats")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Télécharger Rapport JSON"):
                import json
                st.download_button(
                    label="Télécharger",
                    data=json.dumps(data, indent=2),
                    file_name=f"evaluation_{dataset_id}_{data.get('timestamp', 0)}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📊 Télécharger Résumé CSV"):
                # Créer un DataFrame avec les principales métriques
                summary_data = {
                    "Métrique": [],
                    "Valeur": []
                }
                
                if 'diversity' in data and 'jaccard' in data['diversity']:
                    summary_data["Métrique"].append("Diversité (Jaccard)")
                    summary_data["Valeur"].append(data['diversity']['jaccard'].get('average_diversity', 0))
                
                if 'coverage' in data:
                    summary_data["Métrique"].append("Couverture (%)")
                    summary_data["Valeur"].append(data['coverage'].get('coverage_rate', 0))
                
                if 'acceptance' in data:
                    summary_data["Métrique"].append("Acceptation (%)")
                    summary_data["Valeur"].append(data['acceptance'].get('acceptance_rate', 0))
                
                df_summary = pd.DataFrame(summary_data)
                csv = df_summary.to_csv(index=False)
                
                st.download_button(
                    label="Télécharger",
                    data=csv,
                    file_name=f"summary_{dataset_id}.csv",
                    mime="text/csv"
                )
