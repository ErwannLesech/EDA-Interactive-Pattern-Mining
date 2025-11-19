import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict

def display_evaluation_metrics(evaluation_data: Dict):
    """
    Affiche les métriques d'évaluation avec visualisations.
    
    Args:
        evaluation_data: Dictionnaire contenant les résultats d'évaluation
    """
    
    if not evaluation_data:
        st.warning("Aucune donnée d'évaluation disponible")
        return
    
    eval_results = evaluation_data.get("evaluation", {})
    metadata = evaluation_data.get("metadata", {})
    
    # En-tête avec métadonnées
    st.subheader("📊 Métriques d'Évaluation")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Motifs totaux", metadata.get("total_patterns", 0))
    with col2:
        st.metric("Motifs échantillonnés", metadata.get("sampled_patterns", 0))
    with col3:
        st.metric("Feedbacks reçus", metadata.get("total_feedbacks", 0))
    
    st.markdown("---")
    
    # Score global
    overall_score = eval_results.get("overall_score", 0)
    st.subheader(f"🎯 Score Global: {overall_score:.2%}")
    
    # Jauge pour le score global
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=overall_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Score Global (%)"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "lightgray"},
                {'range': [33, 66], 'color': "gray"},
                {'range': [66, 100], 'color': "lightblue"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    st.markdown("---")
    
    # Créer des onglets pour les différentes métriques
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Taux d'Acceptation",
        "🌈 Diversité",
        "📦 Couverture",
        "🔒 Stabilité",
        "⏱️ Performance"
    ])
    
    # Onglet 1: Taux d'Acceptation
    with tab1:
        acceptance = eval_results.get("acceptance", {})
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.metric(
                "Taux d'Acceptation",
                f"{acceptance.get('acceptance_rate', 0):.2%}",
                help="Proportion de motifs aimés (likes) par rapport au total des feedbacks"
            )
            
            # Graphique en barres
            likes = acceptance.get("likes", 0)
            dislikes = acceptance.get("dislikes", 0)
            neutral = acceptance.get("neutral", 0)
            
            fig_feedback = go.Figure(data=[
                go.Bar(name='👍 Likes', x=['Feedback'], y=[likes], marker_color='green'),
                go.Bar(name='👎 Dislikes', x=['Feedback'], y=[dislikes], marker_color='red'),
                go.Bar(name='⚪ Neutral', x=['Feedback'], y=[neutral], marker_color='gray')
            ])
            fig_feedback.update_layout(
                title="Distribution des Feedbacks",
                barmode='group',
                height=300
            )
            st.plotly_chart(fig_feedback, use_container_width=True)
        
        with col2:
            st.metric("Total Feedbacks", acceptance.get("total_feedbacks", 0))
            st.metric("👍 Likes", likes)
            st.metric("👎 Dislikes", dislikes)
            st.metric("⚪ Neutral", neutral)
        
        st.info("""
        **Interprétation:** Un taux d'acceptation élevé (>70%) indique que les motifs échantillonnés 
        sont pertinents pour l'utilisateur. Un taux faible suggère d'ajuster les poids d'échantillonnage.
        """)
    
    # Onglet 2: Diversité
    with tab2:
        diversity = eval_results.get("diversity", {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Score de Diversité",
                f"{diversity.get('diversity_score', 0):.3f}",
                help="Distance de Jaccard moyenne entre motifs (0=identiques, 1=totalement différents)"
            )
        with col2:
            st.metric(
                "Items Uniques",
                diversity.get("unique_items_count", 0),
                help="Nombre d'items différents couverts par les motifs"
            )
        with col3:
            st.metric(
                "Longueur Moyenne",
                f"{diversity.get('average_pattern_length', 0):.1f}",
                help="Taille moyenne des motifs échantillonnés"
            )
        
        # Jauge de diversité
        fig_div = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=diversity.get('diversity_score', 0) * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Diversité (%)"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "purple"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgray"},
                    {'range': [30, 70], 'color': "lavender"},
                    {'range': [70, 100], 'color': "plum"}
                ],
            }
        ))
        fig_div.update_layout(height=300)
        st.plotly_chart(fig_div, use_container_width=True)
        
        st.info("""
        **Interprétation:** Une diversité élevée (>0.7) signifie que les motifs sont très différents 
        les uns des autres, réduisant la redondance. Une faible diversité (<0.3) indique des motifs similaires.
        """)
    
    # Onglet 3: Couverture
    with tab3:
        coverage = eval_results.get("coverage", {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Couverture Motifs",
                f"{coverage.get('pattern_coverage', 0):.2%}",
                help="Proportion de motifs échantillonnés par rapport au total"
            )
        with col2:
            st.metric(
                "Couverture Items",
                f"{coverage.get('item_coverage', 0):.2%}",
                help="Proportion d'items uniques couverts"
            )
        with col3:
            st.metric(
                "Couverture Support",
                f"{coverage.get('support_coverage', 0):.2%}",
                help="Somme des supports échantillonnés / total"
            )
        
        # Graphique de couverture
        coverage_data = pd.DataFrame({
            'Métrique': ['Motifs', 'Items', 'Support'],
            'Couverture': [
                coverage.get('pattern_coverage', 0) * 100,
                coverage.get('item_coverage', 0) * 100,
                coverage.get('support_coverage', 0) * 100
            ]
        })
        
        fig_cov = px.bar(
            coverage_data,
            x='Métrique',
            y='Couverture',
            title="Couverture par Type (%)",
            color='Couverture',
            color_continuous_scale='Blues'
        )
        fig_cov.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_cov, use_container_width=True)
        
        st.info("""
        **Interprétation:** Une bonne couverture (>50%) assure que l'échantillon représente 
        bien le pool complet de motifs. Une couverture faible peut nécessiter d'augmenter k.
        """)
    
    # Onglet 4: Stabilité
    with tab4:
        stability = eval_results.get("stability", {})
        
        if stability.get("stability_score") is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Score de Stabilité",
                    f"{stability.get('stability_score', 0):.3f}",
                    help="Similarité moyenne entre échantillons avec différentes seeds (0=instable, 1=stable)"
                )
            with col2:
                st.metric(
                    "Similarité Moyenne",
                    f"{stability.get('mean_similarity', 0):.3f}"
                )
            with col3:
                st.metric(
                    "Écart-Type",
                    f"{stability.get('std_similarity', 0):.3f}",
                    help="Variabilité des similarités"
                )
            
            # Histogramme des similarités
            similarities = stability.get("jaccard_similarities", [])
            if similarities:
                fig_stab = go.Figure(data=[go.Histogram(
                    x=similarities,
                    nbinsx=20,
                    marker_color='teal'
                )])
                fig_stab.update_layout(
                    title="Distribution des Similarités de Jaccard",
                    xaxis_title="Similarité",
                    yaxis_title="Fréquence",
                    height=300
                )
                st.plotly_chart(fig_stab, use_container_width=True)
            
            st.info("""
            **Interprétation:** Une stabilité élevée (>0.7) indique que l'algorithme produit 
            des résultats reproductibles. Une faible stabilité (<0.3) suggère une forte dépendance à la seed.
            """)
        else:
            st.warning("Données de stabilité non disponibles")
    
    # Onglet 5: Performance
    with tab5:
        response_time = eval_results.get("response_time", {})
        
        if response_time.get("mean_time") is not None:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Temps Moyen",
                    f"{response_time.get('mean_time', 0):.3f}s"
                )
            with col2:
                st.metric(
                    "Temps Min",
                    f"{response_time.get('min_time', 0):.3f}s"
                )
            with col3:
                st.metric(
                    "Temps Max",
                    f"{response_time.get('max_time', 0):.3f}s"
                )
            with col4:
                st.metric(
                    "Écart-Type",
                    f"{response_time.get('std_time', 0):.3f}s"
                )
            
            # Indicateur de performance
            mean_time = response_time.get('mean_time', 0)
            if mean_time < 2:
                perf_status = "🟢 Excellent"
                perf_color = "green"
            elif mean_time < 5:
                perf_status = "🟡 Bon"
                perf_color = "orange"
            else:
                perf_status = "🔴 À améliorer"
                perf_color = "red"
            
            st.markdown(f"### Performance: {perf_status}")
            
            # Barre de progression
            fig_perf = go.Figure(go.Indicator(
                mode="number+delta",
                value=mean_time,
                title={'text': "Temps de Réponse Moyen (s)"},
                delta={'reference': 2, 'relative': False},
                domain={'x': [0, 1], 'y': [0, 1]}
            ))
            fig_perf.update_layout(height=200)
            st.plotly_chart(fig_perf, use_container_width=True)
            
            st.info("""
            **Objectif:** < 2-3 secondes pour une expérience interactive optimale. 
            Un temps supérieur peut nécessiter une optimisation de l'algorithme ou une réduction de k.
            """)
        else:
            st.warning("Données de performance non disponibles")


def display_evaluation_summary(evaluation_data: Dict):
    """
    Affiche un résumé compact des métriques d'évaluation.
    
    Args:
        evaluation_data: Dictionnaire contenant les résultats d'évaluation
    """
    if not evaluation_data:
        return
    
    eval_results = evaluation_data.get("evaluation", {})
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        acceptance_rate = eval_results.get("acceptance", {}).get("acceptance_rate", 0)
        st.metric("Acceptation", f"{acceptance_rate:.1%}")
    
    with col2:
        diversity = eval_results.get("diversity", {}).get("diversity_score", 0)
        st.metric("Diversité", f"{diversity:.2f}")
    
    with col3:
        coverage = eval_results.get("coverage", {}).get("pattern_coverage", 0)
        st.metric("Couverture", f"{coverage:.1%}")
    
    with col4:
        stability = eval_results.get("stability", {}).get("stability_score", 0)
        if stability:
            st.metric("Stabilité", f"{stability:.2f}")
        else:
            st.metric("Stabilité", "N/A")
    
    with col5:
        overall = eval_results.get("overall_score", 0)
        st.metric("Score Global", f"{overall:.1%}")
