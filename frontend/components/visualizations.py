import streamlit as st
import plotly.express as px
import pandas as pd

"""
Ceci n'est qu'un exemple de composant de visualisation.
"""

def visualize_patterns(patterns_df: pd.DataFrame):
    """Visualisation des motifs"""
    
    # Scatter plot support vs lift
    fig = px.scatter(
        patterns_df,
        x="support",
        y="area",
        size="longueur",
        hover_data=["itemsets"],
        title="Support vs Area"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribution des longueurs
    fig2 = px.histogram(
        patterns_df,
        x="longueur",
        title="Distribution des longueurs de motifs"
    )
    st.plotly_chart(fig2, use_container_width=True)
    # Affichage du tableau des motifs
    st.dataframe(patterns_df)