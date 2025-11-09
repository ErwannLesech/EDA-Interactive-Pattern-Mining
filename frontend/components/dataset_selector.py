"""
Composant pour sélectionner et gérer les datasets importés
"""
import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from typing import Optional

def dataset_selector_component(backend_url: str) -> Optional[str]:
    """
    Affiche la liste des datasets disponibles et permet de sélectionner l'actif.
    
    Args:
        backend_url: URL du backend API
        
    Returns:
        dataset_id du dataset sélectionné ou None
    """
    st.subheader("📚 Datasets Importés")
    
    try:
        # Récupérer la liste des datasets
        response = requests.get(f"{backend_url}/api/datasets", timeout=5)
        
        if response.status_code != 200:
            st.error(f"❌ Erreur lors de la récupération des datasets: {response.status_code}")
            return None
        
        data = response.json()
        datasets = data.get("datasets", [])
        
        if not datasets:
            st.info("📭 Aucun dataset importé. Utilisez l'onglet 'Upload' pour importer des données.")
            return None
        
        st.success(f"✅ {data['count']} dataset(s) disponible(s)")
        
        # Afficher les datasets sous forme de cartes cliquables
        for idx, ds in enumerate(datasets):
            with st.expander(
                f"📊 Dataset {idx + 1} - {ds['rows']} lignes x {ds['columns']} colonnes",
                expanded=(idx == 0)  # Premier dataset ouvert par défaut
            ):
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.metric("📏 Lignes", f"{ds['rows']:,}")
                    st.metric("📐 Colonnes", f"{ds['columns']}")
                
                with col2:
                    size_mb = ds['size_bytes'] / (1024 * 1024)
                    st.metric("💾 Taille", f"{size_mb:.2f} MB")
                    
                    # Calculer l'âge de manière lisible
                    age_hours = ds['age_hours']
                    if age_hours < 1:
                        age_str = f"{int(age_hours * 60)} min"
                    elif age_hours < 24:
                        age_str = f"{int(age_hours)} h"
                    else:
                        age_str = f"{int(age_hours / 24)} j"
                    st.metric("⏱️ Âge", age_str)
                
                with col3:
                    # Boutons d'action
                    if st.button("✅ Sélectionner", key=f"select_{ds['dataset_id']}", use_container_width=True):
                        st.session_state['active_dataset_id'] = ds['dataset_id']
                        st.success("✓ Dataset activé!")
                        st.rerun()
                    
                    if st.button("🗑️ Supprimer", key=f"delete_{ds['dataset_id']}", use_container_width=True):
                        if delete_dataset(backend_url, ds['dataset_id']):
                            st.success("✓ Dataset supprimé!")
                            st.rerun()
                
                # Afficher un aperçu si le dataset est sélectionné
                if st.session_state.get('active_dataset_id') == ds['dataset_id']:
                    st.success("🎯 **Dataset actif pour l'analyse**")
                    
                    if st.checkbox("Voir l'aperçu", key=f"preview_{ds['dataset_id']}"):
                        preview_df = get_dataset_preview(backend_url, ds['dataset_id'])
                        if preview_df is not None:
                            st.dataframe(preview_df, use_container_width=True)
                
                # ID technique
                st.caption("🔧 ID technique:")
                st.code(ds['dataset_id'], language=None)
        
        # Retourner l'ID du dataset actif
        return st.session_state.get('active_dataset_id')
        
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Impossible de se connecter au backend: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Erreur inattendue: {str(e)}")
        return None


def get_dataset_preview(backend_url: str, dataset_id: str) -> Optional[pd.DataFrame]:
    """Récupère un aperçu du dataset"""
    try:
        response = requests.get(f"{backend_url}/api/dataset/{dataset_id}", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            return pd.DataFrame(data['preview'])
        else:
            st.error(f"Erreur lors de la récupération du dataset: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"Erreur: {str(e)}")
        return None


def delete_dataset(backend_url: str, dataset_id: str) -> bool:
    """Supprime un dataset"""
    try:
        response = requests.delete(f"{backend_url}/api/dataset/{dataset_id}", timeout=5)
        
        if response.status_code == 200:
            # Retirer du session state si c'était le dataset actif
            if st.session_state.get('active_dataset_id') == dataset_id:
                del st.session_state['active_dataset_id']
            return True
        else:
            st.error(f"Erreur lors de la suppression: {response.status_code}")
            return False
            
    except Exception as e:
        st.error(f"Erreur: {str(e)}")
        return False


def get_active_dataset_info(backend_url: str) -> Optional[dict]:
    """Récupère les informations du dataset actif"""
    dataset_id = st.session_state.get('active_dataset_id')
    
    if not dataset_id:
        return None
    
    try:
        response = requests.get(f"{backend_url}/api/dataset/{dataset_id}", timeout=5)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
            
    except Exception:
        return None
