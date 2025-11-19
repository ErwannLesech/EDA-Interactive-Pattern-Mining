import streamlit as st
import requests
import pandas as pd
import os

def upload_component(backend_url: str):
    """Composant d'upload de fichier"""
    st.header("📤 Upload de Dataset")
    
    uploaded_file = st.file_uploader(
        "Choisissez un fichier CSV ou Excel",
        type=["csv", "xlsx", "xls"]
    )
    
    if uploaded_file is not None:
        # Affichage du nom
        st.success(f"Fichier chargé : {uploaded_file.name}")
        
        # Nom personnalisé du dataset
        default_name = os.path.splitext(uploaded_file.name)[0]
        dataset_name = st.text_input(
            "Nom du dataset",
            value=default_name,
            help="Donnez un nom personnalisé à ce dataset (par défaut : nom du fichier)"
        )
        
        # Options pour fichiers CSV
        separator_option = None
        detected_separator = None
        
        if uploaded_file.name.lower().endswith('.csv'):
            st.subheader("Options CSV")
            
            # Récupérer le séparateur détecté de la session s'il existe
            detected_separator = st.session_state.get('detected_separator', None)
            
            # Détecter automatiquement le séparateur
            col1, col2 = st.columns([3, 1])
            with col1:
                if detected_separator:
                    sep_names = {
                        ",": "Virgule (,)",
                        ";": "Point-virgule (;)",
                        "\t": "Tabulation (\\t)",
                        "|": "Pipe (|)",
                        " ": "Espace ( )"
                    }
                    st.info(f"💡 Séparateur détecté : **{sep_names.get(detected_separator, detected_separator)}**")
                else:
                    st.info("� Cliquez sur 'Détecter' pour analyser le séparateur")
            
            with col2:
                if st.button("🔍 Détecter", help="Analyse le fichier pour détecter automatiquement le séparateur"):
                    with st.spinner("Détection..."):
                        try:
                            # Réinitialiser la position du fichier
                            uploaded_file.seek(0)
                            
                            response = requests.post(
                                f"{backend_url}/api/detect-separator",
                                files={"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                            )
                            
                            if response.status_code == 200:
                                data = response.json()
                                detected_separator = data.get("separator")
                                # Stocker dans session_state
                                st.session_state['detected_separator'] = detected_separator
                                st.rerun()
                            else:
                                st.error("❌ Erreur lors de la détection")
                                
                        except Exception as e:
                            st.error(f"❌ Erreur : {str(e)}")
                        
                        # Réinitialiser la position du fichier pour l'upload
                        uploaded_file.seek(0)
            
            st.markdown("---")
            
            # Choix du séparateur
            sep_mode = st.radio(
                "Mode de séparateur",
                ["Détection automatique", "Choisir manuellement"],
                horizontal=True,
                help="Laissez la détection automatique ou choisissez le séparateur manuellement"
            )
            
            if sep_mode == "Choisir manuellement":
                sep_names = {
                    ",": "Virgule (,)",
                    ";": "Point-virgule (;)",
                    "\t": "Tabulation (\\t)",
                    "|": "Pipe (|)",
                    " ": "Espace ( )"
                }
                # Utiliser le séparateur détecté comme valeur par défaut
                default_index = 0
                seps = [",", ";", "\t", "|", " "]
                detected_separator = st.session_state.get('detected_separator', None)
                if detected_separator and detected_separator in seps:
                    default_index = seps.index(detected_separator)
                
                separator_option = st.selectbox(
                    "Séparateur",
                    seps,
                    index=default_index,
                    format_func=lambda x: sep_names.get(x, x),
                    help="Modifiez le séparateur si la détection n'est pas correcte"
                )
        
        # Envoi au backend
        if st.button("Valider et traiter", type="primary"):
            if not dataset_name or not dataset_name.strip():
                st.error("⚠️ Le nom du dataset ne peut pas être vide")
                return
                
            with st.spinner("Envoi au serveur..."):
                try:
                    # Préparer les données avec le nom personnalisé et le séparateur
                    form_data = {"dataset_name": dataset_name.strip()}
                    if separator_option:
                        form_data["separator"] = separator_option
                    
                    response = requests.post(
                        f"{backend_url}/api/upload",
                        files={"file": (uploaded_file.name, uploaded_file, uploaded_file.type)},
                        data=form_data
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Stocker le dataset_id comme dataset actif
                        st.session_state['active_dataset_id'] = data['dataset_id']
                        st.session_state['active_dataset_name'] = dataset_name.strip()
                        
                        st.success("✅ Fichier traité avec succès!")
                        st.info(f"💡 Ce dataset est maintenant actif pour l'analyse")
                        
                        # Affichage des stats
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Nombre de lignes", data["rows"])
                        with col2:
                            st.metric("Nombre de colonnes", len(data["columns"]))
                        
                        # Aperçu
                        st.subheader("Aperçu des données")
                        preview_df = pd.DataFrame(data["preview"])
                        st.dataframe(preview_df)

                        st.session_state["motifs_df"] = pd.DataFrame()
                        st.session_state["sampled_df"] = pd.DataFrame()
                        st.session_state["extraction_done"] = False
                        
                    else:
                        st.error(f"Erreur : {response.json()}")
                        
                except Exception as e:
                    st.error(f"Erreur de connexion : {str(e)}")
