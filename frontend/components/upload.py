import streamlit as st
import requests
import pandas as pd
import os
from io import BytesIO


def _detect_dataset_type_local(df: pd.DataFrame):
    """
    Petite heuristique locale pour détecter le type du dataset.
    Retourne l'un des: 'transactional', 'inversed', 'matrix', 'sequential'
    et des colonnes candidates (transaction_col, items_col, sequence_col)
    """
    cols = df.columns.tolist()
    cols_lower = [c.lower() for c in cols]

    # inversed: header indicates transaction_id/items or similar (heuristique)
    if len(cols) > 1 and any(c.strip().lower() in ['transaction_id', 'items'] for c in cols_lower[:2]):
        # check first row values for hints
        try:
            first_cell = str(df.iloc[0, 0]).strip().lower() if not df.empty else ''
            if first_cell and 'items' in first_cell:
                return 'inversed', None, None, None
        except Exception:
            pass

    # matrix: plusieurs colonnes indicatrices 0/1
    if len(cols) > 2:
        try:
            numeric_cols = [c for c in cols[1:] if pd.api.types.is_numeric_dtype(df[c])]
            if len(numeric_cols) >= 2:
                # check a few rows for binary values
                sample = df[numeric_cols].head(10)
                if ((sample.isin([0, 1, 0.0, 1.0])).all().all()):
                    return 'matrix', cols[0], None, None
        except Exception:
            pass

    # sequential: colonne 'position' ou 'sequence' présente
    if any(kw in col.lower() for col in cols for kw in ['position', 'sequence', 'step', 'order']):
        seq_col = next((c for c in cols if any(kw in c.lower() for kw in ['sequence', 'trans', 'session'])), cols[0])
        items_col = next((c for c in cols if 'item' in c.lower()), cols[-1] if len(cols) > 1 else None)
        return 'sequential', seq_col, items_col, None

    # default transactional
    transaction_col = next((c for c in cols if any(kw in c.lower() for kw in ['transaction', 'trans', 'tid'])), cols[0])
    items_col = next((c for c in cols if c != transaction_col), None)
    return 'transactional', transaction_col, items_col, None


def upload_component(backend_url: str):
    """Composant d'upload de fichier simplifié et fiable"""
    st.header("📤 Upload de Dataset")

    uploaded_file = st.file_uploader(
        "Choisissez un fichier CSV ou Excel",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is None:
        return

    # Lire le contenu une seule fois et le réutiliser
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name

    st.success(f"Fichier chargé : {filename}")

    # Nom du dataset pré-rempli
    default_name = os.path.splitext(filename)[0]
    dataset_name = st.text_input(
        "Nom du dataset",
        value=default_name,
        help="Donnez un nom personnalisé à ce dataset (par défaut : nom du fichier)"
    )

    # --- Détection du séparateur via le backend pour les CSV ---
    detected_separator = None
    if filename.lower().endswith('.csv'):
        try:
            # Envoyer les bytes au backend pour détection
            resp = requests.post(
                f"{backend_url}/api/detect-separator",
                files={"file": (filename, file_bytes, "text/csv")},
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                detected_separator = data.get('separator')
                st.session_state['detected_separator'] = detected_separator
            else:
                st.warning("⚠️ La détection du séparateur a échoué côté serveur.")
        except Exception as e:
            st.warning(f"⚠️ Impossible de contacter le serveur pour la détection du séparateur: {e}")

    # --- Détection locale du type de dataset pour pré-remplir (heuristique légère) ---
    detected_type = None
    try:
        # Lire un échantillon en DataFrame - privilégier CSV avec sep=None pour flexibilité
        if filename.lower().endswith('.csv'):
            df = pd.read_csv(BytesIO(file_bytes), sep=None, engine='python', nrows=200)
        elif filename.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(BytesIO(file_bytes), nrows=200)
        else:
            df = pd.read_csv(BytesIO(file_bytes), sep=None, engine='python', nrows=200)

        detected_type, tcol, icol, scol = _detect_dataset_type_local(df)
        st.session_state['detected_dataset_type'] = detected_type
    except Exception as e:
        st.warning(f"⚠️ Impossible d'analyser localement le type de fichier: {e}")

    # --- UI pour le séparateur (si CSV) ---
    separator_option = None
    if filename.lower().endswith('.csv'):
        st.subheader("Options CSV")
        sep_names = {
            ',': 'Virgule (,)',
            ';': 'Point-virgule (;)',
            '\t': 'Tabulation (\\t)',
            '|': 'Pipe (|)',
            ' ': 'Espace ( )'
        }
        seps = [',', ';', '\t', '|', ' ']
        default_index = 0
        if detected_separator in seps:
            default_index = seps.index(detected_separator)

        separator_option = st.selectbox(
            "Séparateur (détecté automatiquement)",
            seps,
            index=default_index,
            format_func=lambda x: sep_names.get(x, x),
            help="Le séparateur est détecté automatiquement par défaut; modifiez-le si nécessaire"
        )

    # --- UI pour le type de dataset ---
    st.subheader("Type de dataset")
    type_map = {
        'transactional': 'Transactionnel',
        'inversed': 'Transactionnel inversé',
        'sequential': 'Séquentiel',
        'matrix': 'Matricielle'
    }
    types = list(type_map.keys())
    default_type_index = 0
    if detected_type in types:
        default_type_index = types.index(detected_type)

    selected_type = st.selectbox(
        "Type détecté (modifiable)",
        types,
        index=default_type_index,
        format_func=lambda x: type_map.get(x, x),
        help="Le type est détecté automatiquement (heuristique), modifiez-le si besoin"
    )

    st.markdown("---")

    # Bouton d'envoi
    if st.button("Valider et traiter", type="primary"):
        if not dataset_name or not dataset_name.strip():
            st.error("⚠️ Le nom du dataset ne peut pas être vide")
            return

        with st.spinner("Envoi au serveur et traitement..."):
            try:
                form_data = {
                    'dataset_name': dataset_name.strip(),
                    'dataset_type': selected_type,
                    'auto_detect': 'false'
                }
                if separator_option:
                    form_data['separator'] = separator_option

                resp = requests.post(
                    f"{backend_url}/api/upload",
                    files={"file": (filename, file_bytes, 'application/octet-stream')},
                    data=form_data,
                    timeout=30
                )

                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state['active_dataset_id'] = data['dataset_id']
                    st.session_state['active_dataset_name'] = dataset_name.strip()
                    st.session_state['active_dataset_type'] = data.get('dataset_type')
                    st.session_state['is_sequential'] = data.get('dataset_type') == 'sequential'

                    st.success("✅ Fichier traité avec succès!")
                    st.info(f"💡 Ce dataset est maintenant actif pour l'analyse (Type: {type_map.get(data.get('dataset_type'), data.get('dataset_type'))})")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Nombre de lignes", data.get('rows'))
                    with col2:
                        st.metric("Nombre de colonnes", len(data.get('columns', [])))

                    st.subheader("Aperçu des données")
                    preview_df = pd.DataFrame(data.get('preview', []))
                    st.dataframe(preview_df)
                else:
                    # essayer d'afficher le message d'erreur si présent
                    try:
                        err = resp.json()
                    except Exception:
                        err = resp.text
                    st.error(f"❌ Erreur lors de l'upload : {err}")

            except Exception as e:
                st.error(f"❌ Erreur de connexion : {str(e)}")
