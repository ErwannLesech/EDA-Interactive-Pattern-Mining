import pandas as pd
from typing import List, Tuple, Optional
from io import BytesIO
import logging
from mlxtend.preprocessing import TransactionEncoder

logger = logging.getLogger(__name__)

def detect_separator(file_content: bytes, filename: str) -> str:
    """
    Détecte automatiquement le séparateur d'un fichier CSV.
    
    Args:
        file_content: Contenu du fichier en bytes
        filename: Nom du fichier
        
    Returns:
        Séparateur détecté (str)
    """
    # Liste des séparateurs à tester par ordre de priorité
    separators = [';', '\t', '|', ',', ' ']
    
    try:
        # Lire les premières lignes pour détecter
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
            try:
                sample_lines = file_content.decode(encoding).split('\n')[:10]
                # Filtrer les lignes vides
                sample_lines = [line for line in sample_lines if line.strip()]
                
                if len(sample_lines) < 2:
                    continue
                
                # Compter les occurrences de chaque séparateur
                sep_scores = {}
                for sep in separators:
                    counts = [line.count(sep) for line in sample_lines]
                    
                    # Vérifier que le séparateur apparaît dans toutes les lignes
                    if not all(c > 0 for c in counts):
                        continue
                    
                    # Vérifier la cohérence : même nombre dans toutes les lignes
                    if len(set(counts)) == 1:
                        # Score = nombre d'occurrences * priorité (plus le séparateur est rare, mieux c'est)
                        # On préfère ; et \t à , car , peut être dans le contenu
                        priority = len(separators) - separators.index(sep)
                        sep_scores[sep] = (counts[0], priority)
                
                if sep_scores:
                    # Retourner le séparateur avec le meilleur score
                    # Priorité : 1. Nombre de colonnes raisonnable (2-20), 2. Priorité du séparateur
                    valid_seps = {sep: score for sep, score in sep_scores.items() if 1 <= score[0] <= 20}
                    
                    if valid_seps:
                        detected_sep = max(valid_seps.items(), key=lambda x: (x[1][1], x[1][0]))[0]
                        logger.info(f"Séparateur détecté: '{detected_sep}' ({valid_seps[detected_sep][0]} colonnes)")
                        return detected_sep
                    
            except Exception as e:
                logger.debug(f"Erreur avec encodage {encoding}: {str(e)}")
                continue
        
        # Par défaut, retourner virgule
        logger.info("Séparateur par défaut: ','")
        return ','
        
    except Exception as e:
        logger.warning(f"Erreur lors de la détection du séparateur: {str(e)}")
        return ','

def read_file_to_dataframe(file_content: bytes, filename: str, separator: Optional[str] = None) -> pd.DataFrame:
    """
    Lit un fichier (CSV, Excel, etc.) et retourne un DataFrame.
    
    Args:
        file_content: Contenu du fichier en bytes
        filename: Nom du fichier pour détecter le format
        separator: Séparateur à utiliser (détection automatique si None)
        
    Returns:
        DataFrame
    """
    file_lower = filename.lower()
    
    try:
        if file_lower.endswith('.csv'):
            # Si pas de séparateur spécifié, le détecter
            if separator is None:
                separator = detect_separator(file_content, filename)
            
            # Essayer différents encodages
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                try:
                    df = pd.read_csv(BytesIO(file_content), encoding=encoding, sep=separator)
                    if not df.empty:
                        logger.info(f"CSV chargé avec l'encodage {encoding} et séparateur '{separator}'")
                        return df
                except:
                    continue
            
            # Dernier essai avec détection automatique
            df = pd.read_csv(BytesIO(file_content), sep=None, engine='python')
            return df
            
        elif file_lower.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(BytesIO(file_content))
            return df
            
        elif file_lower.endswith('.json'):
            df = pd.read_json(BytesIO(file_content))
            return df
            
        elif file_lower.endswith(('.tsv', '.txt')):
            df = pd.read_csv(BytesIO(file_content), sep='\t')
            return df
            
        else:
            raise ValueError(f"Format de fichier non supporté: {filename}")
            
    except Exception as e:
        logger.error(f"Erreur lors de la lecture du fichier {filename}: {str(e)}")
        raise


def detect_dataset_type(df: pd.DataFrame) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """
    Détecte automatiquement le type de dataset (transactionnel, séquentiel, matrix, inversed).
    
    Args:
        df: DataFrame à analyser
        
    Returns:
        Tuple (dataset_type, transaction_col, items_col, sequence_col)
    """
    columns = df.columns.tolist()
    columns_lower = [c.lower() for c in columns]
    
    # Check for inversed transactional (first row contains transaction IDs)
    if len(columns) > 2 and columns[0].strip().lower() in ['transaction_id', 'items']:
        if df.iloc[0, 0] in ['items', 'transaction_id']:
            return "inversed", None, None, None
    
    # Check for matrix format (binary values, multiple item columns)
    if len(columns) > 2:
        non_id_cols = [c for c in columns[1:] if df[c].dtype in ['int64', 'float64']]
        if len(non_id_cols) >= 2 and all(df[col].isin([0, 1, 0.0, 1.0]).all() for col in non_id_cols[:3]):
            return "matrix", columns[0], None, None
    
    # Check for sequential (parentheses in values)
    for col in columns:
        if df[col].astype(str).str.contains(r'\(.*\)', regex=True).any():
            return "sequential", columns[0], columns[1] if len(columns) > 1 else None, None
    
    # Transactional by default
    transaction_col = next((c for c in columns if any(kw in c.lower() for kw in ['transaction', 'trans', 'tid'])), columns[0])
    items_col = next((c for c in columns if c != transaction_col), None)
    
    logger.info(f"Type: transactional, transaction_col: {transaction_col}, items_col: {items_col}")
    return "transactional", transaction_col, items_col, None


def normalize_to_transactional_format(
    df: pd.DataFrame, 
    dataset_type: str,
    transaction_col: Optional[str] = None,
    items_col: Optional[str] = None,
    sequence_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Normalise un DataFrame pour avoir une transaction par ligne au format standard.
    
    Le format de sortie aura toujours les colonnes:
    - transaction_id: Identifiant unique de la transaction
    - items: Liste des items séparés par des virgules (str)
    
    Pour les données séquentielles, l'ordre des items est préservé via la position.
    
    Args:
        df: DataFrame original
        dataset_type: Type de dataset ('transactional', 'sequential', 'matrix', 'inversed')
        transaction_col: Nom de la colonne transaction
        items_col: Nom de la colonne items
        sequence_col: Nom de la colonne séquence (optionnel)
        
    Returns:
        DataFrame normalisé
    """
    logger.info(f"Normalisation dataset type={dataset_type}, trans_col={transaction_col}, items_col={items_col}")
    
    # Handle inversed format: transpose and extract
    if dataset_type == "inversed":
        df_t = df.T
        df_t.columns = df_t.iloc[0]
        df_t = df_t[1:]
        df_t.reset_index(drop=True, inplace=True)
        result = pd.DataFrame({
            'transaction_id': df_t['transaction_id'],
            'items': df_t['items']
        })
        logger.info(f"Dataset inversé normalisé: {len(result)} transactions")
        return result
    
    # Handle matrix format: convert binary to items
    if dataset_type == "matrix":
        # Auto-detect transaction column if not provided
        if not transaction_col:
            transaction_col = df.columns[0]
        
        item_cols = [c for c in df.columns if c != transaction_col]
        items_list = df.apply(lambda row: ','.join([col for col in item_cols if row[col] == 1]), axis=1)
        result = pd.DataFrame({
            'transaction_id': df[transaction_col],
            'items': items_list
        })
        # Remove empty transactions
        result = result[result['items'].str.len() > 0]
        logger.info(f"Dataset matriciel normalisé: {len(result)} transactions")
        return result
    
    # Handle sequential format: group by transaction and preserve order via position
    if dataset_type == "sequential":
        # Trouver les colonnes pertinentes
        if not transaction_col:
            transaction_col = next((c for c in df.columns if any(kw in c.lower() for kw in ['sequence', 'session', 'trans'])), df.columns[0])
        if not items_col:
            items_col = next((c for c in df.columns if 'item' in c.lower()), df.columns[-1])
        
        # Déterminer la colonne de position
        position_col = next((c for c in df.columns if any(kw in c.lower() for kw in ['position', 'step', 'order', 'index'])), None)
        
        # Trier par transaction et position si disponible
        if position_col:
            df_sorted = df.sort_values([transaction_col, position_col])
        else:
            df_sorted = df.sort_values(transaction_col)
        
        # Grouper par transaction et joindre les items dans l'ordre
        grouped = df_sorted.groupby(transaction_col)[items_col].apply(
            lambda x: ','.join(str(item) for item in x if pd.notna(item))
        ).reset_index()
        
        normalized_df = pd.DataFrame({
            'transaction_id': grouped[transaction_col],
            'items': grouped[items_col]
        })
        
        # Remove empty transactions
        normalized_df = normalized_df[normalized_df['items'].str.len() > 0]
        logger.info(f"Dataset séquentiel normalisé: {len(normalized_df)} séquences")
        return normalized_df
    
    # Handle standard transactional format
    # Auto-detect columns if not provided
    if not transaction_col:
        transaction_col = next((c for c in df.columns if any(kw in c.lower() for kw in ['transaction', 'trans', 'tid'])), df.columns[0])
    if not items_col:
        items_col = next((c for c in df.columns if c != transaction_col), df.columns[1] if len(df.columns) > 1 else None)
    
    if items_col is None:
        raise ValueError(f"Impossible de trouver la colonne items. Colonnes disponibles: {df.columns.tolist()}")
    
    items = df[items_col].apply(lambda x: ','.join([i.strip() for i in str(x).split(',')]) if pd.notna(x) else "")
    normalized_df = pd.DataFrame({
        'transaction_id': df[transaction_col],
        'items': items
    })
    
    # Remove empty transactions
    normalized_df = normalized_df[normalized_df['items'].str.len() > 0]
    logger.info(f"Dataset transactionnel normalisé: {len(normalized_df)} transactions")
    
    return normalized_df


def parse_transactions(df: pd.DataFrame, transaction_col: str, items_col: str) -> List[List[str]]:
    """Parse un DataFrame en liste de transactions"""
    transactions = []
    
    for _, row in df.iterrows():
        items = row[items_col]
        
        # Gestion de différents formats
        if isinstance(items, str):
            # Format "item1,item2,item3"
            items_list = [item.strip() for item in items.split(',')]
        elif isinstance(items, list):
            items_list = items
        else:
            continue
        
        transactions.append(items_list)
    
    return transactions

def create_sample_dataset(n_transactions: int = 100) -> pd.DataFrame:
    """Génère un dataset d'exemple"""
    import random
    
    items_pool = ["bread", "milk", "eggs", "butter", "cheese", "yogurt", 
                  "apple", "banana", "chicken", "rice"]
    
    transactions = []
    for i in range(n_transactions):
        n_items = random.randint(2, 5)
        items = random.sample(items_pool, n_items)
        transactions.append({
            "transaction_id": i,
            "items": ",".join(items)
        })
    
    return pd.DataFrame(transactions)

def binarise_transactions(transactions: List[List[str]]) -> pd.DataFrame:
    """Convertit une liste de transactions en format binaire (DataFrame)"""

    encoder = TransactionEncoder()
    binary_array = encoder.fit(transactions).transform(transactions, sparse=False)
    df_binary = pd.DataFrame(binary_array, columns=encoder.columns_)
    
    return df_binary