from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from api.models import (
    UploadResponse, 
    DatasetType
)
from typing import List, Optional
import pandas as pd
from io import BytesIO
import logging
import time
from core.pattern_mining import PatternMiner
from core.sampling import PatternSampler
from fastapi.encoders import jsonable_encoder

from utils.data_processing import (
    read_file_to_dataframe,
    detect_dataset_type,
    normalize_to_transactional_format,
    detect_separator,
    binarise_transactions
)
from utils.storage import DatasetStorage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

router = APIRouter()
pattern_miner = PatternMiner(pd.DataFrame())
pattern_sampler = PatternSampler(pd.DataFrame())


@router.post("/detect-separator")
async def detect_file_separator(file: UploadFile = File(...)):
    """
    Détecte automatiquement le séparateur d'un fichier CSV.
    
    Args:
        file: Fichier CSV à analyser
        
    Returns:
        Séparateur détecté
    """
    try:
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Le nom du fichier est requis"
            )
        
        # Vérifier que c'est un fichier CSV
        if not file.filename.lower().endswith('.csv'):
            return {
                "separator": None,
                "message": "La détection de séparateur n'est disponible que pour les fichiers CSV"
            }
        
        # Lire le contenu
        contents = await file.read()
        
        # Détecter le séparateur
        separator = detect_separator(contents, file.filename)
        
        # Mapper vers un nom lisible
        sep_names = {
            ",": "Virgule (,)",
            ";": "Point-virgule (;)",
            "\t": "Tabulation",
            "|": "Pipe (|)",
            " ": "Espace"
        }
        
        return {
            "separator": separator,
            "separator_name": sep_names.get(separator, separator),
            "message": f"Séparateur détecté: {sep_names.get(separator, separator)}"
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la détection du séparateur: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la détection: {str(e)}"
        )

@router.post("/detect-dataset-type")
async def detect_dataset_type_endpoint(file: UploadFile = File(...), separator: Optional[str] = Form(None)):
    """
    Détecte automatiquement le type de dataset (transactional, sequential, matrix, inversed).
    
    Args:
        file: Fichier à analyser
        separator: Séparateur optionnel pour les fichiers CSV
        
    Returns:
        Type détecté et colonnes candidates
    """
    try:
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Le nom du fichier est requis"
            )
        
        # Lire le contenu
        contents = await file.read()
        
        # Convertir en DataFrame
        try:
            df = read_file_to_dataframe(contents, file.filename, separator)
        except Exception as e:
            logger.error(f"Impossible de lire le fichier pour détection de type: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Impossible de lire le fichier: {str(e)}"
            )
        
        # Détecter le type
        detected_type, trans_col, items_col, seq_col = detect_dataset_type(df)
        
        type_names = {
            "transactional": "Transactionnel",
            "sequential": "Séquentiel",
            "matrix": "Matricielle",
            "inversed": "Transactionnel inversé"
        }
        
        return {
            "detected_type": detected_type,
            "type_name": type_names.get(detected_type, detected_type),
            "transaction_col": trans_col,
            "items_col": items_col,
            "sequence_col": seq_col,
            "message": f"Type détecté: {type_names.get(detected_type, detected_type)}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la détection du type de dataset: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur: {str(e)}"
        )

@router.post("/upload", response_model=UploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_name: Optional[str] = Form(None),
    dataset_type: Optional[str] = Form(None),
    transaction_col: Optional[str] = Form(None),
    items_col: Optional[str] = Form(None),
    sequence_col: Optional[str] = Form(None),
    separator: Optional[str] = Form(None),
    auto_detect: bool = Form(True)
):
    """
    Upload et traitement d'un dataset transactionnel ou séquentiel.
    
    Formats supportés: CSV, Excel (.xlsx, .xls), TSV, JSON
    
    Args:
        file: Fichier à uploader
        dataset_name: Nom personnalisé pour le dataset (utilise le nom du fichier si non fourni)
        dataset_type: Type de dataset ('transactional' ou 'sequential'), détecté automatiquement si None
        transaction_col: Nom de la colonne contenant les IDs de transaction
        items_col: Nom de la colonne contenant les items
        sequence_col: Nom de la colonne de séquence (pour les données séquentielles)
        separator: Séparateur à utiliser pour les fichiers CSV (détection automatique si None)
        auto_detect: Active la détection automatique des colonnes
    
    Returns:
        UploadResponse avec les informations du dataset et son identifiant unique
    """
    try:
        # Validation du nom de fichier
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Le nom du fichier est requis"
            )
        
        # Lecture du fichier
        contents = await file.read()
        logger.info(f"Fichier reçu: {file.filename} ({len(contents)} bytes)")
        
        # Conversion en DataFrame avec séparateur optionnel
        df_original = read_file_to_dataframe(contents, file.filename, separator)
        logger.info(f"DataFrame chargé: {len(df_original)} lignes, {len(df_original.columns)} colonnes")
        
        # Détection automatique si demandé
        if auto_detect:
            detected_type, detected_trans_col, detected_items_col, detected_seq_col = detect_dataset_type(df_original)
            
            # Utiliser les valeurs détectées si non spécifiées
            if not dataset_type:
                dataset_type = detected_type
            if not transaction_col:
                transaction_col = detected_trans_col
            if not items_col:
                items_col = detected_items_col
            if not sequence_col:
                sequence_col = detected_seq_col
        
        # Valeurs par défaut
        if not dataset_type:
            dataset_type = DatasetType.TRANSACTIONAL
        
        # Normalisation au format standard
        df_normalized = normalize_to_transactional_format(
            df_original,
            dataset_type,
            transaction_col,
            items_col,
            sequence_col
        )
        
        if df_normalized.empty:
            raise HTTPException(
                status_code=400,
                detail="Le dataset normalisé est vide. Vérifiez le format des données."
            )
        
        # Split the string in 'items' column into lists
        if dataset_type != "sequential":
            items_list = df_normalized['items'].apply(lambda x: x.split(',') if isinstance(x, str) else x)
            pattern_miner.transactions = binarise_transactions(items_list.to_list())
            pattern_miner.sequential = False
        else:
            pattern_miner.transactions = df_normalized['items'].apply(lambda x: x.split(',') if isinstance(x, str) else x).to_frame()
            pattern_miner.sequential = True
        pattern_miner.frequent_itemsets, pattern_miner.rules = None,None
        # Utiliser le nom personnalisé ou le nom du fichier
        final_dataset_name = dataset_name if dataset_name else file.filename
        
        # Sauvegarde temporaire avec le nom personnalisé et le type
        dataset_id = DatasetStorage.save_dataset(df_normalized, final_dataset_name, dataset_type)
        
        # Préparation de la réponse
        preview_data = df_normalized.head(10).to_dict(orient="records")
        
        response = UploadResponse(
            dataset_id=dataset_id,
            filename=final_dataset_name,
            dataset_type=DatasetType(dataset_type),
            rows=len(df_normalized),
            columns=df_normalized.columns.tolist(),
            preview=preview_data,
            message=f"Dataset {dataset_type} uploadé et normalisé avec succès. "
                   f"{len(df_normalized)} transactions prêtes pour le mining."
        )
        
        logger.info(f"Upload réussi - dataset_id: {dataset_id}, nom: {final_dataset_name}, "
                   f"type: {dataset_type}, transactions: {len(df_normalized)}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de l'upload : {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du traitement du fichier : {str(e)}"
        )


@router.get("/dataset/{dataset_id}")
async def get_dataset(dataset_id: str):
    """Récupère les informations d'un dataset sauvegardé"""
    try:
        df = DatasetStorage.load_dataset(dataset_id)
        
        if df is None:
            raise HTTPException(
                status_code=404,
                detail=f"Dataset {dataset_id} non trouvé"
            )
        
        # Récupérer les métadonnées
        metadata = DatasetStorage.get_dataset_metadata(dataset_id)
        
        response = {
            "dataset_id": dataset_id,
            "rows": len(df),
            "columns": df.columns.tolist(),
            "preview": df.head(10).to_dict(orient="records")
        }
        
        # Ajouter les métadonnées si disponibles
        if metadata:
            response["dataset_type"] = metadata.get("dataset_type", "unknown")
            response["is_sequential"] = metadata.get("is_sequential", False)
            response["name"] = metadata.get("name", dataset_id)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du dataset : {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur : {str(e)}"
        )


@router.delete("/dataset/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Supprime un dataset du stockage temporaire"""
    try:
        success = DatasetStorage.delete_dataset(dataset_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Dataset {dataset_id} non trouvé"
            )
        
        return {
            "message": f"Dataset {dataset_id} supprimé avec succès"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la suppression du dataset : {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur : {str(e)}"
        )


@router.get("/datasets")
async def list_datasets():
    """Liste tous les datasets disponibles"""
    try:
        datasets = DatasetStorage.list_all_datasets()
        
        return {
            "count": len(datasets),
            "datasets": datasets
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de la liste des datasets : {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur : {str(e)}"
        )
 
 # ==================== ENDPOINTS D'EXTRACTION ET RÉ-ÉCHANTILLONNAGE ====================
@router.post("/patterns/mine")
async def mine_patterns(
    min_support: float = Form(0.01),
    support_weight: float = Form(0.4),
    surprise_weight: float = Form(0.4),
    redundancy_weight: float = Form(0.2),
    k: int = Form(50),
    replacement: bool = Form(True)
):
    logger.info(f"Début de l'extraction des motifs avec min_support={min_support}")
    """Extrait les motifs fréquents et les règles d'association à partir d'un dataset donné"""
    try: 
        frequent_itemsets, rules = pattern_miner.mine_patterns(min_support=min_support)
        
        # Initialiser le sampler avec le pool complet
        global pattern_sampler
        pattern_sampler = PatternSampler(frequent_itemsets)
        
        sampled_pattern=pattern_sampler.importance_sampling(
            support_weight=support_weight,
            surprise_weight=surprise_weight,
            redundancy_weight=redundancy_weight,
            k=k,
            replacement=replacement
        )
        
        # Note: pattern_sampler.patterns contient maintenant uniquement les motifs échantillonnés
        # Mais pour construire la réponse, on a besoin des infos complètes.
        # pattern_sampler.patterns a les colonnes support, surprise, redundancy, composite_score calculées
        
        sampled_patterns = []
        # PatternSampler.importance_sampling returns (itemset, index)
        # L'index retourné est relatif à pattern_sampler.patterns (qui est le subset échantillonné)
        # OU est-ce l'index original ?
        # Dans ma modif de sampling.py: 
        # result = [(self.patterns.iloc[i]['itemsets'], i) for i in range(len(self.patterns))]
        # Donc l'index est 0, 1, 2... relatif à self.patterns (le subset)
        
        for itemset, idx in sampled_pattern:
            # idx est l'index dans pattern_sampler.patterns
            row = pattern_sampler.patterns.iloc[idx]
            sampled_patterns.append({
                "index": int(idx),
                "itemset": list(itemset),
                "support": float(row["support"]),
                "length": len(itemset),
                "surprise": float(row.get("surprise", 0)),
                "redundancy": float(row.get("redundancy", 0)),
                "composite_score": float(row.get("composite_score", 0))
            })
            
        return jsonable_encoder({
            "frequent_itemsets": pattern_sampler.patterns.to_dict(orient="records"),
            "sampled_patterns": sampled_patterns,
            "message": f"Rééchantillonnage réussi: {len(sampled_pattern)} motifs échantillonnés.",
            "method": "importance_sampling",
            "k": len(sampled_pattern),
            "total_patterns": len(frequent_itemsets),
            "parameters": {
                "support_weight": support_weight,
                "surprise_weight": surprise_weight,
                "redundancy_weight": redundancy_weight,
                "replacement": replacement,
                "min_support": min_support,
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction des motifs : {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'extraction des motifs : {str(e)}"
        )
@router.post("/patterns/resample")
async def resample_patterns(
    support_weight: float = Form(0.4),
    surprise_weight: float = Form(0.4),
    redundancy_weight: float = Form(0.2),
    k: int = Form(50),
    replacement: bool = Form(True)
):
    """Rééchantillonne les motifs fréquents en fonction des poids donnés"""
    try:
        sampled_pattern=pattern_sampler.importance_sampling(
            support_weight=support_weight,
            surprise_weight=surprise_weight,
            redundancy_weight=redundancy_weight,
            k=k,
            replacement=replacement
        )
        
        # pattern_sampler.patterns contient maintenant le subset échantillonné
        
        sampled_patterns = []
        # PatternSampler.importance_sampling returns (itemset, index)
        # L'index est relatif à pattern_sampler.patterns (le subset)
        
        for itemset, idx in sampled_pattern:
            row = pattern_sampler.patterns.iloc[idx]
            sampled_patterns.append({
                "index": int(idx),
                "itemset": list(itemset),
                "support": float(row["support"]),
                "length": len(itemset),
                "surprise": float(row.get("surprise", 0)),
                "redundancy": float(row.get("redundancy", 0)),
                "composite_score": float(row.get("composite_score", 0))
            })
            
        return jsonable_encoder({
            "frequent_itemsets": pattern_sampler.patterns.to_dict(orient="records"),
            "sampled_patterns": sampled_patterns,
            "message": f"Rééchantillonnage réussi: {len(sampled_pattern)} motifs échantillonnés.",
            "method": "importance_sampling",
            "k": len(sampled_pattern),
            "total_patterns": len(pattern_sampler.patterns),
            "parameters": {
                "support_weight": support_weight,
                "surprise_weight": surprise_weight,
                "redundancy_weight": redundancy_weight,
                "replacement": replacement,
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors du rééchantillonnage des motifs : {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du rééchantillonnage des motifs : {str(e)}"
        )
 
 # ==================== ENDPOINT DE FEEDBACK ====================
@router.post("/feedback")
async def provide_pattern_feedback(
    index: int = Form(...),
    alpha: float = Form(0.5),
    beta: float = Form(0.3),
    rating: int = Form(0),
    method: str = Form("Importance Sampling")
):
    """Prend en compte le feedback utilisateur pour ajuster les scores des motifs"""
    logger.info(f"Réception du feedback pour le motif index {index} avec rating {rating} et méthode {method}")
    try:
        # Appliquer le feedback pour toutes les méthodes car nous maintenons maintenant
        # les patterns échantillonnés dans pattern_sampler.patterns
        pattern_sampler.user_feedback(index, alpha, beta, rating)
        msg = f"Feedback appliqué pour le motif index {index} ({method})."
            
        return {
            "message": msg
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors du traitement du feedback : {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du traitement du feedback : {str(e)}"
        )


# ==================== ENDPOINT D'ÉVALUATION ====================
@router.get("/patterns/evaluate")
async def evaluate_patterns():
    """
    Évalue la qualité de l'échantillonnage actuel avec plusieurs métriques :
    - Taux d'acceptation (via feedback)
    - Diversité des motifs
    - Couverture du pool de motifs
    - Stabilité (sensibilité à la seed)
    - Temps de réponse
    """
    try:
        from core.evaluation import PatternEvaluator
        
        # Vérifier qu'on a des motifs
        if pattern_sampler.patterns.empty:
            raise HTTPException(
                status_code=400,
                detail="Aucun motif disponible. Lancez d'abord l'extraction."
            )
        
        evaluator = PatternEvaluator()
        
        # Récupérer les motifs échantillonnés (utiliser les derniers échantillonnés)
        # On utilise tous les motifs avec un composite_score si disponible
        if 'composite_score' in pattern_sampler.patterns.columns:
            # Trier par score et prendre le top k
            sampled_df = pattern_sampler.patterns.nlargest(
                min(50, len(pattern_sampler.patterns)), 
                'composite_score'
            )
        else:
            # Prendre un échantillon aléatoire
            sampled_df = pattern_sampler.patterns.head(
                min(50, len(pattern_sampler.patterns))
            )
        
        # Récupérer l'historique des feedbacks
        feedback_list = pattern_sampler.feedback_history
        
        # Fonction d'échantillonnage pour stabilité et temps
        def sample_func(df, support_weight, surprise_weight, redundancy_weight, k, replacement):
            sampler = PatternSampler(df)
            return sampler.importance_sampling(
                support_weight, surprise_weight, redundancy_weight, k, replacement
            )
        
        # Paramètres par défaut pour les tests
        params = {
            "support_weight": 0.4,
            "surprise_weight": 0.4,
            "redundancy_weight": 0.2,
            "k": min(50, len(pattern_sampler.patterns)),
            "replacement": True
        }
        
        # Évaluation complète
        results = evaluator.comprehensive_evaluation(
            sampled_patterns_df=sampled_df,
            all_patterns_df=pattern_sampler.patterns,
            feedback_list=feedback_list,
            sampling_function=sample_func,
            params=params
        )
        
        return {
            "evaluation": results,
            "metadata": {
                "total_patterns": len(pattern_sampler.patterns),
                "sampled_patterns": len(sampled_df),
                "total_feedbacks": len(feedback_list),
                "evaluation_timestamp": time.time()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation : {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'évaluation : {str(e)}"
        )


# ==================== ENDPOINTS D'ÉCHANTILLONNAGE ====================

@router.post("/sample/twostep")
async def sample_patterns_twostep(
    dataset_id: str = Form(...),
    
    k_init: int = Form(...),
    k: int = Form(...),
    support_weight: float = Form(1.0),
    surprise_weight: float = Form(1.0),
    redundancy_weight: float = Form(1.0),
    replacement: bool = Form(True),
    min_support: float = Form(0.01),
):
    """
    Échantillonne k motifs en utilisant TwoStep sampling.
    
    Args:
        dataset_id: ID du dataset
        k: Nombre de motifs à échantillonner
    """
    try:
        
        # Utiliser le miner global
        frequent_itemsets = pattern_miner.twostep_sampling(k_init)
        
        pattern_sampler.patterns = frequent_itemsets
        
        sampled_pattern=pattern_sampler.importance_sampling(
            support_weight=support_weight,
            surprise_weight=surprise_weight,
            redundancy_weight=redundancy_weight,
            k=k,
            replacement=replacement
        )
        
        indexes=[i for _, i in sampled_pattern]
        sampled_patterns = []
        # PatternSampler.importance_sampling returns (itemset, index)
        for itemset, idx in sampled_pattern:
            sampled_patterns.append({
                "index": int(idx),
                "itemset": list(itemset),
                "support": float(frequent_itemsets.iloc[idx]["support"]),
                "length": len(itemset),
                "surprise": float(frequent_itemsets.iloc[idx].get("surprise", 0)),
                "redundancy": float(frequent_itemsets.iloc[idx].get("redundancy", 0)),
                "composite_score": float(frequent_itemsets.iloc[idx].get("composite_score", 0))
            })
        return jsonable_encoder({
            "frequent_itemsets": pattern_sampler.patterns.iloc[indexes].to_dict(orient="records"),
            "sampled_patterns": sampled_patterns,
            "message": f"Rééchantillonnage réussi: {len(sampled_pattern)} motifs échantillonnés.",
            "method": "twostep_sampling",
            "k": min(k, len(sampled_pattern)),
            "total_patterns": len(frequent_itemsets),
            "parameters": {
                "support_weight": support_weight,
                "surprise_weight": surprise_weight,
                "redundancy_weight": redundancy_weight,
                "replacement": replacement,
                "min_support": min_support,
            }
        })
        
    except Exception as e:
        logger.error(f"Erreur lors du TwoStep sampling : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sample/gdps")
async def sample_patterns_gdps(
    dataset_id: str = Form(...),
    k: int = Form(...),
    k_init: int = Form(...),
    min_norm: int = Form(1),
    max_norm: int = Form(10),
    utility: str = Form("freq"),
    support_weight: float = Form(1.0),
    surprise_weight: float = Form(1.0),
    redundancy_weight: float = Form(1.0),
    replacement: bool = Form(True),
    min_support: float = Form(0.01),
):
    """
    Échantillonne k motifs en utilisant GDPS.
    
    Args:
        dataset_id: ID du dataset
        k: Nombre de motifs à échantillonner
        min_norm: Taille minimale des motifs
        max_norm: Taille maximale des motifs
        utility: Type d'utilité (freq, area, decay)
    """
    try:
        # Utiliser le miner global
        frequent_itemsets = pattern_miner.gdps_sampling(
            k=k_init,
            min_norm=min_norm,
            max_norm=max_norm,
            utility=utility
        )
        
        pattern_sampler.patterns = frequent_itemsets
        
        sampled_pattern=pattern_sampler.importance_sampling(
            support_weight=support_weight,
            surprise_weight=surprise_weight,
            redundancy_weight=redundancy_weight,
            k=k,
            replacement=replacement
        )
        
        indexes=[i for _, i in sampled_pattern]
        sampled_patterns = []
        # PatternSampler.importance_sampling returns (itemset, index)
        for itemset, idx in sampled_pattern:
            sampled_patterns.append({
                "index": int(idx),
                "itemset": list(itemset),
                "support": float(frequent_itemsets.iloc[idx]["support"]),
                "length": len(itemset),
                "surprise": float(frequent_itemsets.iloc[idx].get("surprise", 0)),
                "redundancy": float(frequent_itemsets.iloc[idx].get("redundancy", 0)),
                "composite_score": float(frequent_itemsets.iloc[idx].get("composite_score", 0))
            })
        return jsonable_encoder({
            "frequent_itemsets": pattern_sampler.patterns.iloc[indexes].to_dict(orient="records"),
            "sampled_patterns": sampled_patterns,
            "message": f"Rééchantillonnage réussi: {len(sampled_pattern)} motifs échantillonnés.",
            "method": "twostep_sampling",
            "k": min(k, len(sampled_pattern)),
            "total_patterns": len(frequent_itemsets),
            "parameters": {
                "support_weight": support_weight,
                "surprise_weight": surprise_weight,
                "redundancy_weight": redundancy_weight,
                "replacement": replacement,
                "min_support": min_support,
            }
        })
        
    except Exception as e:
        logger.error(f"Erreur lors du GDPS sampling : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
