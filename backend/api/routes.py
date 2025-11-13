from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Body
from api.models import (
    UploadResponse, 
    DatasetType,
    FeedbackData,
    EvaluationRequest,
    EvaluationResponse,
    MiningRequest,
    MiningResponse,
    PatternItem,
    RuleItem
)
from typing import List, Optional
import pandas as pd
from io import BytesIO
import logging
import time

from utils.data_processing import (
    read_file_to_dataframe,
    detect_dataset_type,
    normalize_to_transactional_format,
    binarise_transactions
)
from utils.storage import DatasetStorage, STORAGE_DIR
from core.evaluation import PatternEvaluator
from core.pattern_mining import PatternMiner

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_type: Optional[str] = Form(None),
    transaction_col: Optional[str] = Form(None),
    items_col: Optional[str] = Form(None),
    sequence_col: Optional[str] = Form(None),
    auto_detect: bool = Form(True)
):
    """
    Upload et traitement d'un dataset transactionnel ou séquentiel.
    
    Formats supportés: CSV, Excel (.xlsx, .xls), TSV, JSON
    
    Args:
        file: Fichier à uploader
        dataset_type: Type de dataset ('transactional' ou 'sequential'), détecté automatiquement si None
        transaction_col: Nom de la colonne contenant les IDs de transaction
        items_col: Nom de la colonne contenant les items
        sequence_col: Nom de la colonne de séquence (pour les données séquentielles)
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
        
        # Conversion en DataFrame
        df_original = read_file_to_dataframe(contents, file.filename)
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
        
        # Sauvegarde temporaire
        dataset_id = DatasetStorage.save_dataset(df_normalized, file.filename)
        
        # Préparation de la réponse
        preview_data = df_normalized.head(10).to_dict(orient="records")
        
        response = UploadResponse(
            dataset_id=dataset_id,
            filename=file.filename,
            dataset_type=DatasetType(dataset_type),
            rows=len(df_normalized),
            columns=df_normalized.columns.tolist(),
            preview=preview_data,
            message=f"Dataset {dataset_type} uploadé et normalisé avec succès. "
                   f"{len(df_normalized)} transactions prêtes pour le mining."
        )
        
        logger.info(f"Upload réussi - dataset_id: {dataset_id}, type: {dataset_type}, "
                   f"transactions: {len(df_normalized)}")
        
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
        
        return {
            "dataset_id": dataset_id,
            "rows": len(df),
            "columns": df.columns.tolist(),
            "preview": df.head(10).to_dict(orient="records")
        }
        
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


# ========== Routes d'évaluation (Partie 4) ==========

@router.post("/evaluate")
async def evaluate_patterns(request: EvaluationRequest):
    """
    Évalue la qualité des motifs extraits selon plusieurs métriques :
    - Taux d'acceptation (via feedback)
    - Diversité (Jaccard, Cosine, Hamming)
    - Couverture des transactions
    - Stabilité (optionnel)
    - Performance (optionnel)
    """
    try:
        # Charger le dataset
        df_transactions = DatasetStorage.load_dataset(request.dataset_id)
        if df_transactions is None:
            raise HTTPException(
                status_code=404,
                detail=f"Dataset {request.dataset_id} non trouvé"
            )
        
        # Charger les motifs (supposer qu'ils sont stockés avec suffix _patterns)
        patterns_id = f"{request.dataset_id}_patterns"
        df_patterns = DatasetStorage.load_dataset(patterns_id)
        
        if df_patterns is None:
            raise HTTPException(
                status_code=404,
                detail=f"Motifs non trouvés pour le dataset {request.dataset_id}. Veuillez d'abord extraire les motifs."
            )
        
        # Créer l'évaluateur
        evaluator = PatternEvaluator(
            patterns=df_patterns,
            transactions=df_transactions
        )
        
        # Ajouter les feedbacks si fournis
        if request.feedback_data:
            for feedback in request.feedback_data:
                evaluator.add_feedback(
                    pattern_id=feedback.get('pattern_id', 0),
                    rating=feedback.get('rating', 0),
                    comment=feedback.get('comment')
                )
        
        # Évaluation complète
        results = evaluator.comprehensive_evaluation(
            feedback_data=request.feedback_data
        )
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation : {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'évaluation : {str(e)}"
        )


@router.post("/evaluate/diversity")
async def evaluate_diversity(
    dataset_id: str = Body(...),
    method: str = Body("jaccard")
):
    """
    Calcule uniquement la diversité des motifs.
    
    Args:
        dataset_id: ID du dataset
        method: Méthode de calcul ('jaccard', 'cosine', 'hamming')
    """
    try:
        patterns_id = f"{dataset_id}_patterns"
        df_patterns = DatasetStorage.load_dataset(patterns_id)
        
        if df_patterns is None:
            raise HTTPException(
                status_code=404,
                detail=f"Motifs non trouvés pour le dataset {dataset_id}"
            )
        
        evaluator = PatternEvaluator(patterns=df_patterns)
        diversity_metrics = evaluator.calculate_diversity(method=method)
        
        return diversity_metrics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors du calcul de diversité : {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur : {str(e)}"
        )


@router.post("/evaluate/coverage")
async def evaluate_coverage(dataset_id: str = Body(...)):
    """
    Calcule la couverture des transactions par les motifs.
    """
    try:
        # Charger transactions et motifs
        df_transactions = DatasetStorage.load_dataset(dataset_id)
        patterns_id = f"{dataset_id}_patterns"
        df_patterns = DatasetStorage.load_dataset(patterns_id)
        
        if df_transactions is None or df_patterns is None:
            raise HTTPException(
                status_code=404,
                detail="Dataset ou motifs non trouvés"
            )
        
        evaluator = PatternEvaluator(
            patterns=df_patterns,
            transactions=df_transactions
        )
        coverage_metrics = evaluator.calculate_coverage()
        
        return coverage_metrics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors du calcul de couverture : {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur : {str(e)}"
        )


@router.post("/feedback")
async def submit_feedback(feedback: FeedbackData):
    """
    Soumet un feedback utilisateur pour un motif.
    Le feedback est stocké pour calculer le taux d'acceptation.
    """
    try:
        # Stocker le feedback (peut être étendu avec une vraie base de données)
        feedback_id = DatasetStorage.save_feedback(feedback.dict())
        
        return {
            "message": "Feedback enregistré avec succès",
            "feedback_id": feedback_id,
            "pattern_id": feedback.pattern_id,
            "rating": feedback.rating
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de l'enregistrement du feedback : {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur : {str(e)}"
        )


# ========== Routes pour l'extraction de motifs ==========

@router.post("/mine", response_model=MiningResponse)
async def mine_patterns(request: MiningRequest):
    """
    Extrait les motifs fréquents et les règles d'association.
    
    Args:
        request: MiningRequest contenant dataset_id, min_support, min_confidence
        
    Returns:
        MiningResponse avec les motifs et règles extraits
    """
    try:
        start_time = time.time()
        
        # Charger le dataset
        df_transactions = DatasetStorage.load_dataset(request.dataset_id)
        if df_transactions is None:
            raise HTTPException(
                status_code=404,
                detail=f"Dataset {request.dataset_id} non trouvé"
            )
        
        logger.info(f"Mining patterns pour dataset {request.dataset_id}: support={request.min_support}, confidence={request.min_confidence}")
        
        # Convertir le dataset en liste de transactions
        # Le dataset normalisé a les colonnes: transaction_id, items
        transactions_list = []
        for _, row in df_transactions.iterrows():
            items_str = row.get('items', '')
            if pd.notna(items_str) and items_str:
                items = [item.strip() for item in str(items_str).split(',')]
                transactions_list.append(items)
        
        logger.info(f"Transactions parsées: {len(transactions_list)}")
        
        # Binariser les transactions pour mlxtend
        df_binary = binarise_transactions(transactions_list)
        logger.info(f"Dataset binarisé: {df_binary.shape}")
        
        # Créer le miner et extraire les motifs
        miner = PatternMiner(df_binary)
        patterns, rules = miner.mine_patterns(
            min_support=request.min_support,
            min_confidence=request.min_confidence
        )
        
        # Sauvegarder les motifs pour évaluation ultérieure
        patterns_id = f"{request.dataset_id}_patterns"
        patterns_filepath = STORAGE_DIR / f"{patterns_id}.csv"
        patterns.to_csv(patterns_filepath, index=False)
        logger.info(f"Motifs sauvegardés: {patterns_id}")
        
        # Sauvegarder les règles
        rules_id = f"{request.dataset_id}_rules"
        rules_filepath = STORAGE_DIR / f"{rules_id}.csv"
        rules.to_csv(rules_filepath, index=False)
        logger.info(f"Règles sauvegardées: {rules_id}")
        
        # Préparer la preview des motifs (top 10)
        patterns_preview = []
        for _, row in patterns.head(10).iterrows():
            patterns_preview.append(PatternItem(
                itemset=list(row['itemsets']),
                support=float(row['support']),
                length=int(row['length']),
                coverage=float(row['coverage'])
            ))
        
        # Préparer la preview des règles (top 10)
        rules_preview = []
        for _, row in rules.head(10).iterrows():
            rules_preview.append(RuleItem(
                antecedents=list(row['antecedents']),
                consequents=list(row['consequents']),
                support=float(row['support']),
                confidence=float(row['confidence']),
                lift=float(row['lift']),
                antecedent_support=float(row['antecedent support']),
                consequent_support=float(row['consequent support'])
            ))
        
        computation_time = time.time() - start_time
        
        response = MiningResponse(
            dataset_id=request.dataset_id,
            num_patterns=len(patterns),
            num_rules=len(rules),
            patterns_preview=patterns_preview,
            rules_preview=rules_preview,
            computation_time=computation_time,
            message=f"Extraction réussie : {len(patterns)} motifs et {len(rules)} règles en {computation_time:.2f}s"
        )
        
        logger.info(f"Mining terminé: {len(patterns)} motifs, {len(rules)} règles, {computation_time:.2f}s")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction : {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'extraction : {str(e)}"
        )


@router.get("/patterns/{dataset_id}")
async def get_patterns(dataset_id: str, limit: Optional[int] = 100):
    """
    Récupère les motifs extraits pour un dataset.
    
    Args:
        dataset_id: ID du dataset
        limit: Nombre maximum de motifs à retourner
        
    Returns:
        Liste des motifs avec leurs métriques
    """
    try:
        patterns_id = f"{dataset_id}_patterns"
        df_patterns = DatasetStorage.load_dataset(patterns_id)
        
        if df_patterns is None:
            raise HTTPException(
                status_code=404,
                detail=f"Motifs non trouvés pour le dataset {dataset_id}. Veuillez d'abord extraire les motifs."
            )
        
        # Limiter le nombre de résultats
        df_patterns = df_patterns.head(limit)
        
        # Convertir en liste de dictionnaires
        patterns_list = []
        for _, row in df_patterns.iterrows():
            patterns_list.append({
                "itemset": list(row['itemsets']),
                "support": float(row['support']),
                "length": int(row['length']),
                "coverage": float(row['coverage'])
            })
        
        return {
            "dataset_id": dataset_id,
            "num_patterns": len(df_patterns),
            "patterns": patterns_list
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des motifs : {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur : {str(e)}"
        )


@router.get("/rules/{dataset_id}")
async def get_rules(dataset_id: str, limit: Optional[int] = 100):
    """
    Récupère les règles d'association pour un dataset.
    
    Args:
        dataset_id: ID du dataset
        limit: Nombre maximum de règles à retourner
        
    Returns:
        Liste des règles avec leurs métriques
    """
    try:
        rules_id = f"{dataset_id}_rules"
        df_rules = DatasetStorage.load_dataset(rules_id)
        
        if df_rules is None:
            raise HTTPException(
                status_code=404,
                detail=f"Règles non trouvées pour le dataset {dataset_id}. Veuillez d'abord extraire les motifs."
            )
        
        # Limiter le nombre de résultats
        df_rules = df_rules.head(limit)
        
        # Convertir en liste de dictionnaires
        rules_list = []
        for _, row in df_rules.iterrows():
            rules_list.append({
                "antecedents": list(row['antecedents']),
                "consequents": list(row['consequents']),
                "support": float(row['support']),
                "confidence": float(row['confidence']),
                "lift": float(row['lift']),
                "antecedent_support": float(row['antecedent support']),
                "consequent_support": float(row['consequent support'])
            })
        
        return {
            "dataset_id": dataset_id,
            "num_rules": len(df_rules),
            "rules": rules_list
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des règles : {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur : {str(e)}"
        )