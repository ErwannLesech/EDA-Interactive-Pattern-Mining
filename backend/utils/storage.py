"""
Gestionnaire de stockage pour les datasets importés.
Stockage persistant avec métadonnées et gestion du cycle de vie.
"""
import os
import pandas as pd
import uuid
import time
from pathlib import Path
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

# Répertoire de stockage (persisté via volume Docker)
STORAGE_DIR = Path("/tmp/pattern_mining_datasets")
STORAGE_DIR.mkdir(exist_ok=True, parents=True)

class DatasetStorage:
    """Gestionnaire de stockage pour les datasets"""
    
    @staticmethod
    def save_dataset(df: pd.DataFrame, original_filename: str) -> str:
        """Sauvegarde un DataFrame et retourne un identifiant unique."""
        dataset_id = str(uuid.uuid4())
        filepath = STORAGE_DIR / f"{dataset_id}.csv"
        
        try:
            df.to_csv(filepath, index=False)
            logger.info(f"Dataset sauvegardé: {dataset_id} ({original_filename})")
            return dataset_id
        except Exception as e:
            logger.error(f"Erreur sauvegarde dataset: {str(e)}")
            raise
    
    @staticmethod
    def load_dataset(dataset_id: str) -> Optional[pd.DataFrame]:
        """Charge un DataFrame depuis le stockage."""
        filepath = STORAGE_DIR / f"{dataset_id}.csv"
        
        if not filepath.exists():
            logger.warning(f"Dataset non trouvé: {dataset_id}")
            return None
        
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            logger.error(f"Erreur chargement dataset: {str(e)}")
            return None
    
    @staticmethod
    def delete_dataset(dataset_id: str) -> bool:
        """Supprime un dataset du stockage."""
        filepath = STORAGE_DIR / f"{dataset_id}.csv"
        
        if filepath.exists():
            try:
                os.remove(filepath)
                logger.info(f"Dataset supprimé: {dataset_id}")
                return True
            except Exception as e:
                logger.error(f"Erreur suppression dataset: {str(e)}")
                return False
        return False
    
    @staticmethod
    def list_all_datasets() -> List[Dict]:
        """Liste tous les datasets avec leurs métadonnées."""
        datasets = []
        current_time = time.time()
        
        for filepath in STORAGE_DIR.glob("*.csv"):
            try:
                dataset_id = filepath.stem
                stats = filepath.stat()
                df = pd.read_csv(filepath)
                
                datasets.append({
                    "dataset_id": dataset_id,
                    "filename": filepath.name,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "size_bytes": stats.st_size,
                    "created_at": stats.st_ctime,
                    "modified_at": stats.st_mtime,
                    "age_hours": (current_time - stats.st_mtime) / 3600
                })
            except Exception as e:
                logger.error(f"Erreur lecture {filepath.name}: {str(e)}")
        
        # Trier par date (plus récent en premier)
        datasets.sort(key=lambda x: x["modified_at"], reverse=True)
        return datasets
    
    @staticmethod
    def cleanup_old_datasets(max_age_hours: int = 24) -> int:
        """Nettoie les datasets plus anciens que max_age_hours."""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        deleted_count = 0
        
        for filepath in STORAGE_DIR.glob("*.csv"):
            file_age = current_time - filepath.stat().st_mtime
            if file_age > max_age_seconds:
                try:
                    os.remove(filepath)
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Erreur suppression {filepath.name}: {str(e)}")
        
        if deleted_count > 0:
            logger.info(f"{deleted_count} dataset(s) nettoyé(s)")
        
        return deleted_count
