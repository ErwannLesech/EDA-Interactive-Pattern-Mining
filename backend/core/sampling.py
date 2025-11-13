import numpy as np
import pandas as pd
from typing import List, Tuple, FrozenSet
import random
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
class PatternSampler:
    """Classe pour l'échantillonnage et l'analyse de motifs"""

    def __init__(self, patterns: pd.DataFrame):
        self.patterns = patterns
        random.seed(42)
        np.random.seed(42)

    def add_surprise_and_redundancy_to_patterns(self):
        """
        Ajoute les colonnes 'surprise' et 'redundancy' au DataFrame des motifs, optimisé pour gros datasets.
        """
        # Pré-calcule le support individuel pour tous les items (itemsets de taille 1)
        single_item_support = {
            tuple(row["itemsets"])[0]: row["support"]
            for _, row in self.patterns.iterrows()
            if len(row["itemsets"]) == 1
        }

        all_itemsets = self.patterns["itemsets"].tolist()
        n = len(all_itemsets)

        # Surprise : vectorisé
        surprise_scores = np.zeros(n)
        for i, row in enumerate(self.patterns.itertuples(index=False)):
            itemset = row.itemsets
            observed_support = float(row.support)
            expected_support = 1.0
            for item in itemset:
                expected_support *= single_item_support.get(item, 0.0)
            surprise_scores[i] = abs(observed_support - expected_support) / expected_support if expected_support else 0.0

        # # Redondance : vectorisé, limitation des comparaisons
        sets = [set(s) for s in all_itemsets]
        redundancy_scores = np.zeros(n)
        for i, s1 in enumerate(sets):
            len_s1 = len(s1)
            similarities = []
            for j, s2 in enumerate(sets):
                if i == j:
                    continue
                if abs(len(s2) - len_s1) > 1:
                    continue
                inter = len(s1 & s2)
                union = len(s1 | s2)
                if union > 0:
                    similarities.append(inter / union)
            redundancy_scores[i] = np.mean(similarities) if similarities else 0.0
        
        # Normalisation rapide
        def normalize(arr):
            arr = np.array(arr)
            if arr.size == 0:
                return arr  # ou np.zeros(0)
            minv, maxv = arr.min(), arr.max()
            return (arr - minv) / (maxv - minv) if maxv > minv else arr

        self.patterns["surprise"] = normalize(surprise_scores)
        self.patterns["redundancy"] = normalize(redundancy_scores)

    def composite_scoring(self, support_weight: float, surprise_weight: float, redundancy_weight: float):
        """
        Calcule un score composite basé sur support, surprise et redondance.
        
        Score = w1 * support + w2 * surprise_norm + w3 * (1 - redundancy_norm)
        """
        self.add_surprise_and_redundancy_to_patterns()

        composite_scores = (
            support_weight * np.array(self.patterns["support"].tolist())
            + surprise_weight * np.array(self.patterns['surprise'].tolist())
            + redundancy_weight * (1 - np.array(self.patterns['redundancy'].tolist()))
        )
        composite_scores=composite_scores/np.sum(composite_scores)
        self.patterns["composite_score"] = composite_scores.tolist()
    
    def importance_sampling(self,support_weight: float, surprise_weight: float, redundancy_weight: float, k: int, replacement: bool) -> List[Tuple[FrozenSet[str], int]]:
        logger.info("Démarrage de l'échantillonnage des motifs avec importance sampling")
        logger.info(f"Support weight: {support_weight}, Surprise weight: {surprise_weight}, replacement: {replacement}")
        if not 'composite_score' in self.patterns:
            self.composite_scoring(support_weight,surprise_weight,redundancy_weight)
        indexes = np.random.choice(range(len(self.patterns)),size=min(k, len(self.patterns)) if not replacement else k, replace=replacement, p=self.patterns["composite_score"])
        result: List[Tuple[FrozenSet[str], int]] = [(self.patterns.iloc[i]['itemsets'], i) for i in indexes]
        return result
    
    def user_feedback(self, index : int, alpha: float, beta: float, rating: int):
        if rating==1:
            self.patterns.iloc[index]['composite_score']+=np.exp(-alpha)
        elif rating==-1:
            self.patterns.iloc[index]['composite_score']-=np.exp(-beta)