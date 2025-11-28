import numpy as np
import pandas as pd
from typing import List, Tuple, FrozenSet
import random
import time
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
        self.feedback_history = []  # Historique des feedbacks pour évaluation
        self.pattern_scores = {} # Dictionnaire persistant pour stocker les scores des motifs (frozenset -> score)
        self.support_weight=0.0
        self.surprise_weight=0.0
        self.redundancy_weight=0.0

    def calculate_surprise(self, itemset: frozenset, observed_support: float) -> float:
        if not itemset:
            return 0.0

        expected_support = 1.0
        for item in itemset:
            # Chercher les itemsets de taille 1 contenant cet item
            item_row = self.patterns[self.patterns["itemsets"].apply(lambda x: len(x) == 1 and item in x)]
            if not item_row.empty:
                expected_support *= float(item_row["support"].iloc[0])
            else:
                return 0.0

        if expected_support == 0:
            return 0.0

        surprise = abs(observed_support - expected_support) / expected_support
        return float(surprise)

    def calculate_redundancy_penalty(
        self, target_itemset: frozenset, all_itemsets: List[frozenset]
    ) -> float:
        if not all_itemsets or len(all_itemsets) <= 1:
            return 0.0

        similarities: List[float] = []
        target_set = set(target_itemset)

        for other_itemset in all_itemsets:
            other_set = set(other_itemset)
            if target_set == other_set:
                continue

            intersection = len(target_set & other_set)
            union = len(target_set | other_set)
            if union > 0:
                similarities.append(intersection / union)

        redundancy = float(np.mean(similarities)) if similarities else 0.0
        return redundancy
    
    def normalize(self, scores: List[float]) -> List[float]:
        if not scores:
            return []
        
        min_value = min(scores)
        max_value = max(scores)
        
        if max_value == min_value:
            return [0.0 for _ in scores]
        
        return list((np.array(scores) - min_value) / (max_value - min_value))

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
        if 'surprise' not in self.patterns or 'redundancy' not in self.patterns:
            self.add_surprise_and_redundancy_to_patterns()

        composite_scores = (
            support_weight * np.array(self.patterns["support"].tolist())
            + surprise_weight * np.array(self.patterns['surprise'].tolist())
            + redundancy_weight * (1 - np.array(self.patterns['redundancy'].tolist()))
        )
        self.patterns["composite_score"] = composite_scores.tolist()
    
    def importance_sampling(self,support_weight: float, surprise_weight: float, redundancy_weight: float, k: int, replacement: bool) -> List[Tuple[FrozenSet[str], int]]:
        logger.info("Démarrage de l'échantillonnage des motifs avec importance sampling")
        logger.info(f"Support weight: {support_weight}, Surprise weight: {surprise_weight}, replacement: {replacement}")
        if 'composite_score' not in self.patterns or  self.support_weight != support_weight or self.surprise_weight != surprise_weight or self.redundancy_weight != redundancy_weight:
            self.composite_scoring(support_weight,surprise_weight,redundancy_weight)
            self.support_weight = support_weight
            self.surprise_weight = surprise_weight
            self.redundancy_weight = redundancy_weight
        
        # Ensure non-negative scores
        self.patterns["composite_score"] = self.patterns["composite_score"].apply(lambda x: max(0.0, x))
        
        # Normalize probabilities
        total_score = np.sum(self.patterns["composite_score"])
        n = len(self.patterns)
        composite_scores=[1.0/n] * n
        if total_score > 0:
            composite_scores = self.patterns["composite_score"] / total_score
        
            
        indexes = np.random.choice(range(len(self.patterns)),size=min(k, len(self.patterns)) if not replacement else k, replace=replacement, p=composite_scores)
        result: List[Tuple[FrozenSet[str], int]] = [(self.patterns.iloc[i]['itemsets'], i) for i in indexes]
        return result
    
    def user_feedback(self, index : int, alpha: float, beta: float, rating: int):
        """Met à jour les scores des motifs en fonction du feedback utilisateur"""
        # Ajuster alpha et beta dans [2.0, 5.0] pour éviter des ajustements trop importants
        alpha = 2.0 + 3.0 * (1 - alpha)
        beta = 2.0 + 3.0 * (1 - beta)
        # Enregistrer le feedback pour l'évaluation
        self.feedback_history.append({
            "pattern_id": index,
            "rating": rating,
            "alpha": alpha,
            "beta": beta,
            "timestamp": time.time()
        })
        
        logger.info(f"Réception du feedback utilisateur pour le motif index {index} avec rating {rating}")
        
        # Mise à jour du score dans le DataFrame actuel
        col_idx : int = self.patterns.columns.get_loc('composite_score') # type: ignore[assignment]
        logger.info(f"score avant {self.patterns.iat[index, col_idx]}")
        if rating == 1:
            adjustment = np.exp(-alpha)
            self.patterns.iat[index, col_idx] += adjustment
        elif rating == -1:
            adjustment = np.exp(-beta)
            self.patterns.iat[index, col_idx] -= adjustment
        logger.info(f"Score apres {self.patterns.iat[index, col_idx]}")