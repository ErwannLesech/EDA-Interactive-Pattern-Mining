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
        random.seed(42)
        np.random.seed(42)

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
        if not 'composite_score' in self.patterns:
            self.composite_scoring(support_weight,surprise_weight,redundancy_weight)
        self.patterns["composite_score"]=self.patterns["composite_score"]/np.sum(self.patterns["composite_score"])
        indexes = np.random.choice(range(len(self.patterns)),size=min(k, len(self.patterns)) if not replacement else k, replace=replacement, p=self.patterns["composite_score"])
        result: List[Tuple[FrozenSet[str], int]] = [(self.patterns.iloc[i]['itemsets'], i) for i in indexes]
        return result
    
    def user_feedback(self, index : int, alpha: float, beta: float, rating: int):
        # Enregistrer le feedback pour l'évaluation
        self.feedback_history.append({
            "pattern_id": index,
            "rating": rating,
            "alpha": alpha,
            "beta": beta,
            "timestamp": time.time()
        })
        
        if rating==1:
            self.patterns.iloc[index]['composite_score']+=np.exp(-alpha)
        else:
            self.patterns.iloc[index]['composite_score']-=np.exp(-beta)
    
    # TwoStep Pattern Sampling (Boley et al., KDD'2011)
    def twostep_sampling(self, transactions: List[List[str]], k: int) -> List[List[str]]:
        """
        TwoStep pattern sampling: échantillonne k motifs depuis transactions.
        
        Args:
            transactions: Liste de transactions (chaque transaction = liste d'items)
            k: Nombre de motifs à échantillonner
            
        Returns:
            Liste de motifs échantillonnés
        """
        from decimal import Decimal
        
        # Étape 1: Calculer les poids cumulatifs pour chaque transaction
        weights = []
        cumulative_weight = Decimal(0)
        
        for transaction in transactions:
            weight = 2 ** len(transaction)
            cumulative_weight += Decimal(weight)
            weights.append(cumulative_weight)
        
        Z = weights[-1] if weights else Decimal(1)
        sampled_patterns = []
        
        # Étape 2: Échantillonner k motifs
        for _ in range(k):
            # Sélectionner une transaction aléatoirement (pondérée)
            rand_value = Decimal(random.random()) * Z
            
            # Recherche binaire pour trouver la transaction
            left, right = 0, len(weights)
            while left < right:
                mid = (left + right) // 2
                if weights[mid] < rand_value:
                    left = mid + 1
                else:
                    right = mid
            
            t_id = min(left, len(transactions) - 1)
            selected_transaction = transactions[t_id]
            
            # Échantillonner un sous-ensemble de la transaction
            pattern = [item for item in selected_transaction if random.random() > 0.5]
            sampled_patterns.append(pattern if pattern else [selected_transaction[0]] if selected_transaction else [])
        
        return sampled_patterns
    
    # GDPS (Generic Direct Pattern Sampling)
    def gdps_sampling(self, transactions: List[List[str]], k: int, 
                     min_norm: int = 1, max_norm: int = 10, 
                     utility: str = "freq") -> List[List[str]]:
        """
        Generic Direct Pattern Sampling avec différentes utilités.
        
        Args:
            transactions: Liste de transactions
            k: Nombre de motifs à échantillonner
            min_norm: Taille minimale des motifs
            max_norm: Taille maximale des motifs
            utility: Type d'utilité ("freq", "area", "decay")
            
        Returns:
            Liste de motifs échantillonnés
        """
        from decimal import Decimal
        import math
        
        def compute_utility(norm: int, utility_type: str) -> float:
            """Calcule l'utilité selon le type"""
            if utility_type == "freq":
                return 1.0
            elif utility_type == "area":
                return float(norm)
            elif utility_type == "decay":
                return math.exp(-norm)
            return 1.0
        
        # Calculer les poids pour chaque transaction
        weights = []
        cumulative_weight = Decimal(0)
        
        for transaction in transactions:
            t_size = len(transaction)
            weight = Decimal(0)
            
            # Sommer les utilités pour toutes les tailles possibles
            for l in range(min_norm, min(max_norm + 1, t_size + 1)):
                from math import comb
                n_patterns = comb(t_size, l)
                util = compute_utility(l, utility)
                weight += Decimal(n_patterns * util)
            
            cumulative_weight += weight
            weights.append(cumulative_weight)
        
        Z = weights[-1] if weights else Decimal(1)
        sampled_patterns = []
        
        # Échantillonner k motifs
        for _ in range(k):
            # Sélectionner une transaction
            rand_value = Decimal(random.random()) * Z
            
            left, right = 0, len(weights)
            while left < right:
                mid = (left + right) // 2
                if weights[mid] < rand_value:
                    left = mid + 1
                else:
                    right = mid
            
            t_id = min(left, len(transactions) - 1)
            transaction = transactions[t_id]
            t_size = len(transaction)
            
            # Déterminer la taille du motif selon l'utilité
            norm_probs = []
            for l in range(min_norm, min(max_norm + 1, t_size + 1)):
                from math import comb
                n_patterns = comb(t_size, l)
                util = compute_utility(l, utility)
                norm_probs.append(n_patterns * util)
            
            # Normaliser les probabilités
            total = sum(norm_probs)
            if total > 0:
                norm_probs = [p / total for p in norm_probs]
            else:
                norm_probs = [1.0 / len(norm_probs)] * len(norm_probs)
            
            # Choisir une taille
            chosen_norm = np.random.choice(
                range(min_norm, min(max_norm + 1, t_size + 1)),
                p=norm_probs
            )
            
            # Échantillonner un motif de cette taille
            pattern = random.sample(transaction, min(chosen_norm, len(transaction)))
            sampled_patterns.append(pattern)
        
        return sampled_patterns
