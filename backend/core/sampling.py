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
        self.mined_patterns = patterns.copy() if not patterns.empty else pd.DataFrame() # Pool complet des motifs minés
        self.patterns = patterns # Motifs actuellement affichés/échantillonnés
        self.feedback_history = []  # Historique des feedbacks pour évaluation
        self.pattern_scores = {} # Dictionnaire persistant pour stocker les scores des motifs (frozenset -> score)
        # Note: Seeds removed to allow proper stability evaluation with different random states

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
        Ajoute les colonnes 'surprise' et 'redundancy' au DataFrame des motifs minés (self.mined_patterns), optimisé pour gros datasets.
        """
        if self.mined_patterns.empty:
            return

        # Pré-calcule le support individuel pour tous les items (itemsets de taille 1)
        single_item_support = {
            tuple(row["itemsets"])[0]: row["support"]
            for _, row in self.mined_patterns.iterrows()
            if len(row["itemsets"]) == 1
        }

        all_itemsets = self.mined_patterns["itemsets"].tolist()
        n = len(all_itemsets)

        # Surprise : vectorisé
        surprise_scores = np.zeros(n)
        for i, row in enumerate(self.mined_patterns.itertuples(index=False)):
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

        self.mined_patterns["surprise"] = normalize(surprise_scores)
        self.mined_patterns["redundancy"] = normalize(redundancy_scores)

    def composite_scoring(self, support_weight: float, surprise_weight: float, redundancy_weight: float):
        """
        Calcule un score composite basé sur support, surprise et redondance pour le pool miné.
        
        Score = w1 * support + w2 * surprise_norm + w3 * (1 - redundancy_norm)
        """
        if self.mined_patterns.empty:
            return

        self.add_surprise_and_redundancy_to_patterns()

        composite_scores = (
            support_weight * np.array(self.mined_patterns["support"].tolist())
            + surprise_weight * np.array(self.mined_patterns['surprise'].tolist())
            + redundancy_weight * (1 - np.array(self.mined_patterns['redundancy'].tolist()))
        )
        self.mined_patterns["composite_score"] = composite_scores.tolist()
    
    def importance_sampling(self,support_weight: float, surprise_weight: float, redundancy_weight: float, k: int, replacement: bool) -> List[Tuple[FrozenSet[str], int]]:
        logger.info("Démarrage de l'échantillonnage des motifs avec importance sampling")
        logger.info(f"Support weight: {support_weight}, Surprise weight: {surprise_weight}, replacement: {replacement}")
        
        if self.mined_patterns.empty:
            logger.error("Tentative d'importance sampling sur un pool vide")
            return []

        if not 'composite_score' in self.mined_patterns:
            self.composite_scoring(support_weight,surprise_weight,redundancy_weight)
        
        # Ensure non-negative scores
        self.mined_patterns["composite_score"] = self.mined_patterns["composite_score"].apply(lambda x: max(0.0, x))
        
        # Appliquer les scores persistants (feedback) au pool miné
        for idx, row in self.mined_patterns.iterrows():
            itemset = row['itemsets']
            if itemset in self.pattern_scores:
                # On combine le score calculé avec le score persistant
                # Par exemple, on multiplie par le score persistant (qui est autour de 0.5 par défaut)
                # Ou on remplace ? Ici on va ajuster le score composite
                # Si score > 0.5 (aimé), on booste. Si < 0.5 (détesté), on réduit.
                feedback_modifier = max(0.0, self.pattern_scores[itemset]) * 2 # 0.5 -> 1.0 (neutre), 1.0 -> 2.0 (boost), 0.0 -> 0.0 (kill)
                self.mined_patterns.at[idx, "composite_score"] *= feedback_modifier

        # Ensure non-negative scores (safety check)
        self.mined_patterns["composite_score"] = self.mined_patterns["composite_score"].apply(lambda x: max(0.0, x))

        # Normalize probabilities
        total_score = np.sum(self.mined_patterns["composite_score"])
        if total_score > 0:
            self.mined_patterns["composite_score"] = self.mined_patterns["composite_score"] / total_score
        else:
            # Fallback to uniform distribution if all scores are 0
            n = len(self.mined_patterns)
            self.mined_patterns["composite_score"] = [1.0/n] * n
            
        indexes = np.random.choice(range(len(self.mined_patterns)),size=min(k, len(self.mined_patterns)) if not replacement else k, replace=replacement, p=self.mined_patterns["composite_score"])
        
        # Mettre à jour self.patterns avec les motifs échantillonnés pour l'affichage et le feedback futur
        self.patterns = self.mined_patterns.iloc[indexes].copy().reset_index(drop=True)
        
        result: List[Tuple[FrozenSet[str], int]] = [(self.patterns.iloc[i]['itemsets'], i) for i in range(len(self.patterns))]
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
        
        logger.info(f"Réception du feedback utilisateur pour le motif index {index} avec rating {rating}")
        
        # Mise à jour du score dans le DataFrame actuel (self.patterns)
        # Note: self.patterns peut contenir 'itemset' (list) ou 'itemsets' (frozenset)
        col_idx : int = self.patterns.columns.get_loc('composite_score') # type: ignore[assignment]
        
        # Récupérer l'itemset pour la persistance
        try:
            # Gérer le cas où 'itemset' est une liste (TwoStep/GDPS) ou un frozenset (Importance)
            itemset_val = self.patterns.iloc[index]['itemset'] if 'itemset' in self.patterns.columns else self.patterns.iloc[index]['itemsets']
            if isinstance(itemset_val, list):
                itemset_key = frozenset(itemset_val)
            else:
                itemset_key = itemset_val
        except Exception as e:
            logger.warning(f"Impossible de récupérer l'itemset pour l'index {index}: {e}")
            itemset_key = None

        if rating == 1:
            adjustment = np.exp(-alpha)
            self.patterns.iat[index, col_idx] += adjustment
            if itemset_key:
                self.pattern_scores[itemset_key] = self.pattern_scores.get(itemset_key, 0.5) + adjustment
        elif rating == -1:
            adjustment = np.exp(-beta)
            self.patterns.iat[index, col_idx] -= adjustment
            if itemset_key:
                self.pattern_scores[itemset_key] = self.pattern_scores.get(itemset_key, 0.5) - adjustment
        
        # Si le motif existe dans le pool miné (self.mined_patterns), mettre à jour son score là aussi
        # pour influencer le futur Importance Sampling
        if itemset_key and not self.mined_patterns.empty:
            # Trouver l'index dans mined_patterns
            # C'est un peu lent si le pool est grand, mais nécessaire
            # On suppose que 'itemsets' est la colonne clé
            mask = self.mined_patterns['itemsets'] == itemset_key
            if mask.any():
                # On ne met pas à jour 'composite_score' directement car il est recalculé à chaque importance_sampling
                # Mais on a déjà mis à jour self.pattern_scores qui est utilisé dans importance_sampling
                pass
            
    # TwoStep Pattern Sampling (Boley et al., KDD'2011)
    def twostep_sampling(self, transactions: List[List[str]], k: int) -> List[List[str]]:
        """
        TwoStep pattern sampling: échantillonne k motifs depuis transactions.
        Intègre le feedback utilisateur via une stratégie de Sampling Importance Resampling (SIR).
        
        Args:
            transactions: Liste de transactions (chaque transaction = liste d'items)
            k: Nombre de motifs à échantillonner
            
        Returns:
            Liste de motifs échantillonnés
        """
        from decimal import Decimal
        
        # Stratégie SIR (Sampling Importance Resampling) pour intégrer le feedback
        # 1. Générer un pool plus large de candidats (oversampling)
        oversampling_factor = 10
        pool_size = k * oversampling_factor
        
        # Étape 1: Calculer les poids cumulatifs pour chaque transaction
        weights = []
        cumulative_weight = Decimal(0)
        
        for transaction in transactions:
            weight = 2 ** len(transaction)
            cumulative_weight += Decimal(weight)
            weights.append(cumulative_weight)
        
        Z = weights[-1] if weights else Decimal(1)
        candidate_pool = []
        candidate_weights = []
        
        # Étape 2: Générer le pool de candidats
        for _ in range(pool_size):
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
            candidate = pattern if pattern else [selected_transaction[0]] if selected_transaction else []
            
            candidate_pool.append(candidate)
            
            # Calculer le poids basé sur le feedback
            # Score par défaut = 0.5. Si feedback positif > 0.5, négatif < 0.5
            score = self.pattern_scores.get(frozenset(candidate), 0.5)
            # On utilise le score comme poids (en s'assurant qu'il est positif)
            candidate_weights.append(max(0.0, score))
            
        # Étape 3: Sélectionner k motifs finaux basés sur les poids (feedback)
        total_weight = sum(candidate_weights)
        if total_weight > 0:
            probs = [w / total_weight for w in candidate_weights]
        else:
            probs = [1.0 / pool_size] * pool_size
            
        # Échantillonnage pondéré sans remise (si possible) ou avec remise
        # Note: np.random.choice attend un tableau 1D pour p
        selected_indices = np.random.choice(
            range(pool_size), 
            size=k, 
            replace=True, # Avec remise car le pool peut contenir des doublons qu'on veut peut-être revoir
            p=probs
        )
        
        return [candidate_pool[i] for i in selected_indices]
    
    # GDPS (Generic Direct Pattern Sampling)
    def gdps_sampling(self, transactions: List[List[str]], k: int, 
                     min_norm: int = 1, max_norm: int = 10, 
                     utility: str = "freq") -> List[List[str]]:
        """
        Generic Direct Pattern Sampling avec différentes utilités.
        Intègre le feedback utilisateur via une stratégie de Sampling Importance Resampling (SIR).
        
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
            
        # Stratégie SIR (Sampling Importance Resampling)
        oversampling_factor = 10
        pool_size = k * oversampling_factor
        
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
        candidate_pool = []
        candidate_weights = []
        
        # Générer le pool de candidats
        for _ in range(pool_size):
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
            valid_lengths = []
            for l in range(min_norm, min(max_norm + 1, t_size + 1)):
                from math import comb
                n_patterns = comb(t_size, l)
                util = compute_utility(l, utility)
                norm_probs.append(n_patterns * util)
                valid_lengths.append(l)
            
            if not valid_lengths:
                # Fallback si aucune taille valide
                candidate_pool.append([])
                candidate_weights.append(0.0)
                continue

            # Normaliser les probabilités
            total = sum(norm_probs)
            if total > 0:
                norm_probs = [p / total for p in norm_probs]
            else:
                norm_probs = [1.0 / len(norm_probs)] * len(norm_probs)
            
            # Choisir une taille
            chosen_norm = np.random.choice(valid_lengths, p=norm_probs)
            
            # Échantillonner un motif de cette taille
            pattern = random.sample(transaction, min(chosen_norm, len(transaction)))
            candidate_pool.append(pattern)
            
            # Calculer le poids basé sur le feedback
            score = self.pattern_scores.get(frozenset(pattern), 0.5)
            candidate_weights.append(max(0.0, score))
        
        # Sélectionner k motifs finaux
        total_weight = sum(candidate_weights)
        if total_weight > 0:
            probs = [w / total_weight for w in candidate_weights]
        else:
            probs = [1.0 / pool_size] * pool_size
            
        selected_indices = np.random.choice(
            range(pool_size), 
            size=k, 
            replace=True,
            p=probs
        )
        
        return [candidate_pool[i] for i in selected_indices]
