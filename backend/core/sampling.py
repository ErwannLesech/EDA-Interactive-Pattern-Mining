import numpy as np
import pandas as pd
from typing import List, Tuple, FrozenSet
import random

class PatternSampler:
    """Classe pour l'échantillonnage et l'analyse de motifs"""

    def __init__(self, patterns: pd.DataFrame):
        self.patterns = patterns
        random.seed(42)
        np.random.seed(42)

    def calculate_surprise(self, itemset: frozenset, observed_support: float) -> float:
        if not itemset:
            return 0.0

        expected_support = 1.0
        for item in itemset:
            item_row = self.patterns[self.patterns["itemset"].apply(lambda x: len(x) == 1 and item in x)]
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
        surprise_scores: List[float] = []
        redundancy_scores: List[float] = []

        all_itemsets = self.patterns["itemset"].tolist()

        for _, row in self.patterns.iterrows():
            itemset = row["itemset"]
            observed_support = float(row["support"])

            surprise = self.calculate_surprise(itemset, observed_support)
            surprise_scores.append(surprise)

            redundancy = self.calculate_redundancy_penalty(itemset, all_itemsets)
            redundancy_scores.append(redundancy)

        surprise_scores=self.normalize(surprise_scores)
        redundancy_scores=self.normalize(redundancy_scores)

        self.patterns["surprise"] = surprise_scores
        self.patterns["redundancy"] = redundancy_scores
        self.patterns['support_normalized'] = self.normalize(self.patterns['support'].tolist())

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
        if not 'composite_score' in self.patterns:
            self.composite_scoring(support_weight,surprise_weight,redundancy_weight)
        indexes = np.random.choice(range(len(self.patterns)),size=k, replace=replacement, p=self.patterns["composite_score"])
        result = [(i, self.patterns.iloc[i]['itemsets']) for i in indexes]
        return result
    
    def user_feedback(self, index : int, alpha: float, beta: float, rating: int):
        if rating==1:
            self.patterns.iloc[index]['composite_score']+=np.exp(-alpha)
        else:
            self.patterns.iloc[index]['composite_score']-=np.exp(-beta)