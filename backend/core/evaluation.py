import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, FrozenSet
import time
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class PatternEvaluator:
    """Classe pour évaluer la qualité de l'échantillonnage de motifs"""
    
    def __init__(self):
        self.feedback_history = []
        self.sampling_times = []
    
    def calculate_acceptance_rate(self, feedback_list: List[Dict]) -> Dict:
        """
        Calcule le taux d'acceptation basé sur les feedbacks utilisateur.
        
        Args:
            feedback_list: Liste de dictionnaires avec 'pattern_id' et 'rating' (1=like, -1=dislike)
        
        Returns:
            Dictionnaire avec taux d'acceptation, nombre de likes/dislikes
        """
        if not feedback_list:
            return {
                "acceptance_rate": 0.0,
                "total_feedbacks": 0,
                "likes": 0,
                "dislikes": 0,
                "neutral": 0
            }
        
        likes = sum(1 for f in feedback_list if f.get("rating", 0) == 1)
        dislikes = sum(1 for f in feedback_list if f.get("rating", 0) == -1)
        neutral = len(feedback_list) - likes - dislikes
        
        acceptance_rate = likes / len(feedback_list) if len(feedback_list) > 0 else 0.0
        
        return {
            "acceptance_rate": float(acceptance_rate),
            "total_feedbacks": len(feedback_list),
            "likes": likes,
            "dislikes": dislikes,
            "neutral": neutral
        }
    
    def calculate_diversity(self, patterns_df: pd.DataFrame) -> Dict:
        """
        Calcule la diversité des motifs échantillonnés.
        Utilise la dissimilarité de Jaccard moyenne entre tous les paires de motifs.
        
        Args:
            patterns_df: DataFrame contenant les motifs avec colonne 'itemsets'
        
        Returns:
            Dictionnaire avec métriques de diversité
        """
        if patterns_df.empty or len(patterns_df) < 2:
            return {
                "average_jaccard_distance": 0.0,
                "diversity_score": 0.0,
                "unique_items_count": 0,
                "average_pattern_length": 0.0
            }
        
        itemsets = patterns_df['itemsets'].tolist()
        n = len(itemsets)
        
        # Calculer les distances de Jaccard
        jaccard_distances = []
        for i in range(n):
            for j in range(i + 1, n):
                set_i = set(itemsets[i])
                set_j = set(itemsets[j])
                
                intersection = len(set_i & set_j)
                union = len(set_i | set_j)
                
                if union > 0:
                    jaccard_sim = intersection / union
                    jaccard_dist = 1 - jaccard_sim
                    jaccard_distances.append(jaccard_dist)
        
        avg_jaccard_distance = float(np.mean(jaccard_distances)) if jaccard_distances else 0.0
        
        # Compter les items uniques
        all_items = set()
        for itemset in itemsets:
            all_items.update(itemset)
        
        # Longueur moyenne des motifs
        avg_length = float(np.mean([len(itemset) for itemset in itemsets]))
        
        return {
            "average_jaccard_distance": avg_jaccard_distance,
            "diversity_score": avg_jaccard_distance,  # Plus c'est élevé, plus c'est diversifié
            "unique_items_count": len(all_items),
            "average_pattern_length": avg_length
        }
    
    def calculate_coverage(self, patterns_df: pd.DataFrame, 
                          all_patterns_df: pd.DataFrame) -> Dict:
        """
        Calcule la couverture de l'échantillon par rapport au pool complet de motifs.
        
        Args:
            patterns_df: DataFrame des motifs échantillonnés
            all_patterns_df: DataFrame du pool complet de motifs
        
        Returns:
            Dictionnaire avec métriques de couverture
        """
        if patterns_df.empty or all_patterns_df.empty:
            return {
                "pattern_coverage": 0.0,
                "item_coverage": 0.0,
                "support_coverage": 0.0,
                "sampled_count": 0,
                "total_count": 0
            }
        
        # Couverture en nombre de motifs
        pattern_coverage = len(patterns_df) / len(all_patterns_df)
        
        # Couverture des items
        sampled_items = set()
        for itemset in patterns_df['itemsets'].tolist():
            sampled_items.update(itemset)
        
        all_items = set()
        for itemset in all_patterns_df['itemsets'].tolist():
            all_items.update(itemset)
        
        item_coverage = len(sampled_items) / len(all_items) if len(all_items) > 0 else 0.0
        
        # Couverture du support (somme des supports des motifs échantillonnés / total)
        sampled_support = patterns_df['support'].sum() if 'support' in patterns_df.columns else 0
        total_support = all_patterns_df['support'].sum() if 'support' in all_patterns_df.columns else 1
        support_coverage = sampled_support / total_support if total_support > 0 else 0.0
        
        return {
            "pattern_coverage": float(pattern_coverage),
            "item_coverage": float(item_coverage),
            "support_coverage": float(support_coverage),
            "sampled_count": len(patterns_df),
            "total_count": len(all_patterns_df)
        }
    
    def calculate_stability(self, sampling_function, patterns_df: pd.DataFrame,
                          params: Dict, n_iterations: int = 10) -> Dict:
        """
        Calcule la stabilité de l'échantillonnage en testant différentes seeds.
        
        Args:
            sampling_function: Fonction d'échantillonnage à tester
            patterns_df: DataFrame des motifs complets
            params: Paramètres pour la fonction d'échantillonnage
            n_iterations: Nombre d'itérations avec différentes seeds
        
        Returns:
            Dictionnaire avec métriques de stabilité
        """
        if patterns_df.empty:
            return {
                "stability_score": 0.0,
                "jaccard_similarities": [],
                "mean_similarity": 0.0,
                "std_similarity": 0.0
            }
        
        samples = []
        original_seed = np.random.get_state()
        
        # Effectuer n échantillonnages avec différentes seeds
        for i in range(n_iterations):
            np.random.seed(42 + i)
            try:
                result = sampling_function(patterns_df, **params)
                # Extraire les itemsets selon le format de retour
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], tuple):
                        # Format (itemset, index)
                        sample_itemsets = [frozenset(itemset) for itemset, _ in result]
                    else:
                        # Liste d'itemsets
                        sample_itemsets = [frozenset(item) if isinstance(item, (list, set)) else item 
                                         for item in result]
                    samples.append(set(sample_itemsets))
            except Exception as e:
                logger.warning(f"Erreur lors de l'échantillonnage itération {i}: {str(e)}")
        
        # Restaurer la seed originale
        np.random.set_state(original_seed)
        
        if len(samples) < 2:
            return {
                "stability_score": 0.0,
                "jaccard_similarities": [],
                "mean_similarity": 0.0,
                "std_similarity": 0.0
            }
        
        # Calculer les similarités de Jaccard entre toutes les paires
        similarities = []
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                intersection = len(samples[i] & samples[j])
                union = len(samples[i] | samples[j])
                if union > 0:
                    similarity = intersection / union
                    similarities.append(similarity)
        
        mean_sim = float(np.mean(similarities)) if similarities else 0.0
        std_sim = float(np.std(similarities)) if similarities else 0.0
        
        return {
            "stability_score": mean_sim,  # Plus c'est élevé, plus c'est stable
            "jaccard_similarities": similarities,
            "mean_similarity": mean_sim,
            "std_similarity": std_sim,
            "n_iterations": len(samples)
        }
    
    def measure_response_time(self, sampling_function, patterns_df: pd.DataFrame,
                            params: Dict, n_runs: int = 5) -> Dict:
        """
        Mesure le temps de réponse de la fonction d'échantillonnage.
        
        Args:
            sampling_function: Fonction à mesurer
            patterns_df: DataFrame des motifs
            params: Paramètres pour la fonction
            n_runs: Nombre d'exécutions pour la moyenne
        
        Returns:
            Dictionnaire avec statistiques de temps
        """
        times = []
        
        for _ in range(n_runs):
            start_time = time.time()
            try:
                sampling_function(patterns_df, **params)
                elapsed = time.time() - start_time
                times.append(elapsed)
            except Exception as e:
                logger.warning(f"Erreur lors de la mesure du temps: {str(e)}")
        
        if not times:
            return {
                "mean_time": 0.0,
                "std_time": 0.0,
                "min_time": 0.0,
                "max_time": 0.0,
                "n_runs": 0
            }
        
        return {
            "mean_time": float(np.mean(times)),
            "std_time": float(np.std(times)),
            "min_time": float(np.min(times)),
            "max_time": float(np.max(times)),
            "n_runs": len(times)
        }
    
    def comprehensive_evaluation(self, sampled_patterns_df: pd.DataFrame,
                                all_patterns_df: pd.DataFrame,
                                feedback_list: List[Dict],
                                sampling_function=None,
                                params: Dict = None) -> Dict:
        """
        Effectue une évaluation complète avec toutes les métriques.
        
        Args:
            sampled_patterns_df: Motifs échantillonnés
            all_patterns_df: Pool complet de motifs
            feedback_list: Liste des feedbacks utilisateur
            sampling_function: Fonction d'échantillonnage (optionnel)
            params: Paramètres pour la fonction (optionnel)
        
        Returns:
            Dictionnaire avec toutes les métriques d'évaluation
        """
        results = {}
        
        # Taux d'acceptation
        results["acceptance"] = self.calculate_acceptance_rate(feedback_list)
        
        # Diversité
        results["diversity"] = self.calculate_diversity(sampled_patterns_df)
        
        # Couverture
        results["coverage"] = self.calculate_coverage(sampled_patterns_df, all_patterns_df)
        
        # Stabilité (si fonction fournie)
        if sampling_function and params:
            results["stability"] = self.calculate_stability(
                sampling_function, all_patterns_df, params
            )
            results["response_time"] = self.measure_response_time(
                sampling_function, all_patterns_df, params
            )
        else:
            results["stability"] = {
                "stability_score": None,
                "message": "Fonction d'échantillonnage non fournie"
            }
            results["response_time"] = {
                "mean_time": None,
                "message": "Fonction d'échantillonnage non fournie"
            }
        
        # Score global (moyenne pondérée)
        weights = {
            "acceptance": 0.30,
            "diversity": 0.25,
            "coverage": 0.25,
            "stability": 0.20
        }
        
        acceptance_score = results["acceptance"]["acceptance_rate"]
        diversity_score = results["diversity"]["diversity_score"]
        coverage_score = results["coverage"]["pattern_coverage"]
        stability_score = results["stability"].get("stability_score", 0) or 0
        
        overall_score = (
            weights["acceptance"] * acceptance_score +
            weights["diversity"] * diversity_score +
            weights["coverage"] * coverage_score +
            weights["stability"] * stability_score
        )
        
        results["overall_score"] = float(overall_score)
        
        return results
