"""
Module d'évaluation pour la partie 4 : Évaluation & reproductibilité

Ce module implémente les métriques d'évaluation suivantes :
- Taux d'acceptation (via feedback)
- Diversité (distance inter-motifs)
- Coverage (couverture des transactions)
- Stabilité (sensibilité au seed)
- Temps de réponse

Auteurs: Lesech Erwann, Le Riboter Aymeric, Aubron Abel, Claude Nathan
Date: Novembre 2025
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, FrozenSet, Set
from collections import Counter
import time
from scipy.spatial.distance import cosine, jaccard
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


class PatternEvaluator:
    """Classe pour l'évaluation de la qualité des motifs extraits et échantillonnés"""
    
    def __init__(self, patterns: pd.DataFrame, transactions: pd.DataFrame = None):
        """
        Initialise l'évaluateur de motifs.
        
        Args:
            patterns: DataFrame contenant les motifs avec leurs métriques
            transactions: DataFrame des transactions originales (pour coverage)
        """
        self.patterns = patterns
        self.transactions = transactions
        self.feedback_history: List[Dict] = []
        
    def calculate_acceptance_rate(self, feedback_data: List[Dict] = None) -> Dict[str, float]:
        """
        Calcule le taux d'acceptation basé sur les feedbacks utilisateur.
        
        Args:
            feedback_data: Liste de dictionnaires {pattern_id: int, rating: int (1=like, -1=dislike)}
            
        Returns:
            Dict avec taux d'acceptation et statistiques
        """
        if feedback_data is None:
            feedback_data = self.feedback_history
            
        if not feedback_data:
            return {
                "acceptance_rate": 0.0,
                "total_feedbacks": 0,
                "likes": 0,
                "dislikes": 0,
                "message": "Aucun feedback disponible"
            }
        
        likes = sum(1 for fb in feedback_data if fb.get('rating', 0) > 0)
        dislikes = sum(1 for fb in feedback_data if fb.get('rating', 0) < 0)
        total = len(feedback_data)
        
        acceptance_rate = (likes / total) * 100 if total > 0 else 0.0
        
        return {
            "acceptance_rate": round(acceptance_rate, 2),
            "total_feedbacks": total,
            "likes": likes,
            "dislikes": dislikes,
            "neutral": total - likes - dislikes
        }
    
    def add_feedback(self, pattern_id: int, rating: int, comment: str = None):
        """Ajoute un feedback utilisateur à l'historique."""
        self.feedback_history.append({
            "pattern_id": pattern_id,
            "rating": rating,
            "comment": comment,
            "timestamp": time.time()
        })
    
    def calculate_diversity(self, method: str = "jaccard") -> Dict[str, float]:
        """
        Calcule la diversité des motifs (distance moyenne inter-motifs).
        
        Args:
            method: Méthode de calcul ('jaccard', 'cosine', 'hamming')
            
        Returns:
            Dict avec métriques de diversité
        """
        if 'itemsets' not in self.patterns.columns and 'itemset' not in self.patterns.columns:
            return {
                "average_diversity": 0.0,
                "min_diversity": 0.0,
                "max_diversity": 0.0,
                "std_diversity": 0.0,
                "method": method,
                "error": "Colonne 'itemsets' ou 'itemset' non trouvée"
            }
        
        # Détecter le nom de la colonne
        itemset_col = 'itemsets' if 'itemsets' in self.patterns.columns else 'itemset'
        itemsets = self.patterns[itemset_col].tolist()
        
        if len(itemsets) < 2:
            return {
                "average_diversity": 0.0,
                "min_diversity": 0.0,
                "max_diversity": 0.0,
                "std_diversity": 0.0,
                "method": method,
                "num_patterns": len(itemsets)
            }
        
        distances = []
        
        # Calculer toutes les distances paires
        for i in range(len(itemsets)):
            for j in range(i + 1, len(itemsets)):
                set_i = set(itemsets[i])
                set_j = set(itemsets[j])
                
                if method == "jaccard":
                    # Distance de Jaccard = 1 - similarité de Jaccard
                    intersection = len(set_i & set_j)
                    union = len(set_i | set_j)
                    distance = 1 - (intersection / union) if union > 0 else 1.0
                    
                elif method == "cosine":
                    # Distance cosinus via vecteurs binaires
                    all_items = list(set_i | set_j)
                    vec_i = [1 if item in set_i else 0 for item in all_items]
                    vec_j = [1 if item in set_j else 0 for item in all_items]
                    
                    # Éviter division par zéro
                    norm_i = np.linalg.norm(vec_i)
                    norm_j = np.linalg.norm(vec_j)
                    
                    if norm_i > 0 and norm_j > 0:
                        similarity = np.dot(vec_i, vec_j) / (norm_i * norm_j)
                        distance = 1 - similarity
                    else:
                        distance = 1.0
                        
                elif method == "hamming":
                    # Distance de Hamming normalisée
                    all_items = list(set_i | set_j)
                    vec_i = [1 if item in set_i else 0 for item in all_items]
                    vec_j = [1 if item in set_j else 0 for item in all_items]
                    distance = sum(a != b for a, b in zip(vec_i, vec_j)) / len(all_items)
                    
                else:
                    raise ValueError(f"Méthode inconnue: {method}")
                
                distances.append(distance)
        
        return {
            "average_diversity": round(float(np.mean(distances)), 4),
            "min_diversity": round(float(np.min(distances)), 4),
            "max_diversity": round(float(np.max(distances)), 4),
            "std_diversity": round(float(np.std(distances)), 4),
            "method": method,
            "num_patterns": len(itemsets),
            "num_comparisons": len(distances)
        }
    
    def calculate_coverage(self) -> Dict[str, float]:
        """
        Calcule la couverture : pourcentage de transactions couvertes par au moins un motif.
        
        Returns:
            Dict avec métriques de couverture
        """
        if self.transactions is None:
            return {
                "coverage_rate": 0.0,
                "covered_transactions": 0,
                "total_transactions": 0,
                "average_coverage_per_pattern": 0.0,
                "error": "Transactions originales non fournies"
            }
        
        if 'itemsets' not in self.patterns.columns and 'itemset' not in self.patterns.columns:
            return {
                "coverage_rate": 0.0,
                "error": "Colonne 'itemsets' ou 'itemset' non trouvée"
            }
        
        itemset_col = 'itemsets' if 'itemsets' in self.patterns.columns else 'itemset'
        itemsets = [set(itemset) for itemset in self.patterns[itemset_col].tolist()]
        
        # Convertir transactions en liste de sets
        # Supposer que transactions a une colonne 'items' ou similaire
        if 'items' in self.transactions.columns:
            transactions_list = [set(str(row).split(',')) for row in self.transactions['items']]
        else:
            # Prendre toutes les colonnes comme items potentiels
            transactions_list = []
            for _, row in self.transactions.iterrows():
                items = set([col for col in self.transactions.columns if row[col] > 0])
                transactions_list.append(items)
        
        total_transactions = len(transactions_list)
        covered = set()
        
        coverage_per_pattern = []
        
        # Pour chaque motif, compter combien de transactions il couvre
        for pattern_set in itemsets:
            pattern_coverage = 0
            for tid, transaction_set in enumerate(transactions_list):
                # Un motif couvre une transaction s'il est un sous-ensemble de celle-ci
                if pattern_set.issubset(transaction_set):
                    covered.add(tid)
                    pattern_coverage += 1
            coverage_per_pattern.append(pattern_coverage)
        
        coverage_rate = (len(covered) / total_transactions * 100) if total_transactions > 0 else 0.0
        avg_coverage_per_pattern = np.mean(coverage_per_pattern) if coverage_per_pattern else 0.0
        
        return {
            "coverage_rate": round(coverage_rate, 2),
            "covered_transactions": len(covered),
            "total_transactions": total_transactions,
            "average_coverage_per_pattern": round(float(avg_coverage_per_pattern), 2),
            "min_coverage": int(np.min(coverage_per_pattern)) if coverage_per_pattern else 0,
            "max_coverage": int(np.max(coverage_per_pattern)) if coverage_per_pattern else 0
        }
    
    def calculate_stability(
        self, 
        sampler_func,
        num_runs: int = 10,
        sample_size: int = 50,
        **sampler_params
    ) -> Dict[str, float]:
        """
        Calcule la stabilité de l'échantillonnage (sensibilité au seed).
        
        Args:
            sampler_func: Fonction d'échantillonnage à tester
            num_runs: Nombre de runs avec des seeds différents
            sample_size: Taille de l'échantillon
            **sampler_params: Paramètres additionnels pour le sampler
            
        Returns:
            Dict avec métriques de stabilité
        """
        samples = []
        
        for seed in range(num_runs):
            # Fixer le seed
            np.random.seed(seed)
            
            # Exécuter l'échantillonnage
            try:
                sample = sampler_func(k=sample_size, **sampler_params)
                # Extraire les IDs ou les itemsets
                if isinstance(sample, list) and len(sample) > 0:
                    if isinstance(sample[0], tuple):
                        sample_ids = [s[0] for s in sample]
                    else:
                        sample_ids = sample
                    samples.append(set(sample_ids))
            except Exception as e:
                logger.warning(f"Erreur lors du run {seed}: {e}")
                continue
        
        if len(samples) < 2:
            return {
                "stability_score": 0.0,
                "average_jaccard_similarity": 0.0,
                "std_jaccard_similarity": 0.0,
                "num_runs": len(samples),
                "error": "Pas assez de runs réussis"
            }
        
        # Calculer similarité de Jaccard entre chaque paire de runs
        jaccard_similarities = []
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                intersection = len(samples[i] & samples[j])
                union = len(samples[i] | samples[j])
                similarity = intersection / union if union > 0 else 0.0
                jaccard_similarities.append(similarity)
        
        avg_similarity = np.mean(jaccard_similarities)
        std_similarity = np.std(jaccard_similarities)
        
        # Score de stabilité : haute similarité = haute stabilité
        stability_score = avg_similarity * 100
        
        return {
            "stability_score": round(stability_score, 2),
            "average_jaccard_similarity": round(avg_similarity, 4),
            "std_jaccard_similarity": round(std_similarity, 4),
            "min_similarity": round(float(np.min(jaccard_similarities)), 4),
            "max_similarity": round(float(np.max(jaccard_similarities)), 4),
            "num_runs": len(samples),
            "num_comparisons": len(jaccard_similarities)
        }
    
    def measure_response_time(
        self,
        operation_func,
        num_iterations: int = 10,
        **operation_params
    ) -> Dict[str, float]:
        """
        Mesure le temps de réponse d'une opération.
        
        Args:
            operation_func: Fonction à mesurer
            num_iterations: Nombre d'itérations pour la moyenne
            **operation_params: Paramètres de la fonction
            
        Returns:
            Dict avec métriques de performance
        """
        execution_times = []
        
        for _ in range(num_iterations):
            start_time = time.time()
            try:
                operation_func(**operation_params)
                end_time = time.time()
                execution_times.append(end_time - start_time)
            except Exception as e:
                logger.warning(f"Erreur lors de la mesure: {e}")
                continue
        
        if not execution_times:
            return {
                "average_time": 0.0,
                "error": "Aucune mesure réussie"
            }
        
        return {
            "average_time": round(float(np.mean(execution_times)), 4),
            "min_time": round(float(np.min(execution_times)), 4),
            "max_time": round(float(np.max(execution_times)), 4),
            "std_time": round(float(np.std(execution_times)), 4),
            "num_iterations": len(execution_times),
            "meets_target": float(np.mean(execution_times)) < 3.0  # Objectif < 2-3 sec
        }
    
    def comprehensive_evaluation(
        self,
        sampler_func=None,
        feedback_data: List[Dict] = None
    ) -> Dict[str, any]:
        """
        Effectue une évaluation complète avec toutes les métriques.
        
        Returns:
            Dict contenant toutes les métriques d'évaluation
        """
        evaluation_results = {
            "timestamp": time.time(),
            "num_patterns": len(self.patterns)
        }
        
        # 1. Taux d'acceptation
        evaluation_results["acceptance"] = self.calculate_acceptance_rate(feedback_data)
        
        # 2. Diversité (avec plusieurs méthodes)
        evaluation_results["diversity"] = {
            "jaccard": self.calculate_diversity(method="jaccard"),
            "cosine": self.calculate_diversity(method="cosine"),
            "hamming": self.calculate_diversity(method="hamming")
        }
        
        # 3. Couverture
        if self.transactions is not None:
            evaluation_results["coverage"] = self.calculate_coverage()
        else:
            evaluation_results["coverage"] = {"error": "Transactions non disponibles"}
        
        # 4. Stabilité (si sampler fourni)
        if sampler_func is not None:
            try:
                evaluation_results["stability"] = self.calculate_stability(
                    sampler_func=sampler_func,
                    num_runs=5,  # Réduit pour la performance
                    sample_size=min(50, len(self.patterns))
                )
            except Exception as e:
                evaluation_results["stability"] = {"error": str(e)}
        else:
            evaluation_results["stability"] = {"error": "Sampler non fourni"}
        
        # 5. Résumé global
        evaluation_results["summary"] = self._generate_summary(evaluation_results)
        
        return evaluation_results
    
    def _generate_summary(self, results: Dict) -> Dict[str, str]:
        """Génère un résumé textuel des résultats."""
        summary = {
            "overall_quality": "N/A",
            "recommendations": []
        }
        
        # Évaluation globale basée sur les métriques
        score = 0
        max_score = 0
        
        # Diversité (Jaccard)
        if "diversity" in results and "jaccard" in results["diversity"]:
            div_score = results["diversity"]["jaccard"].get("average_diversity", 0)
            if div_score > 0.7:
                score += 3
                summary["recommendations"].append("✅ Bonne diversité des motifs")
            elif div_score > 0.4:
                score += 2
                summary["recommendations"].append("⚠️ Diversité moyenne, considérer augmenter la pénalité de redondance")
            else:
                score += 1
                summary["recommendations"].append("❌ Faible diversité, motifs trop similaires")
            max_score += 3
        
        # Coverage
        if "coverage" in results and "coverage_rate" in results["coverage"]:
            cov_rate = results["coverage"]["coverage_rate"]
            if cov_rate > 70:
                score += 3
                summary["recommendations"].append("✅ Excellente couverture des transactions")
            elif cov_rate > 40:
                score += 2
                summary["recommendations"].append("⚠️ Couverture moyenne")
            else:
                score += 1
                summary["recommendations"].append("❌ Faible couverture, réduire le support minimum")
            max_score += 3
        
        # Acceptance
        if "acceptance" in results and results["acceptance"]["total_feedbacks"] > 0:
            acc_rate = results["acceptance"]["acceptance_rate"]
            if acc_rate > 70:
                score += 3
                summary["recommendations"].append("✅ Fort taux d'acceptation utilisateur")
            elif acc_rate > 40:
                score += 2
                summary["recommendations"].append("⚠️ Taux d'acceptation moyen")
            else:
                score += 1
                summary["recommendations"].append("❌ Faible acceptation, ajuster les poids du scoring")
            max_score += 3
        
        # Calcul score global
        if max_score > 0:
            overall_pct = (score / max_score) * 100
            if overall_pct >= 75:
                summary["overall_quality"] = "Excellente"
            elif overall_pct >= 50:
                summary["overall_quality"] = "Bonne"
            elif overall_pct >= 25:
                summary["overall_quality"] = "Moyenne"
            else:
                summary["overall_quality"] = "Faible"
        
        return summary
