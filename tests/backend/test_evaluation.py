"""
Tests unitaires pour le module d'évaluation (Partie 4)

Auteurs: Lesech Erwann, Le Riboter Aymeric, Aubron Abel, Claude Nathan
Date: Novembre 2025
"""

import pytest
import pandas as pd
import numpy as np
from backend.core.evaluation import PatternEvaluator
from backend.core.sampling import PatternSampler
import time


@pytest.fixture
def sample_patterns():
    """Génère un DataFrame de motifs pour les tests."""
    return pd.DataFrame({
        'itemsets': [
            frozenset(['A', 'B']),
            frozenset(['B', 'C']),
            frozenset(['A', 'C']),
            frozenset(['A', 'B', 'C']),
            frozenset(['D', 'E'])
        ],
        'support': [0.5, 0.4, 0.3, 0.2, 0.1],
        'length': [2, 2, 2, 3, 2]
    })


@pytest.fixture
def sample_transactions():
    """Génère un DataFrame de transactions pour les tests."""
    return pd.DataFrame({
        'A': [1, 1, 0, 1, 0],
        'B': [1, 1, 1, 1, 0],
        'C': [0, 1, 1, 1, 0],
        'D': [0, 0, 0, 0, 1],
        'E': [0, 0, 0, 0, 1]
    })


@pytest.fixture
def sample_feedback():
    """Génère des feedbacks de test."""
    return [
        {"pattern_id": 0, "rating": 1},
        {"pattern_id": 1, "rating": 1},
        {"pattern_id": 2, "rating": -1},
        {"pattern_id": 3, "rating": 1},
        {"pattern_id": 4, "rating": 0}
    ]


class TestPatternEvaluator:
    """Tests pour la classe PatternEvaluator"""
    
    def test_initialization(self, sample_patterns, sample_transactions):
        """Test l'initialisation de l'évaluateur."""
        evaluator = PatternEvaluator(sample_patterns, sample_transactions)
        assert evaluator.patterns is not None
        assert evaluator.transactions is not None
        assert len(evaluator.feedback_history) == 0
    
    def test_acceptance_rate_no_feedback(self, sample_patterns):
        """Test le calcul du taux d'acceptation sans feedback."""
        evaluator = PatternEvaluator(sample_patterns)
        result = evaluator.calculate_acceptance_rate()
        
        assert result['acceptance_rate'] == 0.0
        assert result['total_feedbacks'] == 0
        assert result['likes'] == 0
        assert result['dislikes'] == 0
    
    def test_acceptance_rate_with_feedback(self, sample_patterns, sample_feedback):
        """Test le calcul du taux d'acceptation avec feedback."""
        evaluator = PatternEvaluator(sample_patterns)
        result = evaluator.calculate_acceptance_rate(sample_feedback)
        
        assert result['total_feedbacks'] == 5
        assert result['likes'] == 3
        assert result['dislikes'] == 1
        assert result['acceptance_rate'] == 60.0  # 3/5 * 100
    
    def test_add_feedback(self, sample_patterns):
        """Test l'ajout de feedback."""
        evaluator = PatternEvaluator(sample_patterns)
        
        evaluator.add_feedback(0, 1, "Très bon motif")
        assert len(evaluator.feedback_history) == 1
        assert evaluator.feedback_history[0]['pattern_id'] == 0
        assert evaluator.feedback_history[0]['rating'] == 1
        assert evaluator.feedback_history[0]['comment'] == "Très bon motif"
    
    def test_diversity_jaccard(self, sample_patterns):
        """Test le calcul de diversité avec Jaccard."""
        evaluator = PatternEvaluator(sample_patterns)
        result = evaluator.calculate_diversity(method="jaccard")
        
        assert 'average_diversity' in result
        assert 'min_diversity' in result
        assert 'max_diversity' in result
        assert 'std_diversity' in result
        assert result['method'] == 'jaccard'
        assert result['num_patterns'] == 5
        assert 0 <= result['average_diversity'] <= 1
    
    def test_diversity_cosine(self, sample_patterns):
        """Test le calcul de diversité avec Cosine."""
        evaluator = PatternEvaluator(sample_patterns)
        result = evaluator.calculate_diversity(method="cosine")
        
        assert result['method'] == 'cosine'
        assert 0 <= result['average_diversity'] <= 1
    
    def test_diversity_hamming(self, sample_patterns):
        """Test le calcul de diversité avec Hamming."""
        evaluator = PatternEvaluator(sample_patterns)
        result = evaluator.calculate_diversity(method="hamming")
        
        assert result['method'] == 'hamming'
        assert 0 <= result['average_diversity'] <= 1
    
    def test_diversity_single_pattern(self):
        """Test la diversité avec un seul motif."""
        patterns = pd.DataFrame({
            'itemsets': [frozenset(['A', 'B'])],
            'support': [0.5]
        })
        evaluator = PatternEvaluator(patterns)
        result = evaluator.calculate_diversity()
        
        assert result['num_patterns'] == 1
        assert result['average_diversity'] == 0.0
    
    def test_coverage_without_transactions(self, sample_patterns):
        """Test la couverture sans transactions."""
        evaluator = PatternEvaluator(sample_patterns)
        result = evaluator.calculate_coverage()
        
        assert 'error' in result
    
    def test_coverage_with_transactions(self, sample_patterns, sample_transactions):
        """Test le calcul de couverture avec transactions."""
        evaluator = PatternEvaluator(sample_patterns, sample_transactions)
        result = evaluator.calculate_coverage()
        
        if 'error' not in result:
            assert 'coverage_rate' in result
            assert 'covered_transactions' in result
            assert 'total_transactions' in result
            assert 0 <= result['coverage_rate'] <= 100
    
    def test_stability_calculation(self, sample_patterns):
        """Test le calcul de stabilité."""
        evaluator = PatternEvaluator(sample_patterns)
        
        # Fonction d'échantillonnage simulée
        def mock_sampler(k=3):
            return [(i, sample_patterns.iloc[i]['itemsets']) 
                   for i in np.random.choice(len(sample_patterns), k, replace=False)]
        
        result = evaluator.calculate_stability(
            sampler_func=mock_sampler,
            num_runs=3,
            sample_size=3
        )
        
        assert 'stability_score' in result
        assert 'average_jaccard_similarity' in result
        assert 'num_runs' in result
        assert 0 <= result['stability_score'] <= 100
    
    def test_performance_measurement(self, sample_patterns):
        """Test la mesure de performance."""
        evaluator = PatternEvaluator(sample_patterns)
        
        # Fonction de test simple
        def test_operation():
            time.sleep(0.01)  # 10ms
            return True
        
        result = evaluator.measure_response_time(
            operation_func=test_operation,
            num_iterations=3
        )
        
        assert 'average_time' in result
        assert 'min_time' in result
        assert 'max_time' in result
        assert 'meets_target' in result
        assert result['num_iterations'] == 3
        assert result['average_time'] > 0
    
    def test_comprehensive_evaluation(self, sample_patterns, sample_transactions, sample_feedback):
        """Test l'évaluation complète."""
        evaluator = PatternEvaluator(sample_patterns, sample_transactions)
        
        result = evaluator.comprehensive_evaluation(feedback_data=sample_feedback)
        
        # Vérifier la structure
        assert 'timestamp' in result
        assert 'num_patterns' in result
        assert 'acceptance' in result
        assert 'diversity' in result
        assert 'coverage' in result
        assert 'summary' in result
        
        # Vérifier les métriques de diversité
        assert 'jaccard' in result['diversity']
        assert 'cosine' in result['diversity']
        assert 'hamming' in result['diversity']
        
        # Vérifier le résumé
        assert 'overall_quality' in result['summary']
        assert 'recommendations' in result['summary']
        assert isinstance(result['summary']['recommendations'], list)


class TestPatternSamplerFeedback:
    """Tests pour le système de feedback dans PatternSampler"""
    
    def test_feedback_history_initialization(self, sample_patterns):
        """Test l'initialisation de l'historique de feedback."""
        sampler = PatternSampler(sample_patterns)
        assert hasattr(sampler, 'feedback_history')
        assert len(sampler.feedback_history) == 0
    
    def test_user_feedback_like(self, sample_patterns):
        """Test l'application d'un like."""
        # Préparer les patterns avec composite_score
        sample_patterns['composite_score'] = 0.2
        sampler = PatternSampler(sample_patterns)
        
        initial_score = sampler.patterns.loc[0, 'composite_score']
        sampler.user_feedback(index=0, alpha=1.0, beta=1.0, rating=1)
        final_score = sampler.patterns.loc[0, 'composite_score']
        
        assert len(sampler.feedback_history) == 1
        assert sampler.feedback_history[0]['rating'] == 1
        # Le score devrait augmenter (après normalisation)
    
    def test_user_feedback_dislike(self, sample_patterns):
        """Test l'application d'un dislike."""
        sample_patterns['composite_score'] = 0.2
        sampler = PatternSampler(sample_patterns)
        
        sampler.user_feedback(index=0, alpha=1.0, beta=1.0, rating=-1)
        
        assert len(sampler.feedback_history) == 1
        assert sampler.feedback_history[0]['rating'] == -1
    
    def test_get_feedback_stats_empty(self, sample_patterns):
        """Test les statistiques sans feedback."""
        sampler = PatternSampler(sample_patterns)
        stats = sampler.get_feedback_stats()
        
        assert stats['total_feedbacks'] == 0
        assert stats['likes'] == 0
        assert stats['dislikes'] == 0
        assert stats['acceptance_rate'] == 0.0
    
    def test_get_feedback_stats_with_data(self, sample_patterns):
        """Test les statistiques avec des feedbacks."""
        sample_patterns['composite_score'] = 0.2
        sampler = PatternSampler(sample_patterns)
        
        # Ajouter plusieurs feedbacks
        sampler.user_feedback(0, 1.0, 1.0, 1)   # like
        sampler.user_feedback(1, 1.0, 1.0, 1)   # like
        sampler.user_feedback(2, 1.0, 1.0, -1)  # dislike
        sampler.user_feedback(3, 1.0, 1.0, 0)   # neutral
        
        stats = sampler.get_feedback_stats()
        
        assert stats['total_feedbacks'] == 4
        assert stats['likes'] == 2
        assert stats['dislikes'] == 1
        assert stats['neutral'] == 1
        assert stats['acceptance_rate'] == 50.0
    
    def test_export_feedback_history_empty(self, sample_patterns):
        """Test l'export d'un historique vide."""
        sampler = PatternSampler(sample_patterns)
        df = sampler.export_feedback_history()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
    
    def test_export_feedback_history_with_data(self, sample_patterns):
        """Test l'export avec des données."""
        sample_patterns['composite_score'] = 0.2
        sampler = PatternSampler(sample_patterns)
        
        sampler.user_feedback(0, 1.0, 1.0, 1)
        sampler.user_feedback(1, 1.0, 1.0, -1)
        
        df = sampler.export_feedback_history()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'pattern_index' in df.columns
        assert 'rating' in df.columns
        assert 'timestamp' in df.columns


class TestEvaluationMetrics:
    """Tests pour les métriques spécifiques"""
    
    def test_jaccard_distance_identical_sets(self):
        """Test la distance de Jaccard pour des ensembles identiques."""
        patterns = pd.DataFrame({
            'itemsets': [frozenset(['A', 'B']), frozenset(['A', 'B'])]
        })
        evaluator = PatternEvaluator(patterns)
        result = evaluator.calculate_diversity(method="jaccard")
        
        # Distance entre ensembles identiques devrait être 0
        assert result['min_diversity'] == 0.0
    
    def test_jaccard_distance_disjoint_sets(self):
        """Test la distance de Jaccard pour des ensembles disjoints."""
        patterns = pd.DataFrame({
            'itemsets': [frozenset(['A', 'B']), frozenset(['C', 'D'])]
        })
        evaluator = PatternEvaluator(patterns)
        result = evaluator.calculate_diversity(method="jaccard")
        
        # Distance entre ensembles disjoints devrait être 1
        assert result['max_diversity'] == 1.0
    
    def test_acceptance_rate_all_likes(self):
        """Test le taux d'acceptation avec 100% de likes."""
        patterns = pd.DataFrame({'itemsets': [frozenset(['A'])]})
        evaluator = PatternEvaluator(patterns)
        
        feedback = [{"pattern_id": 0, "rating": 1} for _ in range(10)]
        result = evaluator.calculate_acceptance_rate(feedback)
        
        assert result['acceptance_rate'] == 100.0
    
    def test_acceptance_rate_all_dislikes(self):
        """Test le taux d'acceptation avec 100% de dislikes."""
        patterns = pd.DataFrame({'itemsets': [frozenset(['A'])]})
        evaluator = PatternEvaluator(patterns)
        
        feedback = [{"pattern_id": 0, "rating": -1} for _ in range(10)]
        result = evaluator.calculate_acceptance_rate(feedback)
        
        assert result['acceptance_rate'] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
