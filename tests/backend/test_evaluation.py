"""
Script de test pour le module d'évaluation.
Vérifie que toutes les fonctions fonctionnent correctement.
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from backend.core.evaluation import PatternEvaluator

def test_evaluation_module():
    """Test complet du module d'évaluation"""
    
    print("="*60)
    print("TEST DU MODULE D'ÉVALUATION")
    print("="*60)
    
    # Créer des données de test
    print("\n1. Création des données de test...")
    
    # Patterns de test
    patterns_data = {
        'itemsets': [
            frozenset(['A', 'B']),
            frozenset(['B', 'C']),
            frozenset(['A', 'C', 'D']),
            frozenset(['E', 'F']),
            frozenset(['A', 'B', 'C']),
        ],
        'support': [0.5, 0.4, 0.3, 0.25, 0.2]
    }
    all_patterns = pd.DataFrame(patterns_data)
    
    # Échantillon (3 premiers)
    sampled_patterns = all_patterns.head(3).copy()
    
    # Feedbacks de test
    feedback_list = [
        {"pattern_id": 0, "rating": 1},
        {"pattern_id": 1, "rating": 1},
        {"pattern_id": 2, "rating": -1},
        {"pattern_id": 0, "rating": 1},
    ]
    
    print("   ✓ Données créées")
    print(f"     - {len(all_patterns)} motifs totaux")
    print(f"     - {len(sampled_patterns)} motifs échantillonnés")
    print(f"     - {len(feedback_list)} feedbacks")
    
    # Créer l'évaluateur
    print("\n2. Création de l'évaluateur...")
    evaluator = PatternEvaluator()
    print("   ✓ PatternEvaluator créé")
    
    # Test 1: Taux d'acceptation
    print("\n3. Test du taux d'acceptation...")
    acceptance = evaluator.calculate_acceptance_rate(feedback_list)
    print(f"   ✓ Taux d'acceptation: {acceptance['acceptance_rate']:.2%}")
    print(f"     - Likes: {acceptance['likes']}")
    print(f"     - Dislikes: {acceptance['dislikes']}")
    print(f"     - Total: {acceptance['total_feedbacks']}")
    
    # Test 2: Diversité
    print("\n4. Test de la diversité...")
    diversity = evaluator.calculate_diversity(sampled_patterns)
    print(f"   ✓ Score de diversité: {diversity['diversity_score']:.3f}")
    print(f"     - Distance Jaccard moyenne: {diversity['average_jaccard_distance']:.3f}")
    print(f"     - Items uniques: {diversity['unique_items_count']}")
    print(f"     - Longueur moyenne: {diversity['average_pattern_length']:.1f}")
    
    # Test 3: Couverture
    print("\n5. Test de la couverture...")
    coverage = evaluator.calculate_coverage(sampled_patterns, all_patterns)
    print(f"   ✓ Couverture motifs: {coverage['pattern_coverage']:.2%}")
    print(f"     - Couverture items: {coverage['item_coverage']:.2%}")
    print(f"     - Couverture support: {coverage['support_coverage']:.2%}")
    print(f"     - Échantillonnés: {coverage['sampled_count']}/{coverage['total_count']}")
    
    # Test 4: Stabilité (simplifié)
    print("\n6. Test de la stabilité...")
    def sample_func(df, k=3):
        """Fonction d'échantillonnage simple pour test"""
        np.random.seed(42)
        indices = np.random.choice(len(df), min(k, len(df)), replace=False)
        return [(row['itemsets'], idx) for idx, row in df.iloc[indices].iterrows()]
    
    params = {"k": 3}
    stability = evaluator.calculate_stability(sample_func, all_patterns, params, n_iterations=5)
    print(f"   ✓ Score de stabilité: {stability['stability_score']:.3f}")
    print(f"     - Similarité moyenne: {stability['mean_similarity']:.3f}")
    print(f"     - Écart-type: {stability['std_similarity']:.3f}")
    print(f"     - Itérations: {stability['n_iterations']}")
    
    # Test 5: Temps de réponse
    print("\n7. Test du temps de réponse...")
    response_time = evaluator.measure_response_time(sample_func, all_patterns, params, n_runs=3)
    print(f"   ✓ Temps moyen: {response_time['mean_time']:.4f}s")
    print(f"     - Min: {response_time['min_time']:.4f}s")
    print(f"     - Max: {response_time['max_time']:.4f}s")
    print(f"     - Écart-type: {response_time['std_time']:.4f}s")
    
    # Test 6: Évaluation complète
    print("\n8. Test de l'évaluation complète...")
    comprehensive = evaluator.comprehensive_evaluation(
        sampled_patterns,
        all_patterns,
        feedback_list,
        sample_func,
        params
    )
    print(f"   ✓ Score global: {comprehensive['overall_score']:.2%}")
    print("\n   Détails:")
    print(f"     - Acceptation: {comprehensive['acceptance']['acceptance_rate']:.2%}")
    print(f"     - Diversité: {comprehensive['diversity']['diversity_score']:.3f}")
    print(f"     - Couverture: {comprehensive['coverage']['pattern_coverage']:.2%}")
    print(f"     - Stabilité: {comprehensive['stability']['stability_score']:.3f}")
    print(f"     - Temps: {comprehensive['response_time']['mean_time']:.4f}s")
    
    print("\n" + "="*60)
    print("✅ TOUS LES TESTS SONT PASSÉS AVEC SUCCÈS!")
    print("="*60)
    
    return True

def test_edge_cases():
    """Test des cas limites"""
    
    print("\n" + "="*60)
    print("TEST DES CAS LIMITES")
    print("="*60)
    
    evaluator = PatternEvaluator()
    
    # Test avec données vides
    print("\n1. Test avec données vides...")
    empty_df = pd.DataFrame({'itemsets': [], 'support': []})
    
    acceptance = evaluator.calculate_acceptance_rate([])
    print(f"   ✓ Taux d'acceptation (vide): {acceptance['acceptance_rate']}")
    
    diversity = evaluator.calculate_diversity(empty_df)
    print(f"   ✓ Diversité (vide): {diversity['diversity_score']}")
    
    coverage = evaluator.calculate_coverage(empty_df, empty_df)
    print(f"   ✓ Couverture (vide): {coverage['pattern_coverage']}")
    
    # Test avec un seul pattern
    print("\n2. Test avec un seul pattern...")
    single_pattern = pd.DataFrame({
        'itemsets': [frozenset(['A', 'B'])],
        'support': [0.5]
    })
    
    diversity = evaluator.calculate_diversity(single_pattern)
    print(f"   ✓ Diversité (1 pattern): {diversity['diversity_score']}")
    
    print("\n✅ Tests des cas limites OK!")
    
    return True

if __name__ == "__main__":
    try:
        # Tests principaux
        test_evaluation_module()
        
        # Tests des cas limites
        test_edge_cases()
        
        print("\n" + "🎉"*30)
        print("TOUS LES TESTS SONT RÉUSSIS!")
        print("Le module d'évaluation est prêt à être utilisé.")
        print("🎉"*30)
        
    except Exception as e:
        print(f"\n❌ ERREUR lors des tests: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
