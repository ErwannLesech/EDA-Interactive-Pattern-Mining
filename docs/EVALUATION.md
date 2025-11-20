# Module d'Évaluation et Reproductibilité

Ce module implémente les métriques d'évaluation demandées dans le point 4 du projet EDA.

## 📊 Métriques Implémentées

### 1. Taux d'Acceptation (via feedback)
- **Calcul** : Proportion de motifs "likés" par rapport au total des feedbacks
- **Formule** : `likes / (likes + dislikes + neutral)`
- **Interprétation** : Un taux élevé (>70%) indique que les motifs sont pertinents

### 2. Diversité
- **Calcul** : Distance de Jaccard moyenne entre tous les paires de motifs
- **Formule** : `1 - (intersection / union)` pour chaque paire
- **Interprétation** : Plus c'est élevé (>0.7), plus les motifs sont différents

### 3. Couverture (Coverage)
Trois aspects de couverture sont mesurés :
- **Couverture motifs** : `nb_motifs_échantillonnés / nb_motifs_total`
- **Couverture items** : `nb_items_uniques_échantillon / nb_items_uniques_total`
- **Couverture support** : `somme_supports_échantillon / somme_supports_total`

### 4. Stabilité (Sensibilité à la seed)
- **Calcul** : Similarité de Jaccard moyenne entre échantillons avec différentes seeds
- **Méthode** : 10 échantillonnages avec seeds différentes (42, 43, 44, ...)
- **Interprétation** : Plus c'est élevé (>0.7), plus l'algorithme est stable

### 5. Temps de Réponse
- **Mesures** : Temps moyen, min, max, écart-type sur 5 exécutions
- **Objectif** : < 2-3 secondes pour une expérience interactive
- **Interprétation** : Performance critique pour l'UX

## 🏗️ Architecture

### Backend

#### `backend/core/evaluation.py`
Classe `PatternEvaluator` avec méthodes :
- `calculate_acceptance_rate(feedback_list)` : Taux d'acceptation
- `calculate_diversity(patterns_df)` : Diversité des motifs
- `calculate_coverage(sampled, all_patterns)` : Couverture
- `calculate_stability(func, patterns, params)` : Stabilité
- `measure_response_time(func, patterns, params)` : Performance
- `comprehensive_evaluation(...)` : Évaluation complète

#### `backend/core/sampling.py`
Modifications :
- Ajout de `feedback_history` pour tracker les feedbacks
- Mise à jour de `user_feedback()` pour enregistrer l'historique

#### `backend/api/routes.py`
Nouveau endpoint :
- `GET /api/patterns/evaluate` : Retourne toutes les métriques d'évaluation

### Frontend

#### `frontend/components/evaluation.py`
Composants de visualisation :
- `display_evaluation_metrics(data)` : Affichage complet avec graphiques
- `display_evaluation_summary(data)` : Résumé compact

#### `frontend/app.py`
- Tab 4 "📊 Analyse" transformé en "📊 Évaluation & Reproductibilité"
- Intégration des composants d'évaluation
- Boutons d'export CSV et JSON

## 📈 Utilisation

### Via l'Interface Web

1. **Charger un dataset** (Tab "Upload")
2. **Extraire les motifs** (Tab "Motifs")
3. **Donner des feedbacks** sur les motifs échantillonnés (👍/👎)
4. **Lancer l'évaluation** (Tab "Évaluation")
   - Cliquer sur "🚀 Évaluer"
   - Consulter les métriques dans les 5 onglets
   - Exporter les résultats en CSV ou JSON

### Via l'API

```python
import requests

# Lancer l'évaluation
response = requests.get("http://backend:8000/api/patterns/evaluate")
evaluation = response.json()

print(f"Score global: {evaluation['evaluation']['overall_score']:.2%}")
print(f"Taux d'acceptation: {evaluation['evaluation']['acceptance']['acceptance_rate']:.2%}")
print(f"Diversité: {evaluation['evaluation']['diversity']['diversity_score']:.3f}")
print(f"Couverture: {evaluation['evaluation']['coverage']['pattern_coverage']:.2%}")
print(f"Stabilité: {evaluation['evaluation']['stability']['stability_score']:.3f}")
print(f"Temps moyen: {evaluation['evaluation']['response_time']['mean_time']:.3f}s")
```

## 🎯 Score Global

Le score global est une moyenne pondérée des métriques :
```
Score = 0.30 × Acceptation + 0.25 × Diversité + 0.25 × Couverture + 0.20 × Stabilité
```

## 📊 Visualisations

L'interface propose plusieurs types de visualisations :
- **Jauge** : Score global et diversité
- **Barres** : Distribution des feedbacks, couverture
- **Histogramme** : Distribution des similarités (stabilité)
- **Indicateurs** : Temps de réponse

## 🔄 Reproductibilité

### Stabilité de l'échantillonnage
Le module teste la reproductibilité en :
1. Exécutant l'algorithme 10 fois avec des seeds différentes
2. Calculant la similarité de Jaccard entre chaque paire d'échantillons
3. Moyennant ces similarités pour obtenir le score de stabilité

### Export des résultats
Deux formats d'export :
- **CSV** : Métriques principales pour analyse dans Excel/R/Python
- **JSON** : Rapport complet avec tous les détails

## ⚡ Performance

### Optimisations implémentées
- Vectorisation numpy pour calculs de diversité et redondance
- Limitation des comparaisons pour la redondance (patterns de taille similaire)
- Cache des résultats intermédiaires

### Temps de réponse typiques
- Dataset < 1000 motifs : < 1s
- Dataset 1000-5000 motifs : 1-3s
- Dataset > 5000 motifs : 3-10s

## 🧪 Exemple de Résultat

```json
{
  "evaluation": {
    "acceptance": {
      "acceptance_rate": 0.75,
      "total_feedbacks": 20,
      "likes": 15,
      "dislikes": 3,
      "neutral": 2
    },
    "diversity": {
      "diversity_score": 0.682,
      "unique_items_count": 45,
      "average_pattern_length": 3.2
    },
    "coverage": {
      "pattern_coverage": 0.10,
      "item_coverage": 0.85,
      "support_coverage": 0.62
    },
    "stability": {
      "stability_score": 0.721,
      "mean_similarity": 0.721,
      "std_similarity": 0.089
    },
    "response_time": {
      "mean_time": 1.234,
      "std_time": 0.056,
      "min_time": 1.189,
      "max_time": 1.312
    },
    "overall_score": 0.698
  }
}
```

## 📝 Notes d'Implémentation

### Choix de design
1. **Séparation des responsabilités** : Évaluation dans module dédié
2. **Flexibilité** : Méthodes individuelles + évaluation complète
3. **Visualisation** : Graphiques interactifs Plotly
4. **Export** : Formats multiples pour différents usages

### Limitations connues
1. Stabilité : Limitée à 10 itérations pour performance
2. Temps de réponse : Mesuré sur 5 runs (peut varier selon charge système)
3. Acceptance rate : Nécessite des feedbacks utilisateur

### Extensions possibles
- Sauvegarder l'historique des évaluations
- Comparer différentes stratégies d'échantillonnage
- Tests statistiques (t-test) pour la stabilité
- Métriques additionnelles (nouveauté, surprise globale)

## 🔗 Références

- **Diversité** : Jaccard Distance pour similarité de sets
- **Stabilité** : Approche Monte Carlo avec seeds multiples
- **Performance** : Benchmarking avec répétitions

## 👥 Auteurs

Module d'évaluation implémenté pour le projet EDA - SCIA-G
