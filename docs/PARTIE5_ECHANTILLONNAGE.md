# Partie 5 : Échantillonnage de Sortie de Motifs

## 📚 Vue d'ensemble

Cette partie implémente trois méthodes d'échantillonnage de motifs permettant de sélectionner un sous-ensemble représentatif et intéressant de motifs extraits.

## 🎯 Méthodes implémentées

### 1. **Importance Sampling**
Échantillonne les motifs selon un score composite basé sur :
- **Support** : Fréquence du motif dans le dataset
- **Surprise** : Écart entre support observé et support attendu (modèle d'indépendance)
- **Redondance** : Similarité avec les autres motifs (pénalise les motifs similaires)

**Formule** :  
```
Score = w1 × support + w2 × surprise + w3 × (1 - redondance)
```

**Avantages** :
- Contrôle fin via les poids
- Diversité des motifs grâce à la pénalité de redondance
- Feedback utilisateur intégré

### 2. **TwoStep Pattern Sampling** (Boley et al., KDD'2011)
Échantillonne en deux étapes :
1. Sélection d'une transaction (pondérée par 2^taille)
2. Échantillonnage d'un sous-ensemble de cette transaction

**Utilisation** : Classification, binarisation de données

### 3. **GDPS** (Generic Direct Pattern Sampling - Diop et al., KAIS 2019)
Échantillonnage direct avec différentes fonctions d'utilité :
- **freq** : Uniforme (tous les motifs équiprobables)
- **area** : Proportionnel à la taille
- **decay** : Décroissance exponentielle (e^(-taille))

**Paramètres** :
- `min_norm` : Taille minimale des motifs
- `max_norm` : Taille maximale des motifs
- `utility` : Type de fonction d'utilité

## 🔧 Utilisation

### Backend (API)

```python
from core.sampling import PatternSampler

# Importance Sampling
sampler = PatternSampler(patterns_df)
sampled = sampler.importance_sampling(
    support_weight=0.33,
    surprise_weight=0.33,
    redundancy_weight=0.34,
    k=50,
    replacement=True
)

# TwoStep Sampling
sampled = sampler.twostep_sampling(transactions, k=100)

# GDPS
sampled = sampler.gdps_sampling(
    transactions, 
    k=50, 
    min_norm=2, 
    max_norm=10, 
    utility="area"
)
```

### Frontend (Interface utilisateur)

```python
from components.sampling import sampling_tab

# Dans votre app Streamlit
sampling_tab(backend_url, dataset_id)
```

## 📡 Endpoints API

### POST `/api/sample/importance`
**Paramètres** :
- `dataset_id` : ID du dataset
- `k` : Nombre de motifs
- `support_weight` : Poids du support (0-1)
- `surprise_weight` : Poids de la surprise (0-1)
- `redundancy_weight` : Poids anti-redondance (0-1)
- `replacement` : Avec/sans remise (boolean)

### POST `/api/sample/twostep`
**Paramètres** :
- `dataset_id` : ID du dataset
- `k` : Nombre de motifs

### POST `/api/sample/gdps`
**Paramètres** :
- `dataset_id` : ID du dataset
- `k` : Nombre de motifs
- `min_norm` : Taille minimale (défaut: 1)
- `max_norm` : Taille maximale (défaut: 10)
- `utility` : Type d'utilité ("freq", "area", "decay")

### POST `/api/feedback`
**Paramètres** :
- `pattern_index` : Index du motif
- `rating` : Note (1=like, 0=dislike)
- `alpha` : Param feedback positif (défaut: 0.1)
- `beta` : Param feedback négatif (défaut: 0.1)

## 🎨 Interface utilisateur

L'onglet **Échantillonnage** propose :
1. **Sélection de la méthode** : Dropdown pour choisir la méthode
2. **Configuration des paramètres** : Sliders et inputs pour ajuster
3. **Visualisation des résultats** :
   - Tableau des motifs échantillonnés
   - Distribution des tailles
   - Export CSV
4. **Feedback utilisateur** : Boutons Like/Dislike

## 📊 Exemple de workflow

1. **Upload** : Charger un dataset dans l'onglet Upload
2. **Extraction** : Extraire les motifs (onglet Motifs)
3. **Échantillonnage** : Aller dans l'onglet Échantillonnage
   - Choisir une méthode
   - Ajuster les paramètres
   - Lancer l'échantillonnage
4. **Feedback** : Noter les motifs pour affiner
5. **Analyse** : Visualiser les distributions dans l'onglet Analyse

## 📈 Comparaison des méthodes

| Méthode | Avantages | Inconvénients | Use case |
|---------|-----------|---------------|----------|
| **Importance Sampling** | Contrôle fin, diversité, feedback | Calcul coûteux | Exploration interactive |
| **TwoStep** | Rapide, simple | Peu de contrôle | Classification, preprocessing |
| **GDPS** | Flexible (utilités), tailles contrôlées | Paramètres à ajuster | Analyse par taille |

## 🔬 Métriques calculées

### Surprise
```
surprise = |support_observé - support_attendu| / support_attendu
```
Où `support_attendu = ∏ support(item)` (modèle d'indépendance)

### Redondance
```
redondance = moyenne(Jaccard(motif, autres_motifs))
```
Où `Jaccard(A,B) = |A ∩ B| / |A ∪ B|`

## 📚 Références

- **TwoStep** : Boley et al., "One Click Mining: Interactive Local Pattern Discovery through Implicit Preference and Performance Learning", KDD'2011
- **GDPS** : Diop et al., "Pattern Sampling in Distributed Databases", Knowledge and Information Systems, 2019

## 🐛 Points d'attention

1. **Performance** : Importance Sampling peut être lent sur de gros datasets
2. **Mémoire** : Stocker tous les motifs peut être coûteux
3. **État** : Le feedback nécessite de maintenir l'état du sampler
4. **Format** : Adapter le format des transactions selon votre implémentation

## 🚀 Prochaines étapes

- [ ] Implémenter le cache pour les scores
- [ ] Ajouter plus de visualisations comparatives
- [ ] Sauvegarder les feedbacks en base de données
- [ ] Paralléliser les calculs pour de gros datasets
- [ ] Ajouter des tests unitaires
