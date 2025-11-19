# Exemples de Datasets

Ce dossier contient des exemples de datasets dans différents formats pour tester l'upload et la normalisation.

## Types de datasets

### 1. Transactionnel (Standard)

Format : `transaction_id, items`

**Fichiers disponibles :**
- `transactional_exemple.csv` - Séparateur virgule (`,`)
- `transactional_semicolon.csv` - Séparateur point-virgule (`;`)
- `transactional_tab.csv` - Séparateur tabulation (`\t`)
- `transactional_pipe.csv` - Séparateur pipe (`|`)

**Exemple :**
```csv
transaction_id,items
1,"bread,milk,eggs"
2,"bread,butter"
3,"milk,butter,cheese"
```

**Normalisation :** Aucune transformation nécessaire, déjà au format standard.

---

### 2. Transactionnel inversé (Inversed)

Format : Transposé - première ligne = IDs de transaction, deuxième ligne = items

**Fichiers disponibles :**
- `inversed_transactional_exemple.csv` - Séparateur virgule (`,`)

**Exemple :**
```csv
transaction_id, 1, 2, 3, 4
items, "bread,milk,eggs", "bread,butter", "milk,butter,cheese", "bread,milk,butter"
```

**Normalisation :** Le fichier est transposé pour obtenir le format standard.

**Résultat après normalisation :**
```csv
transaction_id,items
1,"bread,milk,eggs"
2,"bread,butter"
3,"milk,butter,cheese"
4,"bread,milk,butter"
```

---

### 3. Séquentiel (Sequential)

Format : `sequence_id, position, item`

Chaque ligne représente un item dans une séquence. L'ordre est préservé via la colonne `position`.

**Fichiers disponibles :**
- `sequential_example.csv` - Séparateur virgule (`,`)
- `sequential_tab.csv` - Séparateur tabulation (`\t`)

**Exemple :**
```csv
sequence_id,position,item
0,0,home
0,1,products
0,2,cart
0,3,checkout
1,0,home
1,1,products
```

**Normalisation :** Les items sont groupés par `sequence_id` et triés par `position` pour préserver l'ordre.

**Résultat après normalisation :**
```csv
transaction_id,items
0,"home,products,cart,checkout"
1,"home,products"
```

**Important :** L'ordre des items est crucial pour le pattern mining séquentiel (PrefixSpan, GSP).

---

### 4. Matricielle (Matrix)

Format : `Transaction_ID, item1, item2, item3, ...`

Chaque colonne (sauf la première) représente un item. Les valeurs sont binaires (0/1).

**Fichiers disponibles :**
- `matrix_example.csv` - Séparateur virgule (`,`)
- `matrix_semicolon.csv` - Séparateur point-virgule (`;`)

**Exemple :**
```csv
Transaction_ID,pain,lait,beurre,fromage,œufs
T1,1,1,1,0,0
T2,1,0,1,0,0
T3,0,1,0,1,1
```

**Normalisation :** Conversion binaire → liste d'items.

**Résultat après normalisation :**
```csv
transaction_id,items
T1,"pain,lait,beurre"
T2,"pain,beurre"
T3,"lait,fromage,œufs"
```

---

## Séparateurs supportés

| Séparateur | Symbole | Exemples de fichiers |
|------------|---------|----------------------|
| Virgule | `,` | `transactional_exemple.csv`, `sequential_example.csv`, `matrix_example.csv` |
| Point-virgule | `;` | `transactional_semicolon.csv`, `matrix_semicolon.csv` |
| Tabulation | `\t` | `transactional_tab.csv`, `sequential_tab.csv` |
| Pipe | `|` | `transactional_pipe.csv` |

Le séparateur est **détecté automatiquement** par le backend lors de l'upload. Vous pouvez néanmoins le modifier manuellement dans l'interface.

---

## Détection automatique

L'application détecte automatiquement :

1. **Le séparateur** (via l'endpoint `/api/detect-separator`)
   - Analyse les premières lignes du fichier
   - Teste les séparateurs courants : `;`, `\t`, `|`, `,`, ` `
   - Privilégie les séparateurs rares (`;`, `\t`) pour éviter les faux positifs

2. **Le type de dataset** (heuristique locale dans le frontend)
   - **Inversed** : Détecte si la première colonne contient "transaction_id" ou "items" et si la première ligne contient "items"
   - **Matrix** : Détecte si plusieurs colonnes contiennent uniquement des valeurs 0/1
   - **Sequential** : Détecte la présence d'une colonne "position", "sequence", "step" ou "order"
   - **Transactional** : Par défaut si aucun autre type n'est détecté

---

## Test des exemples

Pour tester l'upload avec tous les exemples :

```bash
# Démarrer le backend et frontend
docker-compose up

# Aller sur http://localhost:8501
# Uploader chaque fichier et vérifier :
# - Le séparateur détecté est correct
# - Le type détecté est correct
# - L'aperçu des données est cohérent
```

**Tests recommandés :**

1. ✅ `transactional_exemple.csv` → Type: Transactionnel, Séparateur: `,`
2. ✅ `transactional_semicolon.csv` → Type: Transactionnel, Séparateur: `;`
3. ✅ `transactional_tab.csv` → Type: Transactionnel, Séparateur: `\t`
4. ✅ `transactional_pipe.csv` → Type: Transactionnel, Séparateur: `|`
5. ✅ `inversed_transactional_exemple.csv` → Type: Inversé, Séparateur: `,`
6. ✅ `sequential_example.csv` → Type: Séquentiel, Séparateur: `,`
7. ✅ `sequential_tab.csv` → Type: Séquentiel, Séparateur: `\t`
8. ✅ `matrix_example.csv` → Type: Matricielle, Séparateur: `,`
9. ✅ `matrix_semicolon.csv` → Type: Matricielle, Séparateur: `;`

---

## Création de nouveaux exemples

Pour créer un nouveau dataset d'exemple :

1. **Choisir le type** : transactional, inversed, sequential, ou matrix
2. **Choisir le séparateur** : `,`, `;`, `\t`, ou `|`
3. **Respecter le format** documenté ci-dessus
4. **Tester l'upload** pour vérifier la détection automatique

**Bonnes pratiques :**
- Utiliser au moins 5-10 transactions pour tester les algorithmes
- Inclure des variations (transactions courtes et longues)
- Vérifier que le fichier est encodé en UTF-8
