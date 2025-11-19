#!/usr/bin/env python3
"""
Script de test pour vérifier la normalisation des différents types de datasets.
Lance ce script depuis le dossier backend pour tester tous les exemples.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../backend'))

from utils.data_processing import detect_separator, read_file_to_dataframe, detect_dataset_type, normalize_to_transactional_format
import pandas as pd

def test_file(filepath: str):
    """Test l'upload et la normalisation d'un fichier"""
    print(f"\n{'='*80}")
    print(f"Test: {os.path.basename(filepath)}")
    print(f"{'='*80}")
    
    try:
        # Lire le fichier
        with open(filepath, 'rb') as f:
            content = f.read()
        
        # Détecter le séparateur
        sep = detect_separator(content, filepath)
        print(f"✓ Séparateur détecté: {repr(sep)}")
        
        # Lire en DataFrame
        df = read_file_to_dataframe(content, filepath, sep)
        print(f"✓ DataFrame chargé: {len(df)} lignes, {len(df.columns)} colonnes")
        print(f"  Colonnes: {df.columns.tolist()}")
        
        # Détecter le type
        dataset_type, trans_col, items_col, seq_col = detect_dataset_type(df)
        print(f"✓ Type détecté: {dataset_type}")
        print(f"  transaction_col: {trans_col}")
        print(f"  items_col: {items_col}")
        print(f"  sequence_col: {seq_col}")
        
        # Normaliser
        df_normalized = normalize_to_transactional_format(df, dataset_type, trans_col, items_col, seq_col)
        print(f"✓ Normalisé: {len(df_normalized)} transactions")
        
        # Afficher un aperçu
        print(f"\nAperçu (5 premières lignes):")
        print(df_normalized.head().to_string(index=False))
        
        return True
        
    except Exception as e:
        print(f"✗ ERREUR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test tous les fichiers d'exemple"""
    examples_dir = os.path.join(os.path.dirname(__file__), 'data', 'examples')
    
    if not os.path.exists(examples_dir):
        print(f"Erreur: Dossier {examples_dir} introuvable")
        return
    
    # Liste des fichiers à tester
    test_files = [
        'transactional_exemple.csv',
        'transactional_semicolon.csv',
        'transactional_tab.csv',
        'transactional_pipe.csv',
        'inversed_transactional_exemple.csv',
        'sequential_example.csv',
        'sequential_tab.csv',
        'matrix_example.csv',
        'matrix_semicolon.csv'
    ]
    
    results = {}
    
    for filename in test_files:
        filepath = os.path.join(examples_dir, filename)
        if os.path.exists(filepath):
            results[filename] = test_file(filepath)
        else:
            print(f"\n⚠️  Fichier non trouvé: {filename}")
            results[filename] = False
    
    # Résumé
    print(f"\n{'='*80}")
    print("RÉSUMÉ DES TESTS")
    print(f"{'='*80}")
    
    success = sum(1 for v in results.values() if v)
    total = len(results)
    
    for filename, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {filename}")
    
    print(f"\n{success}/{total} tests réussis")
    
    if success == total:
        print("\n🎉 Tous les tests sont passés avec succès!")
    else:
        print(f"\n⚠️  {total - success} test(s) en échec")
        sys.exit(1)

if __name__ == '__main__':
    main()
