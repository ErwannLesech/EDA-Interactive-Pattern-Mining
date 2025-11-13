import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

class PatternMiner:
    """Classe pour l'extraction de motifs fréquents"""
    
    def __init__(self, transactions: pd.DataFrame):
        self.transactions = transactions
        self.frequent_itemsets = None
        self.rules = None
    def mine_patterns(self, min_support: float = 0.01, min_confidence: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Extrait les motifs fréquents et les règles d'association
        
        Args:
            min_support (float): Le support minimum pour les motifs fréquents.
            min_confidence (float): La confiance minimum pour les règles d'association.
        
        Returns:
            pd.DataFrame: Un DataFrame contenant les règles d'association.
        """
        # Générer les itemsets fréquents
        self.frequent_itemsets = apriori(self.transactions, min_support=min_support, use_colnames=True)
        # Générer les règles d'association
        self.rules = association_rules(self.frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        self.frequent_itemsets['length'] = self.frequent_itemsets['itemsets'].apply(lambda x: len(x))
        self.frequent_itemsets['coverage'] = self.frequent_itemsets['support'] * len(self.transactions)
        
        # Supprimer les colonnes inutiles (seulement celles qui existent)
        columns_to_drop = ['leverage', 'conviction', 'zhangs_metric', 'jaccard', 'certainty', 'kulczynski', 'representativity']
        existing_columns_to_drop = [col for col in columns_to_drop if col in self.rules.columns]
        if existing_columns_to_drop:
            self.rules.drop(columns=existing_columns_to_drop, inplace=True)

        return self.frequent_itemsets, self.rules
