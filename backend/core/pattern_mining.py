import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
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
        logger.info(f"Nombre de motifs fréquents extraits: {len(self.frequent_itemsets)}")
        #self.rules = association_rules(self.frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        #logger.info(f"Nombre de règles d'association générées: {len(self.rules)}")
        self.frequent_itemsets['longueur'] = self.frequent_itemsets['itemsets'].apply(lambda x: len(x))
        self.frequent_itemsets['couverture'] = self.frequent_itemsets['support'] * len(self.transactions)
        self.frequent_itemsets['area'] = self.frequent_itemsets['support'] * self.frequent_itemsets['longueur']
        cols = ['itemsets', 'support', 'longueur', 'couverture', 'area']
        self.frequent_itemsets = self.frequent_itemsets[cols]
        #self.rules.drop(columns=['leverage', 'conviction','zhangs_metric', 'jaccard', 'certainty', 'kulczynski','representativity'], inplace=True)
        self.rules = pd.DataFrame()  # Placeholder si les règles ne sont pas utilisées
        return self.frequent_itemsets, self.rules