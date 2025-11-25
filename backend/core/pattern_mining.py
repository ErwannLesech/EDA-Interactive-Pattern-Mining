import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
import logging
from prefixspan import PrefixSpan
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class PatternMiner:
    """Classe pour l'extraction de motifs fréquents
    
    Note: Pour les datasets séquentiels (is_sequential=True), cette classe
    extrait les motifs fréquents sans tenir compte de l'ordre.
    Pour le mining séquentiel (ordre important), utilisez PrefixSpan ou GSP.
    
    Exemple d'utilisation avec le flag is_sequential:
        from utils.storage import DatasetStorage
        
        metadata = DatasetStorage.get_dataset_metadata(dataset_id)
        if metadata and metadata.get('is_sequential'):
            # TODO: Implémenter le mining séquentiel avec PrefixSpan
            # from prefixspan import PrefixSpan
            # patterns = mine_sequential_patterns(dataset_id, min_support)
            pass
        else:
            # Mining classique (ordre non important)
            miner = PatternMiner(transactions)
            itemsets, rules = miner.mine_patterns(min_support, min_confidence)
    """
    
    def __init__(self, transactions: pd.DataFrame, sequential: bool = False):
        self.transactions = transactions
        self.sequential = sequential
        self.frequent_itemsets = None
        self.rules = None

    def mine_patterns(self, min_support: float = 0.01, min_confidence: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Extrait les motifs fréquents et les règles d'association
        
        Args:
            min_support (float): Le support minimum pour les motifs fréquents.
            min_confidence (float): La confiance minimum pour les règles d'association.
        
        Returns:
            tuple: (frequent_itemsets, rules) - DataFrames contenant les motifs et les règles
        """
        transactions_len = len(self.transactions)
        if not self.sequential:
            # Générer les itemsets fréquents
            self.frequent_itemsets = fpgrowth(self.transactions, min_support=min_support, use_colnames=True)
            # Générer les règles d'association
            logger.info(f"Nombre de motifs fréquents extraits: {len(self.frequent_itemsets)}")
            logger.info(self.frequent_itemsets.head())
            #self.rules = association_rules(self.frequent_itemsets, metric="confidence", min_threshold=min_confidence)
            #logger.info(f"Nombre de règles d'association générées: {len(self.rules)}")
            self.frequent_itemsets['couverture'] = self.frequent_itemsets['support'] * transactions_len
            #self.rules.drop(columns=['leverage', 'conviction','zhangs_metric', 'jaccard', 'certainty', 'kulczynski','representativity'], inplace=True)
        else:
            transactions_list = self.transactions['items'].values.tolist()
            ps = PrefixSpan(transactions_list)
            freq_patterns = ps.frequent(minsup=int(min_support * len(transactions_list)))
            self.frequent_itemsets = pd.DataFrame(freq_patterns, columns=['support', 'itemsets'])
            self.frequent_itemsets['couverture'] = self.frequent_itemsets['support'].copy()
            self.frequent_itemsets['support'] = self.frequent_itemsets['support'] / transactions_len
            logger.info(f"Nombre de motifs séquentiels extraits: {len(self.frequent_itemsets)}")
        
        self.frequent_itemsets['longueur'] = self.frequent_itemsets['itemsets'].apply(lambda x: len(x))
        self.frequent_itemsets['area'] = self.frequent_itemsets['support'] * self.frequent_itemsets['longueur']
        cols = ['itemsets', 'support', 'longueur', 'couverture', 'area']
        self.frequent_itemsets = self.frequent_itemsets[cols]
        self.rules = pd.DataFrame()  # Placeholder si les règles ne sont pas utilisées
        return self.frequent_itemsets, self.rules