from decimal import Decimal
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from typing import List, Tuple, FrozenSet
import logging
import numpy as np
from prefixspan import PrefixSpan
import random
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class PatternMiner:
    """Classe pour l'extraction de motifs fréquents
    
    Note: Pour les datasets séquentiels (is_sequential=True), cette classe
    extrait les motifs fréquents sans tenir compte de l'ordre avec FP-Growth.
    Pour le mining séquentiel (ordre important), on utilise PrefixSpan.

    """
    
    def __init__(self, transactions: pd.DataFrame, sequential: bool = False):
        self.transactions = transactions
        self.sequential = sequential
        self.frequent_itemsets = None
        self.rules = None
        random.seed(42)
        np.random.seed(42)

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

            logger.info(f"Nombre de motifs fréquents extraits: {len(self.frequent_itemsets)}")
            logger.info(self.frequent_itemsets.head())
            self.frequent_itemsets['couverture'] = self.frequent_itemsets['support'] * transactions_len
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
    
    def twostep_sampling(self, k: int) -> pd.DataFrame:
        """
        Two-Step Sampling on a binarized transaction DataFrame.
        Returns sampled itemsets with their support.
        
        Args:
            df: Binarized DataFrame (rows=transactions, columns=items)
            k: Number of unique motifs to sample
            
        Returns:
            DataFrame with columns ['itemset', 'support']
        """
        logger.info(self.transactions.head())
        if self.transactions.empty:
            return pd.DataFrame(columns=['itemset', 'support'])

        # Step 1: Compute cumulative weights for transactions
        weights = []
        cumulative_weight = Decimal(0)

        for _, row in self.transactions.iterrows():
            if not self.sequential:
                transaction_items = row.index[row == 1].tolist()
            else:
                transaction_items = row['items']
            weight = 2 ** len(transaction_items)
            cumulative_weight += Decimal(weight)
            weights.append(cumulative_weight)
        Z = weights[-1]

        sampled_patterns = set()
        max_iterations = k * 10  # Prevent infinite loops
        iterations = 0
        # Step 2: Sample until k unique motifs
        while len(sampled_patterns) < k and iterations < max_iterations:
            iterations += 1
            # logger.info(f"Sampling iteration, currently have {len(sampled_patterns)} unique patterns.")
            # Weighted transaction selection
            rand_value = Decimal(random.random()) * Z
            left, right = 0, len(weights)
            while left < right:
                mid = (left + right) // 2
                if weights[mid] < rand_value:
                    left = mid + 1
                else:
                    right = mid
            t_id = min(left, len(self.transactions) - 1)
            if not self.sequential:
                transaction_items = self.transactions.iloc[t_id].index[self.transactions.iloc[t_id] == 1].tolist()
            else:
                transaction_items = self.transactions.iloc[t_id]['items']
            # logger.info(f"Selected transaction {t_id} with items {transaction_items}")

            # Sample a subset of the transaction
            pattern = [item for item in transaction_items if random.random() > 0.5]
            if not pattern and transaction_items:
                pattern = [random.choice(transaction_items)]

            sampled_patterns.add(tuple(pattern))
        
        # Step 3: Compute support for each sampled motif
        output_data = []
        for pattern in sampled_patterns:
            cols = list(pattern)
            if self.sequential:
                support = self.compute_sequential_support(pattern, self.transactions['items'].tolist())
            else:
                support = (self.transactions[cols].all(axis=1).sum()) / len(self.transactions)
            output_data.append({'itemsets': cols, 'support': support})

        return pd.DataFrame(output_data)
    def is_subsequence(self, subseq, seq):
        """
        Check if `subseq` is an ordered subsequence of `seq`
        using a linear two-pointer scan.
        """
        i, j = 0, 0
        while i < len(subseq) and j < len(seq):
            if subseq[i] == seq[j]:
                i += 1
            j += 1
        return i == len(subseq)

    def compute_sequential_support(self, subseq, sequences):
        """Calcule le support séquentiel d'un sous-ensemble dans une liste de séquences."""
        count = 0
        for seq in sequences:
            if self.is_subsequence(subseq, seq):
                count += 1
        return count / len(sequences)
    
    def gdps_sampling(self, k: int,
                     min_norm: int = 1,
                     max_norm: int = 10,
                     utility: str = "freq") -> pd.DataFrame:
        """
        Generic Direct Pattern Sampling (GDPS) for TRANSACTIONAL binarized data only.

        Args:
            k: Number of patterns to sample
            min_norm: Minimum pattern size
            max_norm: Maximum pattern size
            utility: Utility type ("freq", "area", "decay")

        Returns:
            DataFrame with columns ['itemset', 'support']
        """
        from decimal import Decimal
        import math

        if self.transactions.empty:
            return pd.DataFrame(columns=['itemset', 'support'])

        def compute_utility(norm: int, utility_type: str) -> float:
            if utility_type == "freq":
                return 1.0
            elif utility_type == "area":
                return float(norm)
            elif utility_type == "decay":
                return math.exp(-norm)
            return 1.0

        weights = []
        cumulative_weight = Decimal(0)
        # Step 1: Compute cumulative weights for transactions
        for _, row in self.transactions.iterrows():
            transaction = row.index[row == 1].tolist()
            t_size = len(transaction)

            if t_size == 0:
                weights.append(cumulative_weight)
                continue

            weight = Decimal(0)

            for l in range(min_norm, min(max_norm + 1, t_size + 1)):
                n_patterns = math.comb(t_size, l)
                util = compute_utility(l, utility)
                weight += Decimal(n_patterns * util)

            cumulative_weight += weight
            weights.append(cumulative_weight)

        Z = weights[-1] if weights else Decimal(1)
        sampled_patterns = set()


        max_attempts = 20 * k
        attempts = 0
        # Step 2: Sample until k unique motifs
        while len(sampled_patterns) < k and attempts < max_attempts:
            attempts += 1

            # Weighted transaction selection
            rand_value = Decimal(random.random()) * Z
            left, right = 0, len(weights)

            while left < right:
                mid = (left + right) // 2
                if weights[mid] < rand_value:
                    left = mid + 1
                else:
                    right = mid

            t_id = min(left, len(self.transactions) - 1)
            transaction = self.transactions.iloc[t_id].index[
                self.transactions.iloc[t_id] == 1
            ].tolist()

            t_size = len(transaction)
            if t_size == 0:
                continue

            valid_norms = list(range(min_norm, min(max_norm + 1, t_size + 1)))

            norm_probs = []
            for l in valid_norms:
                n_patterns = math.comb(t_size, l)
                util = compute_utility(l, utility)
                norm_probs.append(n_patterns * util)

            total = sum(norm_probs)
            norm_probs = [p / total for p in norm_probs] if total > 0 else \
                        [1 / len(norm_probs)] * len(norm_probs)

            chosen_norm = np.random.choice(valid_norms, p=norm_probs)


            pattern = random.sample(transaction, chosen_norm)
            sampled_patterns.add(tuple(pattern))

        output_data = []
        total_transactions = len(self.transactions)
        # Step 3: Compute support for each sampled motif
        for pattern in sampled_patterns:
            pattern = list(pattern)
            support = (self.transactions[pattern].all(axis=1).sum()) / total_transactions

            output_data.append({
                'itemset': pattern,
                'support': support
            })

        return pd.DataFrame(output_data)
