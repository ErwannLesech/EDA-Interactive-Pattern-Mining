import pandas as pd
from typing import List

class PatternMiner:
    """Classe pour l'extraction de motifs fréquents"""
    
    def __init__(self, transactions: List[List[str]]):
        self.transactions = transactions