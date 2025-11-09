import numpy as np
from typing import List
import pandas as pd

class PatternSampler:
    """Classe pour l'échantillonnage de motifs"""
    
    def __init__(self, patterns: pd.DataFrame):
        self.patterns = patterns