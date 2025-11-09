from pydantic import BaseModel
from typing import List, Optional
from enum import Enum

"""
Modèles de données pour les patterns extraits et la gestion des datasets.
"""

class DatasetType(str, Enum):
    TRANSACTIONAL = "transactional"
    SEQUENTIAL = "sequential"
    MATRIX = "matrix"
    INVERSED = "inversed"

class UploadResponse(BaseModel):
    dataset_id: str
    filename: str
    dataset_type: DatasetType
    rows: int
    columns: List[str]
    preview: List[dict]
    message: str

class Pattern(BaseModel):
    id: int
    items: List[str]
    confidence: Optional[float] = None
    lift: Optional[float] = None # score additionnel
    length: int
    score: Optional[float] = None

class MiningResult(BaseModel):
    patterns: List[Pattern]
    total_patterns: int
    computation_time: float # utile pour la perf des algos

class Feedback(BaseModel):
    pattern_id: int
    useful: bool
    comments: Optional[str] = None