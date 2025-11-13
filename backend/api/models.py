from pydantic import BaseModel
from typing import List, Optional, Dict, Any
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


# ========== Modèles pour l'évaluation (Partie 4) ==========

class FeedbackData(BaseModel):
    """Modèle pour soumettre un feedback utilisateur"""
    dataset_id: str
    pattern_id: int
    rating: int  # 1 pour like, -1 pour dislike, 0 pour neutre
    comment: Optional[str] = None


class DiversityMetrics(BaseModel):
    """Métriques de diversité des motifs"""
    average_diversity: float
    min_diversity: float
    max_diversity: float
    std_diversity: float
    method: str
    num_patterns: int
    num_comparisons: Optional[int] = None


class CoverageMetrics(BaseModel):
    """Métriques de couverture des transactions"""
    coverage_rate: float  # Pourcentage
    covered_transactions: int
    total_transactions: int
    average_coverage_per_pattern: float
    min_coverage: Optional[int] = None
    max_coverage: Optional[int] = None


class AcceptanceMetrics(BaseModel):
    """Métriques d'acceptation utilisateur"""
    acceptance_rate: float  # Pourcentage
    total_feedbacks: int
    likes: int
    dislikes: int
    neutral: Optional[int] = 0


class StabilityMetrics(BaseModel):
    """Métriques de stabilité de l'échantillonnage"""
    stability_score: float  # Pourcentage
    average_jaccard_similarity: float
    std_jaccard_similarity: float
    min_similarity: Optional[float] = None
    max_similarity: Optional[float] = None
    num_runs: int
    num_comparisons: Optional[int] = None


class PerformanceMetrics(BaseModel):
    """Métriques de performance / temps de réponse"""
    average_time: float  # Secondes
    min_time: float
    max_time: float
    std_time: float
    num_iterations: int
    meets_target: bool  # Vrai si < 3 secondes


class EvaluationSummary(BaseModel):
    """Résumé de l'évaluation globale"""
    overall_quality: str  # Excellente, Bonne, Moyenne, Faible
    recommendations: List[str]


class EvaluationResponse(BaseModel):
    """Réponse complète d'évaluation"""
    timestamp: float
    num_patterns: int
    acceptance: AcceptanceMetrics
    diversity: Dict[str, DiversityMetrics]  # jaccard, cosine, hamming
    coverage: CoverageMetrics
    stability: Optional[StabilityMetrics] = None
    performance: Optional[PerformanceMetrics] = None
    summary: EvaluationSummary


class EvaluationRequest(BaseModel):
    """Requête pour déclencher une évaluation"""
    dataset_id: str
    include_stability: bool = False
    include_performance: bool = False
    stability_runs: Optional[int] = 5
    performance_iterations: Optional[int] = 10
    feedback_data: Optional[List[Dict[str, Any]]] = None


# ========== Modèles pour l'extraction de motifs ==========

class MiningRequest(BaseModel):
    """Requête pour l'extraction de motifs"""
    dataset_id: str
    min_support: float = 0.05
    min_confidence: float = 0.5


class PatternItem(BaseModel):
    """Modèle pour un motif individuel"""
    itemset: List[str]
    support: float
    length: int
    coverage: float


class RuleItem(BaseModel):
    """Modèle pour une règle d'association"""
    antecedents: List[str]
    consequents: List[str]
    support: float
    confidence: float
    lift: float
    antecedent_support: float
    consequent_support: float


class MiningResponse(BaseModel):
    """Réponse de l'extraction de motifs"""
    dataset_id: str
    num_patterns: int
    num_rules: int
    patterns_preview: List[PatternItem]
    rules_preview: List[RuleItem]
    computation_time: float
    message: str


class SamplingRequest(BaseModel):
    """Requête pour l'échantillonnage interactif"""
    dataset_id: str
    k: int = 50  # Nombre de motifs à échantillonner
    replacement: bool = False  # Avec ou sans remise
    support_weight: float = 0.4
    surprise_weight: float = 0.4
    redundancy_weight: float = 0.2


class SamplingResponse(BaseModel):
    """Réponse de l'échantillonnage"""
    dataset_id: str
    num_sampled: int
    sampled_patterns: List[PatternItem]
    message: str

