"""
Recommendations Module

Provides fix recommendations for security vulnerabilities.
"""

from .mappings import RECOMMENDATIONS, get_recommendation
from .generator import RecommendationGenerator

__all__ = [
    "RECOMMENDATIONS",
    "get_recommendation",
    "RecommendationGenerator",
]
