"""
EDS-NLP
"""

from pathlib import Path

from spacy import pipeline as _spacy_pipeline  # noqa: F401
from . import extensions
from . import patch_spacy
from .core.pipeline import Pipeline, blank
from .core.registry import registry

__version__ = "0.8.1"

BASE_DIR = Path(__file__).parent
