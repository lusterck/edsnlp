from typing import Any, Dict, Optional

from edsnlp.core import PipelineProtocol, registry

from .peptic_ulcer_disease import PepticUlcerDisease

DEFAULT_CONFIG = dict(patterns=None)


@registry.factory.register(
    "eds.peptic_ulcer_disease",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: PipelineProtocol,
    name: str,
    patterns: Optional[Dict[str, Any]],
):
    return PepticUlcerDisease(nlp, patterns=patterns)
