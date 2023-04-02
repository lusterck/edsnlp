from typing import Any, Dict, Optional

from edsnlp.core import PipelineProtocol, registry

from .alcohol import Alcohol

DEFAULT_CONFIG = dict(patterns=None)


@registry.factory.register(
    "eds.alcohol",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: PipelineProtocol,
    name: str,
    patterns: Optional[Dict[str, Any]],
):
    return Alcohol(nlp, patterns=patterns)
