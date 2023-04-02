from typing import Any, Dict, Optional

from edsnlp.core import PipelineProtocol, registry

from .diabetes import Diabetes

DEFAULT_CONFIG = dict(patterns=None)


@registry.factory.register(
    "eds.diabetes",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: PipelineProtocol,
    name: str,
    patterns: Optional[Dict[str, Any]],
):
    return Diabetes(nlp, patterns=patterns)
