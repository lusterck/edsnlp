from typing import Any, Dict, Optional

from edsnlp.core import PipelineProtocol, registry

from .lymphoma import Lymphoma

DEFAULT_CONFIG = dict(patterns=None)


@registry.factory.register(
    "eds.lymphoma",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: PipelineProtocol,
    name: str,
    patterns: Optional[Dict[str, Any]],
):
    return Lymphoma(nlp, patterns=patterns)
