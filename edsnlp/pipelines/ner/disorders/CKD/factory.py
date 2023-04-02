from typing import Any, Dict, Optional

from edsnlp.core import PipelineProtocol, registry

from .CKD import CKD

DEFAULT_CONFIG = dict(patterns=None)


@registry.factory.register(
    "eds.CKD",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: PipelineProtocol,
    name: str,
    patterns: Optional[Dict[str, Any]],
):
    return CKD(nlp, patterns=patterns)
