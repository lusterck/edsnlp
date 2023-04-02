from typing import Any, Dict, Optional

from edsnlp.core import PipelineProtocol, registry

from .congestive_heart_failure import CongestiveHeartFailure

DEFAULT_CONFIG = dict(patterns=None)


@registry.factory.register(
    "eds.congestive_heart_failure",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: PipelineProtocol,
    name: str,
    patterns: Optional[Dict[str, Any]],
):
    return CongestiveHeartFailure(nlp, patterns=patterns)
