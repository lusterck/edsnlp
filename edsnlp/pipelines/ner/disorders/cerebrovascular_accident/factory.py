from typing import Any, Dict, Optional

from edsnlp.core import PipelineProtocol, registry

from .cerebrovascular_accident import CerebrovascularAccident

DEFAULT_CONFIG = dict(patterns=None)


@registry.factory.register(
    "eds.cerebrovascular_accident",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: PipelineProtocol,
    name: str,
    patterns: Optional[Dict[str, Any]],
):
    return CerebrovascularAccident(nlp, patterns=patterns)
