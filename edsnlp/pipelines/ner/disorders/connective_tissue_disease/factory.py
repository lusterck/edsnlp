from typing import Any, Dict, Optional

from edsnlp.core import PipelineProtocol, registry

from .connective_tissue_disease import ConnectiveTissueDisease

DEFAULT_CONFIG = dict(patterns=None)


@registry.factory.register(
    "eds.connective_tissue_disease",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: PipelineProtocol,
    name: str,
    patterns: Optional[Dict[str, Any]],
):
    return ConnectiveTissueDisease(nlp, patterns=patterns)
