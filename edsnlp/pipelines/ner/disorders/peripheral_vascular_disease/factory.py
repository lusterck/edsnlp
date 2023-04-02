from typing import Any, Dict, Optional

from edsnlp.core import PipelineProtocol, registry

from .peripheral_vascular_disease import PeripheralVascularDisease

DEFAULT_CONFIG = dict(patterns=None)


@registry.factory.register(
    "eds.peripheral_vascular_disease",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: PipelineProtocol,
    name: str,
    patterns: Optional[Dict[str, Any]],
):
    return PeripheralVascularDisease(nlp, patterns=patterns)
