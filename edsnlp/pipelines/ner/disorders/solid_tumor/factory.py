from typing import Any, Dict, Optional

from edsnlp.core import PipelineProtocol, registry

from .solid_tumor import SolidTumor

DEFAULT_CONFIG = dict(
    patterns=None,
    use_tnm=False,
)


@registry.factory.register(
    "eds.solid_tumor",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: PipelineProtocol,
    name: str,
    patterns: Optional[Dict[str, Any]],
    use_tnm: bool,
):
    return SolidTumor(nlp, patterns=patterns, use_tnm=use_tnm)
