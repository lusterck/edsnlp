from typing import Any, Dict, Optional

from edsnlp.core import PipelineProtocol, registry

from .myocardial_infarction import MyocardialInfarction

DEFAULT_CONFIG = dict(patterns=None)


@registry.factory.register(
    "eds.myocardial_infarction",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: PipelineProtocol,
    name: str,
    patterns: Optional[Dict[str, Any]],
):
    return MyocardialInfarction(nlp, patterns=patterns)
