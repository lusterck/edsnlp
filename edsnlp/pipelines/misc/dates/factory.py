from typing import List, Optional, Set, Union

from edsnlp.core import PipelineProtocol, registry
from edsnlp.utils.deprecation import deprecated_factory

from . import Dates

DEFAULT_CONFIG = dict(
    absolute=None,
    relative=None,
    duration=None,
    false_positive=None,
    detect_periods=False,
    detect_time=True,
    on_ents_only=False,
    as_ents=False,
    attr="LOWER",
)


@deprecated_factory(
    "dates", "eds.dates", default_config=DEFAULT_CONFIG, assigns=["doc.spans"]
)
@registry.factory.register(
    "eds.dates", default_config=DEFAULT_CONFIG, assigns=["doc.spans"]
)
def create_component(
    nlp: PipelineProtocol,
    name: str,
    absolute: Optional[List[str]],
    relative: Optional[List[str]],
    duration: Optional[List[str]],
    false_positive: Optional[List[str]],
    on_ents_only: Union[bool, str, List[str], Set[str]],
    detect_periods: bool,
    detect_time: bool,
    as_ents: bool,
    attr: str,
):
    return Dates(
        nlp,
        absolute=absolute,
        relative=relative,
        duration=duration,
        false_positive=false_positive,
        on_ents_only=on_ents_only,
        detect_periods=detect_periods,
        detect_time=detect_time,
        as_ents=as_ents,
        attr=attr,
    )
