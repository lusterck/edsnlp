from typing import Dict, List, Optional, Union

from edsnlp.core import PipelineProtocol, registry
from edsnlp.utils.deprecation import deprecated_factory

from .tables import TablesMatcher

DEFAULT_CONFIG = dict(
    tables_pattern=None,
    sep_pattern=None,
    attr="TEXT",
    ignore_excluded=True,
)


@deprecated_factory("tables", "eds.tables", default_config=DEFAULT_CONFIG)
@registry.factory.register("eds.tables", default_config=DEFAULT_CONFIG)
def create_component(
    nlp: PipelineProtocol,
    name: str,
    tables_pattern: Optional[Dict[str, Union[List[str], str]]],
    sep_pattern: Optional[str],
    attr: str,
    ignore_excluded: bool,
):
    return TablesMatcher(
        nlp,
        tables_pattern=tables_pattern,
        sep_pattern=sep_pattern,
        attr=attr,
        ignore_excluded=ignore_excluded,
    )
