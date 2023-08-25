from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from pydantic import StrictStr
from spacy.tokens import Doc, Span
from typing_extensions import NotRequired, TypedDict


class ListStr(list, List[StrictStr]):
    """
    A coercing list of str, i.e. a list that can be initialized with a str or a list of
    str.
    """

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value: Union[str, List[StrictStr]]) -> List[str]:
        if isinstance(value, str):
            return [value]
        elif isinstance(value, Sequence) and all(isinstance(v, str) for v in value):
            return list(value)
        else:
            raise TypeError(
                f"Invalid entry {value} ({type(value)}) for ListStr, "
                f"expected str or sequence of str"
            )


Spans = List[Span]
SpanGroups = Dict[str, Spans]
SpanGetterCallable = Callable[[Doc], Iterable[Span]]
SpanGetterMapping = TypedDict(
    "SpanGetterMapping",
    {
        "ents": NotRequired[
            Union[
                bool,  # get all spans in .ents
                # ListStr,  # get only spans in .ents that match these labels
            ]
        ],
        "span_groups": NotRequired[
            Union[
                ListStr,  # get all spans in these .spans groups
                # Dict[
                #     str, ListStr
                # ],  # get only spans in these .spans groups that match these labels
            ]
        ],
        "labels": NotRequired[ListStr],  # get only spans with these labels
    },
)

SpanGetter = Union[
    ListStr,
    SpanGetterMapping,
    Callable[[Doc], Iterable[Span]],
]
"""
A SpanGetter can be either:

- a str or a list of str to filter spans by label
- a mapping with the following keys:
    + "ents": Whether to look into `doc.ents` for spans to classify.
    + "span_groups": Whether to look into `doc.spans` for spans to classify.
        If a list of str is provided, only these span groups will be kept.
    + "labels": a list of str to filter spans by label
- a callable that takes a Doc and returns an iterable of spans
"""


def get_spans(
    doc: Doc,
    span_getter: SpanGetter,
    return_origin: bool = False,
) -> Union[Tuple[Spans], Tuple[Spans, Optional[Spans], SpanGroups]]:
    """
    Make a span qualifier candidate getter function.

    Parameters
    ----------
    doc: Doc
    span_getter: SpanGetter
        - ents: Whether to look into `doc.ents` for spans to classify. If a list of str
            is provided, only the span of the given labels will be considered. If None
            and `on_spans_groups` is False, labels mentioned in `label_constraints`
            will be used.
        - span_groups:
            Whether to look into `doc.spans` for spans to classify:

            - If True, all span groups will be considered
            - If False, no span group will be considered
            - If a list of str is provided, only these span groups will be kept
            - If a mapping is provided, the keys are the span group names and the values
              are either a list of allowed labels in the group or True to keep them all
    return_origin: bool
        Whether to return the original spans, entities and span groups
    """
    flattened_spans = []
    span_groups = {}
    ents = None
    if span_getter.get("ents"):
        # /!\ doc.ents is not a list but a Span iterator, so to ensure referential
        # equality between the spans of `flattened_spans` and `ents`,
        # we need to convert it to a list to "extract" the spans first
        ents = list(doc.ents)
        flattened_spans.extend(
            (
                span
                for span in ents
                if "labels" not in span_getter or span.label_ in span_getter["labels"]
            )
            if isinstance(span_getter["ents"], Sequence)
            else ents
        )

    if span_getter.get("span_groups"):
        if isinstance(span_getter["span_groups"], Sequence):
            for name in span_getter["span_groups"]:
                span_groups[name] = list(doc.spans.get(name, ()))
                flattened_spans.extend(
                    (
                        span
                        for span in span_groups[name]
                        if "labels" not in span_getter
                        or span.label_ in span_getter["labels"]
                    )
                    if isinstance(span_getter["span_groups"], Sequence)
                    else span_groups[name]
                )
        else:
            for name, spans_ in doc.spans.items():
                # /!\ spans_ is not a list but a SpanGroup, so to ensure referential
                # equality between the spans of `flattened_spans` and `span_groups`,
                # we need to convert it to a list to "extract" the spans first
                span_groups[name] = list(spans_)
                flattened_spans.extend(span_groups[name])

    if return_origin:
        return flattened_spans, ents, span_groups
    else:
        return flattened_spans
