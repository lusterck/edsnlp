import re
from typing import Dict, List, Optional, Union

from spacy.language import Language
from spacy.tokens import Doc, Span

from edsnlp.matchers.regex import RegexMatcher
from edsnlp.pipelines.base import BaseComponent
from edsnlp.utils.filter import filter_spans

from . import patterns
from .patterns import default_enabled


class PollutionTagger(BaseComponent):
    """
    Tags pollution tokens.

    Populates a number of spaCy extensions :

    - `Token._.pollution` : indicates whether the token is a pollution
    - `Doc._.clean` : lists non-pollution tokens
    - `Doc._.clean_` : original text with pollutions removed.
    - `Doc._.char_clean_span` : method to create a Span using character
      indices extracted using the cleaned text.

    Parameters
    ----------
    nlp : Language
        The pipeline object
    name : Optional[str]
        The component name.
    pollution : Dict[str, Union[str, List[str]]]
        Dictionary containing regular expressions of pollution.
    """

    # noinspection PyProtectedMember
    def __init__(
        self,
        nlp: Language,
        name: Optional[str] = "eds.pollution",
        *,
        pollution: Dict[str, Union[bool, str, List[str]]] = default_enabled,
    ):

        self.nlp = nlp
        self.name = name
        self.nlp.vocab.strings.add("EXCLUDED")

        self.pollution = dict()

        for k, v in pollution.items():
            if v is True:
                pattern = patterns.pollution[k]
                self.pollution[k] = pattern if isinstance(pattern, list) else [pattern]
            elif isinstance(v, str):
                self.pollution[k] = [v]
            elif isinstance(v, list):
                self.pollution[k] = v

        self.regex_matcher = RegexMatcher(flags=re.MULTILINE)
        self.build_patterns()

    def build_patterns(self) -> None:
        """
        Builds the patterns for phrase matching.
        """

        # efficiently build spaCy matcher patterns
        for k, v in self.pollution.items():
            self.regex_matcher.add(k, v)

    def process(self, doc: Doc) -> List[Span]:
        """
        Find pollutions in doc and clean candidate negations to remove pseudo negations

        Parameters
        ----------
        doc:
            spaCy Doc object

        Returns
        -------
        pollution:
            list of pollution spans
        """

        pollutions = self.regex_matcher(doc, as_spans=True)
        pollutions = filter_spans(pollutions)

        return pollutions

    def __call__(self, doc: Doc) -> Doc:
        """
        Tags pollutions.

        Parameters
        ----------
        doc:
            spaCy Doc object

        Returns
        -------
        doc:
            spaCy Doc object, annotated for pollutions.
        """
        excluded_hash = doc.vocab.strings["EXCLUDED"]
        pollutions = self.process(doc)

        for pollution in pollutions:

            for token in pollution:
                token._.excluded = True
                token.tag = excluded_hash

        doc.spans["pollutions"] = pollutions

        return doc
