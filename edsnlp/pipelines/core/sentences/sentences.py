from typing import List, Optional

from spacy import Vocab
from spacy.tokens import Doc

from .fast_sentences import FastSentenceSegmenter
from .terms import punctuation


class SentenceSegmenter:
    """
    Segments the Doc into sentences using a rule-based strategy,
    specific to AP-HP documents.

    Applies the same rule-based pipeline as spaCy's sentencizer,
    and adds a simple rule on the new lines : if a new line is followed by a
    capitalised word, then it is also an end of sentence.
    """

    def __init__(
        self,
        vocab: Vocab,
        punct_chars: Optional[List[str]] = None,
        use_endlines: bool = None,
        ignore_excluded: bool = True,
    ):
        """
        Parameters
        ----------
        vocab : Vocab
            Vocabulary
        punct_chars: Optional[List[str]]
            Punctuation characters.
        use_endlines: bool
            Whether to use endlines prediction.
        ignore_excluded: bool
            Whether to ignore excluded tokens.
        """

        if punct_chars is None:
            punct_chars = punctuation

        self.fast_segmenter = FastSentenceSegmenter(
            vocab,
            punct_chars,
            use_endlines,
            ignore_excluded,
        )

    def __call__(self, doc: Doc):
        return self.fast_segmenter(doc)
