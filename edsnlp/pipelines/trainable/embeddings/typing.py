import torch
from typing_extensions import TypedDict

from edsnlp.core.component import BatchInput  # noqa: F401

WordEmbeddingBatchOutput = TypedDict(
    "WordEmbeddingBatchOutput",
    {
        "embeddings": torch.Tensor,
        "mask": torch.Tensor,
    },
)
