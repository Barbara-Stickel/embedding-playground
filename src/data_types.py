from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TextRecord:
    text: str
    label: str
    embedding: np.ndarray | None = None
    predicted_label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TextDataset:
    train: list[TextRecord]
    test: list[TextRecord]
    validation: list[TextRecord] = field(default_factory=list)
