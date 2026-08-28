from dataclasses import dataclass

import numpy as np


@dataclass
class ExpenseRecord:
    text: str
    category: str
    embedding: np.ndarray | None = None
    predicted_category: str | None = None


@dataclass
class ExpenseDataset:
    train: list[ExpenseRecord]
    test: list[ExpenseRecord]
