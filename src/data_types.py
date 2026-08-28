from dataclasses import dataclass
import datetime

import numpy as np


@dataclass
class ExpenseRecord:
    date: datetime.date
    text: str
    category: str
    embedding: np.ndarray | None = None
    predicted_category: str | None = None


@dataclass
class ExpenseDataset:
    train: list[ExpenseRecord]
    test: list[ExpenseRecord]
