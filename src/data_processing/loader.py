from data_types import TextDataset

from .expenses import load_expenses
from .hc3 import load_hc3
from .mage import load_mage

DATASET_LOADERS = {
    "expenses": load_expenses,
    "mage": load_mage,
    "hc3": load_hc3,
}


def load_dataset(config: dict) -> TextDataset:
    name = config["name"]
    try:
        loader = DATASET_LOADERS[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown dataset '{name}'. Available choices: {', '.join(DATASET_LOADERS)}"
        ) from error
    dataset = loader(config)
    if not dataset.train or not dataset.test:
        raise ValueError("The dataset must have non-empty training and test splits")
    return dataset
