from collections.abc import Callable
from pathlib import Path
import pandas as pd

from .common import require_columns, resolve_source, split_rows, validate_text_labels

from data_types import TextDataset, TextRecord


def read_data(file_name: str | Path) -> pd.DataFrame:
    return pd.read_csv(file_name)


def load_expenses(config: dict) -> TextDataset:
    input_factory = create_input(config["input"])
    category_factory = create_category(config["category"])
    df = read_data(resolve_source(config["file"]))
    require_columns(df, ["Date", "Description", "Debit", "Type 1", "Details"])
    df = process_data(df, remove_misc=config["remove_misc"])
    df["input"] = df.apply(input_factory, axis=1)
    df["category"] = df.apply(category_factory, axis=1)
    validate_text_labels(df[["input", "category"]].rename(columns={"input": "text", "category": "label"}))
    return split_train_test(
        df,
        split_data=config["split_data"],
        test_size=config["test_size"],
        seed=config["seed"],
    )


def create_merchant_and_price_input(row):
    return f"Merchant: {row['label']}; Price: {row['price']}"


def create_merchant_input(row):
    return str(row['label'])


def create_detailed_category(row) -> str:

    category_2 = row['category_2']
    if pd.isna(category_2) or category_2 == "":
        category_2 = None

    if category_2 is None:
        return str(row['category_1'])
    else:
        return f"{row['category_1']} - {category_2}"


def create_primary_category(row) -> str:
    return str(row['category_1'])


INPUT_FACTORIES = {
    "merchant_and_price": create_merchant_and_price_input,
    "merchant_only": create_merchant_input,
}

CATEGORY_FACTORIES = {
    "detailed": create_detailed_category,
    "primary": create_primary_category,
}


def _get_factory(factories: dict, name: str, factory_type: str):
    try:
        return factories[name]
    except KeyError as error:
        choices = ", ".join(factories)
        raise ValueError(
            f"Unknown {factory_type} '{name}'. Available choices: {choices}"
        ) from error


def create_input(name: str) -> Callable:
    return _get_factory(INPUT_FACTORIES, name, "input")


def create_category(name: str) -> Callable:
    return _get_factory(CATEGORY_FACTORIES, name, "category")


def process_data(df, remove_misc=True):
    df = df.rename(columns={
        "Date": "date",
        "Description": "label",
        "Debit": "price",
        "Type 1": "category_1",
        "Details": "category_2"
    })

    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    df = df.sort_values("date")

    text_columns = df.select_dtypes(include=["object", "string"]).columns
    df[text_columns] = df[text_columns].apply(
        lambda column: column.str.lower()
    )

    if remove_misc:
        df = df[~df["category_1"].str.contains("misc", na=False)]

    return df


def create_records(df) -> list[TextRecord]:
    return [
        TextRecord(
            text=row.input,
            label=row.category,
            metadata={"date": row.date.date()}
        )
        for row in df.itertuples()
    ]


def split_train_test(df, split_data: str, test_size: float = 0.2, seed: int = 42) -> TextDataset:
    train, test = split_rows(df, split_data, test_size=test_size, seed=seed)
    return TextDataset(train=create_records(train), test=create_records(test))
