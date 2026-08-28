from collections.abc import Callable
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


DATA_FOLDER = Path(__file__).resolve().parent.parent / "data"

TRAINING_DATA = DATA_FOLDER / "expenses_training.csv"


def read_data(file_name: str | Path) -> pd.DataFrame:
    return pd.read_csv(file_name)


def get_data(input_name: str, category_name: str, data_processing_config: dict) -> tuple[
    pd.Series, pd.Series, pd.Series, pd.Series
]:

    input_factory = create_input(input_name)
    category_factory = create_category(category_name)

    df = read_data(TRAINING_DATA)
    df = process_data(df, remove_misc=data_processing_config["remove_misc"])

    # Create input and category columns
    df["input"] = df.apply(input_factory, axis=1)
    df["category"] = df.apply(category_factory, axis=1)

    X_train, X_test, y_train, y_test = split_train_test(df)

    return X_train, X_test, y_train, y_test


def create_merchant_and_price_input(row):
    return f"Merchant: {row['label']}; Price: {row['price']}"


def create_merchant_input(row):
    return f"Merchant: {row['label']}"


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
    df = df.sort_values("date")

    if remove_misc:    
        df = df[~df["category_1"].str.contains("Misc")]

    return df


def split_train_test(df):
    X = df["input"]
    y = df["category"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    return X_train, X_test, y_train, y_test
