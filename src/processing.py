from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


DATA_FOLDER = Path(__file__).resolve().parent.parent / "data"

TRAINING_DATA = DATA_FOLDER / "expenses_training.csv"


def read_data(file_name: str | Path) -> pd.DataFrame:
    return pd.read_csv(file_name)


def get_data() -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:

    df = read_data(TRAINING_DATA)
    df = process_data(df)

    X_train, X_test, y_train, y_test = split_train_test(df)

    return X_train, X_test, y_train, y_test


def create_input(row):

    return f"Merchant: {row['label']}; Price: {row['price']}"

def create_input_2(row):

    return f"Merchant: {row['label']}"


def create_category(row) -> str:

    category_2 = row['category_2']
    if pd.isna(category_2) or category_2 == "":
        category_2 = None

    if category_2 is None:
        return str(row['category_1'])
    else:
        return f"{row['category_1']} - {category_2}"


def create_category_2(row) -> str:

    return str(row['category_1'])


def process_data(df):
    df = df.rename(columns={
        "Date": "date",
        "Description": "label",
        "Debit": "price",
        "Type 1": "category_1",
        "Details": "category_2"
    })
    df = df.sort_values("date")

    df["input"] = df.apply(create_input_2, axis=1)
    df["category"] = df.apply(create_category_2, axis=1)

    return df


def split_train_test(df):
    X = df["input"]
    y = df["category"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    return X_train, X_test, y_train, y_test
