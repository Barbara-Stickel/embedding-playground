from pathlib.path import Path
import pandas as pd
import numpy as np
from typing import Any

from sklearn.model_selection import train_test_split


DATA_FOLDER = Path("." / "data")

TRAINING_DATA = DATA_FOLDER / "expenses_training.csv"


def read_data(file_name: str) -> Any:

    with open(file_name):
        df = pd.read_csv(file_name)

    return df

def get_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    df = read_data(TRAINING_DATA)
    df = process_data(df)

    X_train, X_test, y_train, y_test = split_train_test(df)

    return X_train, X_test, y_train, y_test


def create_input(row):

    return f"Merchant: {row['label']}; Price: {row['price']}"


def create_category(row) -> str:

    category_2 = row['category_2']
    if np.isna(category_2) or category_2 == "":
        category_2 = None

    if category_2 is None:
        return str(row['category_1'])
    else:
        return f"{row['category_1']} - {category_2}"


def process_data(df):
    df = df.rename(columns={
        "Date": "date",
        "Description": "label",
        "Debit": "price",
        "Type 1": "category_1",
        "Details": "category_2"
    })
    df = df.sort_values("date")

    df["input"] = df.apply(create_input, axis=1)
    df["category"] = df.apply(create_category, axis=1)

    return df


def split_train_test(df):
    X = df["input"]
    y = df["category"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    return X_train, X_test, y_train, y_test

