from pathlib import Path

import yaml

from processing import get_data
from embedding import embed
from classifier import classify
from evaluation import evaluate


CONFIG_FILE = Path(__file__).resolve().parent.parent / "config.yaml"


def read_config():
    with CONFIG_FILE.open() as config_file:
        return yaml.safe_load(config_file)


def main():
    config = read_config()

    X_train, X_test, y_train, y_test = get_data(
        config["input"], config["category"], config["data_processing"]
    )

    embeddings_train = embed(X_train.to_list(), model_name=config["model"])
    embeddings_test = embed(X_test.to_list(), model_name=config["model"])

    y_test_predict = classify(
        embeddings_train,
        embeddings_test,
        y_train,
        config["classifier"],
    )

    result = evaluate(y_test, y_test_predict)

    print(f"result: {result[2]:%} - {result[0]} / {result[1]}")


if __name__ == "__main__":

    main()
