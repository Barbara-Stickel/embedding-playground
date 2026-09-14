import argparse
from pathlib import Path

import yaml

from data_processing import load_dataset
from embedding_eval import embed
from classifier import classify
from evaluation import evaluate, print_confusion_matrix


CONFIG_FILE = Path(__file__).resolve().parent.parent / "config.yaml"


def read_config(path: str | Path = CONFIG_FILE):
    with Path(path).open() as config_file:
        return yaml.safe_load(config_file)


def main():
    parser = argparse.ArgumentParser(description="Experiment with labeled text datasets")
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    args = parser.parse_args()
    config = read_config(args.config)

    # Existing expense configurations continue to work.
    dataset_config = config.get("dataset")
    if dataset_config is None:
        dataset_config = {
            "name": "expenses",
            "input": config["input"],
            "category": config["category"],
            **config["data_processing"],
        }
    data = load_dataset(dataset_config)

    embed(data, model_name=config["model"], cache_config=config.get("embedding_cache"))

    classify(data, config["classifier"])

    default_limit = None if dataset_config.get("name", "expenses") == "expenses" else 10
    result = evaluate(
        data,
        max_incorrect_predictions=config.get("evaluation", {}).get(
            "max_incorrect_predictions", default_limit,
        ),
    )

    print(f"result: {result[2]:%} - {result[0]} / {result[1]}")
    print_confusion_matrix(data)


if __name__ == "__main__":

    main()
