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

    data = get_data(
        config["input"], config["category"], config["data_processing"]
    )

    embed(data, model_name=config["model"])

    classify(data, config["classifier"])

    result = evaluate(data)

    print(f"result: {result[2]:%} - {result[0]} / {result[1]}")


if __name__ == "__main__":

    main()
