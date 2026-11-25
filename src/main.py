import argparse
from pathlib import Path

import yaml

from classifier import classify
from data_processing import load_dataset
from embedding_eval import embed
from evaluation import category_recalls, evaluate
from reporting import (
    print_confusion_matrix,
    print_evaluation_summary,
    print_run_result,
    print_run_start,
)


CONFIG_FILE = Path(__file__).resolve().parent.parent / "config.yaml"


def read_config(path: str | Path = CONFIG_FILE):
    with Path(path).open() as config_file:
        return yaml.safe_load(config_file)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Experiment with labeled text datasets")
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    return parser.parse_args()


def run_experiment(config: dict, dataset_config: dict, seed: int, incorrect_limit: int | None):
    """Load, embed, classify, and evaluate one dataset split."""
    data = load_dataset({**dataset_config, "seed": seed})

    embed(data, model_name=config["model"], cache_config=config["embedding_cache"])
    classify(data, config["classifier"])

    result = evaluate(data, max_incorrect_predictions=incorrect_limit)
    recalls = category_recalls(data.test)
    balanced_accuracy = sum(recalls.values()) / len(recalls) if recalls else 0.0
    print_run_result(result, balanced_accuracy)

    return data, {
        "seed": seed,
        "accuracy": result[2],
        "balanced_accuracy": balanced_accuracy,
        "recalls": recalls,
    }


def run_experiments(config: dict, dataset_config: dict) -> None:
    """Run configured seeds and report their results and the final confusion matrix."""
    evaluation_config = config["evaluation"]
    seeds = evaluation_config["seeds"]
    incorrect_limit = evaluation_config["max_incorrect_predictions"]
    results = []
    for index, seed in enumerate(seeds, start=1):
        print_run_start(index, len(seeds), seed)
        # Show detailed mistakes only for the final run to keep repeated output small.
        limit = incorrect_limit if index == len(seeds) else 0
        data, result = run_experiment(config, dataset_config, seed, limit)
        results.append(result)

    if len(seeds) > 1:
        print_evaluation_summary(results)

    title = f"Confusion matrix (final run, seed {seeds[-1]})" if len(seeds) > 1 else "Confusion matrix"
    print_confusion_matrix(data, title=title)


def main():
    args = parse_arguments()
    config = read_config(args.config)
    dataset_config = config["dataset"]
    run_experiments(config, dataset_config)


if __name__ == "__main__":
    main()
