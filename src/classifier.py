import numpy as np
from sklearn.neighbors import NearestNeighbors

from data_types import ExpenseDataset, ExpenseRecord


class NearestNeighborsClassifier:

    def __init__(
        self, training_embeddings, n_neighbors: int = 10, metric: str = "cosine"
    ):

        self.training_embeddings = training_embeddings
        self.n_neighbors = n_neighbors
        self.metric = metric

        self.fit(training_embeddings)

    def fit(self, training_embeddings):

        self.knn = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
        )

        self.knn.fit(training_embeddings)

    def classify(self, new_embedding):

        distances, indices = self.knn.kneighbors(
            new_embedding.reshape(1, -1), return_distance=True
        )

        similarities = 1 - distances[0]

        return similarities, indices[0]


CLASSIFIER_FACTORIES = {
    "nearest_neighbors": NearestNeighborsClassifier,
}


def create_classifier(name: str, training_embeddings, **options):
    try:
        classifier_class = CLASSIFIER_FACTORIES[name]
    except KeyError as error:
        choices = ", ".join(CLASSIFIER_FACTORIES)
        raise ValueError(
            f"Unknown classifier '{name}'. Available choices: {choices}"
        ) from error

    return classifier_class(training_embeddings, **options)


def get_embeddings(records: list[ExpenseRecord]) -> np.ndarray:
    embeddings = [record.embedding for record in records]
    if any(embedding is None for embedding in embeddings):
        raise ValueError("All records must be embedded before classification")

    return np.stack(embeddings)


def classify(data: ExpenseDataset, classifier_config: dict) -> None:
    training_embeddings = get_embeddings(data.train)

    classifier = create_classifier(
        classifier_config["name"],
        training_embeddings,
        **classifier_config.get("options", {}),
    )

    for record in data.test:
        if record.embedding is None:
            raise ValueError("All records must be embedded before classification")

        similarities, indices = classifier.classify(record.embedding)

        scores = {}
        for index, similarity in zip(indices, similarities):
            category = data.train[index].category
            scores[category] = scores.get(category, 0) + 1 #similarity

        record.predicted_category = max(scores, key=scores.get)
