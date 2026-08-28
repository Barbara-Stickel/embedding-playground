from sklearn.neighbors import NearestNeighbors


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


def classify(
    embeddings_train,
    embeddings_test,
    y_train,
    classifier_config: dict,
):

    classifier = create_classifier(
        classifier_config["name"],
        embeddings_train,
        **classifier_config.get("options", {}),
    )

    y_test_predict = []

    for new_embedding in embeddings_test:
        similarities, indices = classifier.classify(new_embedding)

        scores = {}
        for index, similarity in zip(indices, similarities):
            category = y_train.iloc[index]
            scores[category] = scores.get(category, 0) + 1 #similarity

        best_category = max(scores, key=scores.get)

        y_test_predict.append(best_category)

    return y_test_predict
