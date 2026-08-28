from sklearn.neighbors import NearestNeighbors


class NearestNeighborsClassifier:

    def __init__(self, training_embeddings):

        self.training_embeddings = training_embeddings

        self.fit(training_embeddings)

    def fit(self, training_embeddings):

        self.knn = NearestNeighbors(
            n_neighbors=10,
            metric="cosine"
        )

        self.knn.fit(training_embeddings)

    def classify(self, new_embedding):

        distances, indices = self.knn.kneighbors(
            new_embedding.reshape(1, -1), return_distance=True
        )

        similarities = 1 - distances[0]

        return similarities, indices[0]


def classify(embeddings_train, embeddings_test, y_train):

    classifier = NearestNeighborsClassifier(embeddings_train)

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
