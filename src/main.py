
from processing import get_data
from embedding import embed
from classifier import classify
from evaluation import evaluate

MODEL_NAME = "mini"

def main():
    X_train, X_test, y_train, y_test = get_data()

    embeddings_train = embed(X_train.to_list(), model_name=MODEL_NAME)
    embeddings_test = embed(X_test.to_list(), model_name=MODEL_NAME)

    y_test_predict = classify(embeddings_train, embeddings_test, y_train)

    result = evaluate(y_test, y_test_predict)

    print(f"result: {result[2]:%} - {result[0]} / {result[1]}")


if __name__ == "__main__":

    main()
