

def evaluate(y_test, y_test_predict):
    true_value = 0
    total_values = 0

    for y, y_predict in list(zip(y_test, y_test_predict)):
        if y == y_predict:
            true_value += 1
        else:
            print(f"y_predict: {y_predict}; y: {y}")

        total_values += 1

    percentage_correct = true_value / total_values if total_values != 0 else 0

    return true_value, total_values, percentage_correct
