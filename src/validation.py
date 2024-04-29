import numpy
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def validate(model, test_data, test_labels, algorithm: str, task: str):
    """
    Validate the given `model` against the given `data`.
    Args:
        :param model: The model to validate.
        :param test_data : The data to validate the model on.
        :param test_labels : The labels to validate the model on.
        :param algorithm : The algorithm the model is based on.
        :param task : The task the model is trained for.
    Returns:
        A dictionary holding several validation metrics.
    """

    binary_task = True if numpy.unique(test_labels).size == 2 else False

    if algorithm == 'transformer':
        predicted_labels = model.predict(test_data, task)
    else:
        predicted_labels = model.predict(test_data)

    accuracy = accuracy_score(test_labels, predicted_labels)
    if binary_task:
        f1 = f1_score(test_labels, predicted_labels)
        precision = precision_score(test_labels, predicted_labels)
        recall = recall_score(test_labels, predicted_labels)

        validation = {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }
    else:
        f1_micro = f1_score(test_labels, predicted_labels, average="micro")
        f1_macro = f1_score(test_labels, predicted_labels, average="macro")
        precision_micro = precision_score(test_labels, predicted_labels, average="micro")
        precision_macro = precision_score(test_labels, predicted_labels, average="macro")
        recall_micro = recall_score(test_labels, predicted_labels, average="micro")
        recall_macro = recall_score(test_labels, predicted_labels, average="macro")

        validation = {
            "accuracy": accuracy,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "precision_micro": precision_micro,
            "precision_macro": precision_macro,
            "recall_micro": recall_micro,
            "recall_macro": recall_macro
        }

    return validation
