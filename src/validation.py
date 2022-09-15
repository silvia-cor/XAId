import numpy
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def validate(model, data: numpy.ndarray, labels: numpy.ndarray) -> dict:
    """
    Validate the given `model` against the given `data`.
    Args:
        model: The model to validate.
        data: The data to validate the model on.
        labels: The labels to validate the model on.

    Returns:
        A dictionary holding several validation metrics.
    """
    binary_task = True if numpy.unique(labels).size == 2 else False
    predicted_labels = model.predict(data)

    accuracy = accuracy_score(labels, predicted_labels)
    if binary_task:
        f1 = f1_score(labels, predicted_labels)
        precision = precision_score(labels, predicted_labels)
        recall = recall_score(labels, predicted_labels)

        validation = {
            "accuracy": accuracy,
            "f1_micro": f1,
            "precision": precision,
            "recall": recall,
        }
    else:
        f1_micro = f1_score(labels, predicted_labels, average="micro")
        f1_macro = f1_score(labels, predicted_labels, average="macro")
        precision_micro = precision_score(labels, predicted_labels, average="micro")
        precision_macro = precision_score(labels, predicted_labels, average="macro")
        recall_micro = recall_score(labels, predicted_labels, average="micro")
        recall_macro = recall_score(labels, predicted_labels, average="macro")

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
