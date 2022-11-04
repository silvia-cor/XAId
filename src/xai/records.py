from typing import Tuple, Union

import numpy
import scipy.spatial.distance
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


class LocalCounterfactualExplainer:
    """
    Provide local feature importance.
    """
    def __init__(self, model: Union[LogisticRegression, LinearSVC], labels: numpy.ndarray, data: numpy.ndarray, task: str):
        self.model = model
        self.labels = labels
        self.task = task
        self.tr_data = data

    def explain(self, data: numpy.ndarray, distance: str = "euclidean") -> Tuple[numpy.ndarray, int, numpy.ndarray]:
        """
        Explain the model on the given input.
        Args:
            data: The input data.
            distance: The distance to use to look for a counterfactual.

        Returns:
            The model prediction and a feature importance vector.
        """
        # AV, SAV
        if self.task in ("sav", "av"):
            coefficients = self.model.coef_
        else:
            prediction_index = self.model.predict(data).argmax()
            coefficients = self.model.coef_[prediction_index]
        # last feature is the author, set its importance to 0
        if self.task == "av":
            coefficients[-1] = 0

        if self.task in ("sav", "av"):
            prediction = self.model.predict(data).item()
        else:
            prediction_index = self.model.predict(data).argmax()
            prediction = self.labels[prediction_index]

        counterfactual_indices = numpy.argwhere(self.tr_data[-1] != prediction).squeeze()
        counterfactual_candidates = self.tr_data[counterfactual_indices]
        distances = scipy.spatial.distance.cdist(data.reshape(1, -1), counterfactual_candidates,
                                                 metric=distance).squeeze()
        counterfactual_index = distances.argmin()
        actual_counterfactual_index = counterfactual_indices[counterfactual_index]
        counterfactual_sample = self.tr_data[actual_counterfactual_index][:-1]
        counterfactual_prediction = self.tr_data[actual_counterfactual_index][-1]
        counterfactual_distance = distances[counterfactual_index]

        return counterfactual_sample, counterfactual_prediction, counterfactual_distance


class LocalFactualExplainer:
    """
       Provide local feature importance.
    """
    def __init__(self, model: Union[LogisticRegression, LinearSVC], labels: numpy.ndarray, data: numpy.ndarray, task: str):
        self.model = model
        self.labels = labels
        self.task = task
        self.tr_data = data

    def explain(self, data: numpy.ndarray, distance: str = "euclidean") -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Explain the model on the given input.
        Args:
            data: The input data.
            distance: The distance to use to look for a similar instance.

        Returns:
            The most similar factual instance and the distance between it and the original instance.
        """
        # AV, SAV
        if self.task in ("sav", "av"):
            coefficients = self.model.coef_
        else:
            prediction_index = self.model.predict(data).argmax()
            coefficients = self.model.coef_[prediction_index]
        # last feature is the author, set its importance to 0
        if self.task == "av":
            coefficients[-1] = 0

        if self.task in ("sav", "av"):
            prediction = self.model.predict(data).item()
        else:
            prediction_index = self.model.predict(data).argmax()
            prediction = self.labels[prediction_index]

        factual_indices = numpy.argwhere(self.tr_data[-1] == prediction).squeeze()
        factual_candidates = self.tr_data[factual_indices]
        distances = scipy.spatial.distance.cdist(data.reshape(1, -1), factual_candidates,
                                                 metric=distance).squeeze()
        factual_index = distances.argmin()
        actual_factual_index = factual_indices[factual_index]
        factual_sample = self.tr_data[actual_factual_index][:-1]
        factual_distance = distances[factual_index]

        return factual_sample, factual_distance
