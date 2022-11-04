from typing import Tuple, Union

import numpy
import shap as shap
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


class GlobalExplainer:
    """
    Provide local feature importance.
    """
    def __init__(self, model: Union[LogisticRegression, LinearSVC], labels: numpy.ndarray, data: numpy.ndarray, task: str):
        self.model = model
        self.labels = labels
        self.task = task
        self.tr_data = data

    def explain(self, data: numpy.ndarray, scope: str = "weights") -> Tuple[int, numpy.ndarray]:
        """
        Explain the model on the given input.
        Args:
            data: The input data.
            scope: Explanation algorithm to use: one of "weights" or "shap".

        Returns:
            The model prediction and a feature importance vector.
        """
        if scope == "shap":
            explainer = shap.KernelExplainer(self.model.predict_proba, numpy.ndarray([data]), link="logit")
            coefficients = explainer.shap_values(numpy.ndarray([data]), nsamples=1)
        else:
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

        return prediction, coefficients
