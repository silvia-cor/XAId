from typing import Tuple, Union

import numpy
import shap as shap
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from validation import validate
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


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


def irof(model, vectorizer, selector, test_data, test_labels, algorithm, task, output):
    selected_idxs = selector.get_support(indices=True)
    features_names = vectorizer.get_feature_names_out()[selected_idxs]
    idfs = vectorizer.idf_[selected_idxs]
    coefs = model.coef_[0]
    sorted_coefs_indexes = [coef[0] for coef in sorted(enumerate(coefs), key=lambda i:i[1], reverse=True)]
    biggest_coefs = [(features_names[i], coefs[i], idfs[i]) for i in sorted_coefs_indexes[:5]]
    smallest_coef = [(features_names[i], coefs[i], idfs[i]) for i in sorted_coefs_indexes[-5:]]
    print("Features with highest coef (feat_name, coef, idf value):")
    print(biggest_coefs)
    print("Features with lowest coef (feat_name, coef, idf value):")
    print(smallest_coef)
    corr, _ = pearsonr(coefs, idfs)
    print(f'Pearson correlation among coefs and idf: {corr:.3f}')
    abs_coefs = abs(coefs)
    sorted_abs_coefs_indexes = [coef[0] for coef in sorted(enumerate(abs_coefs), key=lambda i:i[1], reverse=True)]
    irof_validations = []
    for sorted_coefs_index in sorted_abs_coefs_indexes:
        model.coef_[0][sorted_coefs_index] = 0
        irof_validations.append(validate(model, test_data, test_labels, algorithm, task))
    validation_df = pandas.DataFrame(irof_validations)
    validation_df = validation_df.reset_index().melt('index', var_name='cols', value_name='vals')
    sns.lineplot(x="index", y="vals", data=validation_df, hue='cols')
    plt.legend()
    plt.xlabel("# features removed")
    plt.ylabel("Performance")
    plt.savefig(output + '_irof.png')


