import itertools
from typing import Tuple, Union

import numpy
import shap as shap
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from validation import validate
import pandas
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import random


class GlobalExplainer:
    """
    Provide local feature importance.
    """

    def __init__(self, model: Union[LogisticRegression, LinearSVC], labels: numpy.ndarray, data: numpy.ndarray,
                 task: str):
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


def irof(model, vectorizer, selector, test_data, test_labels, algorithm, task, seed, output):
    logging.debug("Performing IROF experiment...")
    selected_idxs = selector.get_support(indices=True)
    features_names = vectorizer.get_feature_names_out()[selected_idxs]
    # idfs = vectorizer.idf_[selected_idxs]
    coefs = model.coef_[0]
    print("Intercept:", model.intercept_)
    sorted_coefs_indexes = [coef[0] for coef in sorted(enumerate(coefs), key=lambda i: i[1], reverse=True)]
    biggest_coefs = [(features_names[i], coefs[i]) for i in sorted_coefs_indexes[:5]]
    smallest_coef = [(features_names[i], coefs[i]) for i in sorted_coefs_indexes[-5:]]
    print("Features with highest coef (feat_name, coef):")
    print(biggest_coefs)
    print("Features with lowest coef (feat_name, coef):")
    print(smallest_coef)
    # corr, _ = pearsonr(coefs, idfs)
    # print(f'Pearson correlation among coefs and idf: {corr:.3f}')
    abs_coefs = abs(coefs)
    sorted_abs_coefs_indexes = [coef[0] for coef in sorted(enumerate(abs_coefs), key=lambda i: i[1], reverse=True)]
    sorted_f1s = _zeroing_coef_validations(copy.deepcopy(model), sorted_abs_coefs_indexes, test_data, test_labels,
                                           algorithm, task)
    df_sorted = pandas.DataFrame({'sorted_f1': sorted_f1s})
    random.seed(seed)
    random_f1s = []
    for i in range(10):
        random.shuffle(sorted_abs_coefs_indexes)
        random_f1s.append(
            _zeroing_coef_validations(copy.deepcopy(model), sorted_abs_coefs_indexes, test_data, test_labels,
                                      algorithm, task))
    sns.lineplot(x=df_sorted.index, y='sorted_f1', data=df_sorted, label='sorted_coefs')
    df_random = pandas.DataFrame(random_f1s).reset_index().melt('index', var_name='step', value_name='f1s')
    sns.lineplot(x="step", y="f1s", data=df_random, errorbar='sd', label='random_coefs')
    plt.xlabel("# features removed")
    plt.ylabel("F1")
    plt.legend()
    plt.show()
    plt.savefig(output + '_feat_irof.png')
    plt.show()
    plt.cla()


def _zeroing_coef_validations(model, coefs_indexes, test_data, test_labels, algorithm, task):
    f1s = []
    for coefs_index in coefs_indexes:
        model.coef_[0][coefs_index] = 0
        f1s.append(validate(model, test_data, test_labels, algorithm, task)['f1'])
    return f1s


def local_explanation(model, vectorizer, selector, selected_samples, selected_labels, output):
    logging.debug("Getting local explanation...")
    selected_idxs = selector.get_support(indices=True)
    features_names = vectorizer.get_feature_names_out()[selected_idxs]
    coefs = model.coef_[0]
    sorted_coefs_indexes = [coef[0] for coef in sorted(enumerate(coefs), key=lambda i: i[1], reverse=True)]
    biggest_coefs = sorted_coefs_indexes[:5]
    smallest_coef = sorted_coefs_indexes[-5:]
    feats_names = [features_names[i] for i in biggest_coefs] + [features_names[i] for i in smallest_coef]
    df = pandas.DataFrame(
        [[sample[idx] * coefs[idx] for idx in biggest_coefs + smallest_coef] for sample in selected_samples],
        columns=[feat_name.replace(' ', '_') for feat_name in feats_names])
    df['class'] = [f'{i + 1}_SameAuthor' if selected_label == 1 else f'{i + 1}_DifferentAuthor'
                   for i, selected_label in enumerate(selected_labels)]
    df = df.reset_index().melt(id_vars=['class', 'index'], var_name='feats', value_name='vals', ignore_index=False)
    ax = sns.barplot(x='feats', y="vals", data=df, hue='class')
    hatches = ['-' if 'Same' in cl else '//' for cl in list(df['class'])]
    # Loop over the bars
    for bars, hatch in zip(ax.containers, hatches):
        # Set a different hatch for each group of bars
        for bar in bars:
            bar.set_hatch(hatch)
    plt.ylabel("feat_value * coef")
    ax.legend(title='Examples')
    plt.savefig(output + '_feat_local.png')
    plt.show()
    plt.cla()
