from typing import Tuple, Union

import numpy
import scipy.spatial.distance
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import random
import logging
from models.preprocessing import n_grams
import json


class LocalCounterfactualExplainer:
    """
    Provide local feature importance.
    """

    def __init__(self, model: Union[LogisticRegression, LinearSVC], labels: numpy.ndarray, data: numpy.ndarray,
                 task: str):
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

    def __init__(self, model: Union[LogisticRegression, LinearSVC], labels: numpy.ndarray, data: numpy.ndarray,
                 task: str):
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


def counter_factual_examples(model, train_df, test_df, test_data, test_labels, algorithm, task, seed, output,
                             n_factuals=1, **kwargs):
    random.seed(seed)
    train_labels = train_df.label.values
    train_texts = train_df.text.values if task != 'sav' else \
        [(row['text_A'], row['text_B']) for _, row in train_df.iterrows()]
    test_sample_index = random.choice(range(len(test_labels)))
    test_text_sample = test_df.at[test_sample_index, 'text'] if task != 'sav' else \
        (test_df.at[test_sample_index, 'text_A'], test_df.at[test_sample_index, 'text_B'])

    if algorithm != 'transformer':
        pred = model.predict([test_data[test_sample_index]])
        logging.info("\tExtracting features...")
        train_data, _, _ = n_grams(train_df, kwargs['char_n_grams'], task, kwargs['vectorizer'], kwargs['max_len'])
        train_data = kwargs['selector'].transform(train_data)
        test_data = test_data[test_sample_index]
    else:
        pred = model.predict(test_data.iloc[[test_sample_index]], task)
        logging.info("\tEncoding the data...")
        train_data = model.encode(train_df, task)
        test_data = model.encode(test_data.iloc[[test_sample_index]], task)[0]
    logging.info("\tGetting factuals...")
    factual_indices = numpy.argwhere(train_labels == pred).squeeze()
    logging.info("\tGetting counter-factuals...")
    counterfactual_indices = numpy.argwhere(train_labels != pred).squeeze()
    factual_candidates = [train_data[factual_index] for factual_index in factual_indices]
    counterfactual_candidates = [train_data[counterfactual_index] for counterfactual_index in counterfactual_indices]
    factual_distances = scipy.spatial.distance.cdist([test_data], factual_candidates, metric="euclidean").squeeze()
    counterfactual_distances = scipy.spatial.distance.cdist([test_data], counterfactual_candidates,
                                                            metric="euclidean").squeeze()
    min_factual_distances_indexes = [dist[0] for dist in sorted(enumerate(factual_distances),
                                                                key=lambda i: i[1])][:n_factuals]
    min_counterfactual_distances_indexes = [dist[0] for dist in sorted(enumerate(counterfactual_distances),
                                                                       key=lambda i: i[1])][:n_factuals]
    json_file = {'text_sample': test_text_sample,
                 'factuals': [(train_texts[idx], factual_distances[idx]) for idx in min_factual_distances_indexes],
                 'counterfactuals': [(train_texts[idx], counterfactual_distances[idx]) for idx
                                     in min_counterfactual_distances_indexes]}
    with open(output + '_factuals.json', 'w') as f:
        json.dump(json_file, f)
