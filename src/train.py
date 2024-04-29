from __future__ import annotations

import json
import logging
import os
import sys

import numpy
from sklearn.feature_selection import SelectKBest, chi2

from models.linear import LinearSVMTrainer, LogisticRegressorTrainer
from models.preprocessing import n_grams
from models.transformer import Transformer

__CW = os.path.dirname(os.path.abspath(__file__)) + "/"
__DATA_FOLDER = __CW + "../data/"
__DATASET_FOLDER = __DATA_FOLDER + "datasets/"
__MODELS_FOLDER = __DATA_FOLDER + "models/"
__OUTPUT_FOLDER = __DATA_FOLDER + "output/"


def train(train_df, algorithm: str, task: str, chr_n_grams: int = 3, hyperparameters=None,
          seed: int = 42, n_jobs: int = 1):
    """
    Train a model with the given `algorithm` to perform `task` on the given `dataset`.
    Args:
        :param train_df: The training dataframe.
        :param algorithm: The algorithm for the model.
        :param task: The task to train the model on.
        :param chr_n_grams: Value for character n-grams for model training; defaults to 3.
        :param hyperparameters: Hyperparameter distributions for the model selection.
        :param seed: Random seed for the experiments; defaults to 42.
        :param n_jobs: Parallelism degree, defaults to 1. Use -1 to use all available resources.

    Returns:
        The trained model.
    """

    if hyperparameters is not None:
        if not os.path.exists(hyperparameters):
            logging.error(f"Path to hyperparameters object does not exist: {hyperparameters}.")
            sys.exit(-1)
        else:
            with open(hyperparameters, "r") as log:
                hyperparameters_distributions = json.load(log)
    else:
        hyperparameters_distributions = dict()

    if algorithm != 'transformer':
        logging.debug(f"Creating {chr_n_grams}-grams.")
        train_labels = train_df.label.values
        train_data, vectorizer = n_grams(train_df, ngrams=chr_n_grams, task=task)
        logging.debug(f"\tFeature selection on n_grams...")
        selector = SelectKBest(chi2, k=1000).fit(train_data, train_labels)
        train_data = selector.transform(train_data)
        if algorithm == "lr":
            logging.debug(f"Fitting Logistic Regressor...")
            trainer = LogisticRegressorTrainer(seed, n_jobs)
        else:
            logging.debug(f"Fitting Support Vector Machine...")
            trainer = LinearSVMTrainer(seed, n_jobs)
        model, optimal_hyperparameters = trainer.fit(train_data, train_labels, hyperparameters_distributions)
        logging.debug(f"Model fit.")
        return model, optimal_hyperparameters, vectorizer, selector
    else:
        logging.debug(f"Fitting Transformer...")
        num_labels = len(numpy.unique(train_df.label.values)) if task == "aa" else 2
        model = Transformer(num_labels=num_labels, seed=seed)
        model.fit(train_df, task)
        logging.debug(f"Model fit.")
        return model


