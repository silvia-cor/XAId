from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import random
from datetime import datetime
from typing import Optional, Tuple, Dict

import numpy
import pandas
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, chi2

from models.linear import LinearSVMTrainer, LogisticRegressorTrainer
from models.preprocessing import n_grams, preprocess_for_task
from models.transformer import Transformer, AuthorshipDataloader
from validation import validate
from xai.feature_importance import irof

__CW = os.path.dirname(os.path.abspath(__file__)) + "/"
__DATA_FOLDER = __CW + "../data/"
__DATASET_FOLDER = __DATA_FOLDER + "datasets/"
__MODELS_FOLDER = __DATA_FOLDER + "models/"
__OUTPUT_FOLDER = __DATA_FOLDER + "output/"


def train(train_df, algorithm: str, task: str, analyzer: str = "char",
          chr_n_grams: int = 3, hyperparameters=None,
          seed: int = 42, n_jobs: int = 1):
    """
    Train a model with the given `algorithm` to perform `task` on the given `dataset`.
    Args:
        dataset: The dataset, currently only "victoria" is supported.
        algorithm: One of "svm", "lr".
        task: One of "sav" (Same-Author Verification), "av" (Authorship Verification), and "aa" (Authorship Attribution)
        analyzer: The features to use to train the linear model (algorithm in ["lr", "svm"]). One of "char", "pos",
        nr_authors: Number of authors to randomly select. Applies to AA tasks.
        positive_sampling_size: Number (if integer) or percentage (if float) of positive samples for adaptation to AV tasks.
                        Defaults to 1.0 (use all samples).
        negative_sampling_size: Number (if integer) or percentage (if float) of negative samples for adaptation to
                                AV tasks. If percentage, the percentage is computed according to `sampling_size`.
                                To use all samples, set to -1. Defaults to 1.0 (use as many negative samples as
                                positive ones).
        chr_n_grams: Character n-grams for model training. Defaults to 3.
        hyperparameters: Hyperparameter distributions for the model selection.
        output: Output file for configuration. The script generates a `output.cfg` (holding run configuration), a
                `output.pickle` holding the trained model, and a `output.results.json` holding validation results.
        seed: Random seed for the experiments. Defaults to 42.
        n_jobs: Parallelism degree, defaults to 1. Use -1 to use all available resources.
        logging_level: Logging level, defaults to "info".

    Returns:
        A triple (model, optimal hyperparameters, validation dictionary).
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
        train_data, vectorizer, max_len = n_grams(train_df, ngrams=chr_n_grams, task=task, analyzer=analyzer)
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
        return model, optimal_hyperparameters, vectorizer, selector, max_len
    else:
        logging.debug(f"Fitting Transformer...")
        num_labels = len(numpy.unique(train_df.label.values)) if task == "aa" else 2
        model = Transformer(num_labels=num_labels)
        # trainer.model = trainer.model.load_state_dict(torch.load("../../data/models/victoria_bert.pt"))
        model.fit(train_df, task)
        logging.debug(f"Model fit.")
        return model


