from __future__ import annotations

import json
import logging
import os
from typing import Optional
import random
import numpy
import pandas

import fire
from datetime import datetime
from train import train
from models.preprocessing import preprocess_for_task, n_grams
import pickle
from models.transformer import Transformer
from validation import validate
from xai.feature_importance import irof
from xai.records import counter_factual_examples
from xai.probing import probe
import torch

__CW = os.path.dirname(os.path.abspath(__file__)) + "/"
__DATA_FOLDER = __CW + "../data/"
__DATASET_FOLDER = __DATA_FOLDER + "datasets/"
__MODELS_FOLDER = __DATA_FOLDER + "models/"
__OUTPUT_FOLDER = __DATA_FOLDER + "output/"


def run(dataset: str, algorithm: str, task: str, analyzer: str = "char",
        nr_authors: int = 10, positive_sampling_size: int | float = 1.,
        negative_sampling_size: int | float = 1.,
        char_n_grams: int = 3,
        hyperparameters: Optional[str] = None,
        output: Optional[str] = None, probe_type="pos",
        seed: int = 42, n_jobs: int = 1, logging_level: str = "info"):
    """
    Train a model with the given `algorithm` to perform `task` on the given `dataset`.
    Args:
        dataset: The dataset, currently only "victoria" is supported.
        algorithm: One of "svm", "lr".
        task: One of "sav" (Same-Author Verification), "av" (Authorship Verification), and "aa" (Authorship Attribution).
        analyzer: The features to use to train the linear model (algorithm in ["lr", "svm"]). One of "char", "pos", 
        nr_authors: Number of authors to randomly select. Applies to AA tasks.
        positive_sampling_size: Number (if integer) or percentage (if float) of positive samples for adaptation to AV tasks.
                        Defaults to 1.0 (use all samples).
        negative_sampling_size: Number (if integer) or percentage (if float) of negative samples for adaptation to
                                AV tasks. If percentage, the percentage is computed according to `sampling_size`.
                                To use all samples, set to -1. Defaults to 1.0 (use as many negative samples as
                                positive ones).
        char_n_grams: Character n-grams for model training. Defaults to 3.
        hyperparameters: Path to additional hyperparameters, to be stored in a json file.
        output: Output file for configuration. The script generates a `output.cfg` (holding run configuration), a
                `output.pickle` holding the trained model, and a `output.results.json` holding validation results.
        seed: Random seed for the experiments. Defaults to 42.
        n_jobs: Parallelism degree, defaults to 1. Use -1 to use all available resources.
        logging_level: Logging level, defaults to "info".
    """

    if dataset != "victoria":
        raise ValueError(f"Dataset not supported: {dataset}")
    if algorithm not in ["svm", "lr", "transformer"]:
        raise ValueError(f"Algorithm not supported: {algorithm}")
    if task not in ["sav", "av", "aa"]:
        raise ValueError(f"Task not supported: {task}")
        # set debug level
    if logging_level == "info":
        logging.basicConfig(level=logging.INFO)
    if logging_level == "debug":
        logging.basicConfig(level=logging.DEBUG)
    if logging_level == "warning":
        logging.basicConfig(level=logging.WARNING)
    if logging_level == "error":
        logging.basicConfig(level=logging.ERROR)
    if logging_level == "critical":
        logging.basicConfig(level=logging.CRITICAL)

    if output is None:
        output = str(datetime.now())

    # if os.path.exists(output + ".cfg"):
    #     logging.error(f"{output}.cfg already exists, provide another output name.")
    #     sys.exit(-1)
    # if os.path.exists(output + ".pickle"):
    #     logging.error(f"{output}.pickle already exists, provide another output name.")
    #     sys.exit(-1)

    config = {
        "task": task,
        "model": algorithm,
        "model_file": f"{output}.pickle",
        "model_configuration": f"{output}.model.json",
        "number_of_authors_selected": nr_authors,
        "positive_samples": positive_sampling_size,
        "negative_samples": negative_sampling_size,
        "seed": seed
    }

    logging.info("Running...")
    logging.info(f"\tTask: {task}")
    logging.info(f"\tModel: {algorithm}")
    logging.info(f"\tModel file: {output}.pickle")
    logging.info(f"\tValidation file: {output}.validation.json")
    logging.info(f"\tModel parameters: {output}.model.json")
    logging.info(f"\tNumber of authors selected: {nr_authors}")
    logging.info(f"\tPositive samples: {positive_sampling_size}")
    logging.info(f"\tNegative samples: {negative_sampling_size}")
    logging.info(f"\tSeed: {seed}")
    logging.info(f"\tParallelism degree: {n_jobs}")  # NB: putting -1 rise a Fire error

    # set seed
    random.seed(seed)
    numpy.random.seed(seed)
    logging.debug(f"Reading dataset...")
    data = pandas.read_csv(__DATASET_FOLDER + "Gungor_2018_VictorianAuthorAttribution_data-train.csv",
                           encoding="latin-1")
    logging.debug(f"Done")
    authors = numpy.unique(data.author).squeeze()
    random_authors = numpy.random.choice(authors, nr_authors, replace=False) if nr_authors > 0 \
        else list(range(authors.size))
    data = data[data["author"].isin(random_authors)]
    logging.debug(f"\tSelected {nr_authors} authors.")

    logging.debug(f"Preparing for task...")
    if task == 'aa':
        train_df, test_df, label_encoder = preprocess_for_task(data, task, positive_sampling_size,
                                                               negative_sampling_size, seed=seed)
    else:
        train_df, test_df = preprocess_for_task(data, task, positive_sampling_size,
                                                negative_sampling_size, seed=seed)

    ############
    # Train #
    ############
    if os.path.exists(output + ".json"):
        logging.debug(f"\tTrained model found!")
        if algorithm != 'transformer':
            with open(output + ".pickle", "rb") as f:
                model, vectorizer, selector, max_len = pickle.load(f)
        else:
            num_labels = len(numpy.unique(train_df.label.values)) if task == "aa" else 2
            model = Transformer(state_dict=output + ".state_dict.pt", num_labels=num_labels)
    else:
        if algorithm != 'transformer':
            model, optimal_hyperparameters, vectorizer, selector, max_len = \
                train(train_df, algorithm, task, analyzer, char_n_grams, hyperparameters=hyperparameters, seed=seed,
                      n_jobs=n_jobs)
            with open(output + ".hyperparameters.json", "w") as log:
                json.dump(optimal_hyperparameters, log)
            with open(output + ".pickle", "wb") as log:
                pickle.dump((model, vectorizer, selector, max_len), log)
        else:
            model = train(train_df, algorithm, task, analyzer, char_n_grams, hyperparameters=hyperparameters, seed=seed,
                          n_jobs=n_jobs)
            torch.save(model.model.state_dict(), output + ".state_dict.pt")

    ############
    # Validate #
    ############
    logging.info("Validating...")
    if algorithm == 'transformer':
        test_data = test_df
    else:
        test_data, _, _ = n_grams(test_df, char_n_grams, task, analyzer, vectorizer, max_len)
        test_data = selector.transform(test_data)
    validation = validate(model, test_data, test_df.label.values, algorithm, task)

    ################
    # Dump to disk #
    ################
    logging.info("Dumping info...")
    with open(output + ".validation.json", "w") as log:
        json.dump(validation, log)
    with open(output + ".json", "w") as log:
        json.dump(config, log)

    ################
    # XAId #
    ################

    if task == 'sav' and algorithm == 'svm':
        logging.info("Performing XAI: feature importance...")
        irof(model, vectorizer, selector, test_data, test_df.label.values, algorithm, task, output)
    if task == 'av':
        logging.info("Performing XAI: factuals and counter-factuals...")
        if algorithm == 'svm':
            counter_factual_examples(model, train_df, test_df, test_data, test_df.label.values, algorithm, task, seed,
                                     output, n_factuals=3, char_n_grams=char_n_grams, analyzer=analyzer,
                                     vectorizer=vectorizer, max_len=max_len, selector=selector)
        else:
            counter_factual_examples(model, train_df, test_df, test_data, test_df.label.values, algorithm, task, seed,
                                     output, n_factuals=3)
    if task == 'aa' and algorithm == 'transformer':
        k = 5 if probe_type == 'pos' else 3
        probe(model, train_df, task, label_encoder, output, probe_type=probe_type, k=k, seed=seed, n_authors=nr_authors)


if __name__ == "__main__":
    fire.Fire()
