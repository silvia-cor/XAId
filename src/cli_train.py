from __future__ import annotations

import json
import logging
import os
import pickle
import random
import sys
from typing import Optional

import numpy.random
import pandas

import fire

from src.models.linear import LogisticRegressorTrainer, LinearSVMTrainer
from src.models.preprocessing import preprocess_for_task, n_grams
from src.validation import validate

__CW = os.path.dirname(os.path.abspath(__file__)) + "/"
__DATA_FOLDER = __CW + "../data/"
__DATASET_FOLDER = __DATA_FOLDER + "datasets/"
__MODELS_FOLDER = __DATA_FOLDER + "models/"
__OUTPUT_FOLDER = __DATA_FOLDER + "output/"


def train(dataset: str, algorithm: str, task: str,
          nr_authors: int = 10, sampling_size: int | float = 1., negative_sampling_size: int | float = 1.,
          chr_n_grams: int = 3,
          hyperparameters: Optional[str] = None,
          output: Optional[str] = None,
          seed: int = 42, n_jobs: int = 1, logging_level: str = "info"):
    """
    Train a model with the given `algorithm` to perform `task` on the given `dataset`.
    Args:
        dataset: The dataset, currently only "victoria" is supported.
        algorithm: One of "svm", "lr", "bert".
        task: One of "SAV" (Same-Author Verification), "AV" (Authorship Verification), and "AA" (Authorship Attribution)
        nr_authors: Number of authors to randomly select. Applies to AA tasks.
        sampling_size: Number (if integer) or percentage (if float) of positive samples for adaptation to AV tasks.
                        Defaults to 1.0 (use all samples).
        negative_sampling_size: Number (if integer) or percentage (if float) of negative samples for adaptation to
                                AV tasks. If percentage, the percentage is computed according to `sampling_size`.
                                To use all samples, set to -1. Defaults to 1.0 (use as many negative samples as
                                positive ones).
        chr_n_grams: Character n-grams for model training. Defaults to 3.
        hyperparameters: Path to additional hyperparameters, to be stored in a json file.
        output: Output file for configuration. The script generates a `output.cfg` (holding run configuration), a
                `output.pickle` holding the trained model, and a `output.results.json` holding validation results.
        seed: Random seed for the experiments. Defaults to 42.
        n_jobs: Parallelism degree, defaults to 1. Use -1 to use all available resources.
        logging_level: Logging level, defaults to "info".
    """
    if dataset != "victoria":
        raise ValueError(f"Dataset not supported: {dataset}")
    if algorithm not in ["svm", "lr"]:
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

    if os.path.exists(output + ".cfg"):
        logging.error(f"{output}.cfg already exists, provide another output name.")
        sys.exit(-1)
    if os.path.exists(output + ".pickle"):
        logging.error(f"{output}.pickle already exists, provide another output name.")
        sys.exit(-1)
    if hyperparameters is not None:
        if not os.path.exists(hyperparameters):
            logging.error(f"Path to hyperparameters object does not exist: {hyperparameters}.")
            sys.exit(-1)
        else:
            with open(hyperparameters, "r") as log:
                hyperparameters_distributions = json.load(log)
    else:
        hyperparameters_distributions = dict()

    logging.info("Running...")
    logging.info(f"\tTask: {task}")
    logging.info(f"\tModel: {algorithm}")
    logging.info(f"\tModel file: {output}.pickle")
    logging.info(f"\tValidation file: {output}.validation.json")
    logging.info(f"\tModel parameters: {output}.model.json")
    logging.info(f"\tNumber of authors selected: {nr_authors}")
    logging.info(f"\tPositive samples: {sampling_size}")
    logging.info(f"\tNegative samples: {negative_sampling_size}")
    logging.info(f"\tSeed: {seed}")
    logging.info(f"\tParallelism degree: {n_jobs}")

    # set seed
    random.seed(seed)
    numpy.random.seed(seed)

    data = pandas.read_csv(__DATASET_FOLDER + "Gungor_2018_VictorianAuthorAttribution_data-train.csv",
                           encoding="latin-1")
    random_authors = numpy.random.choice(numpy.unique(data.author), nr_authors, replace=False)
    data = data[data["author"].isin(random_authors)]
    logging.debug(f"Selected {nr_authors} authors.")

    #########
    # Train #
    #########
    data = preprocess_for_task(data, task, sampling_size, negative_sampling_size, seed=seed)
    if algorithm == "lr":
        logging.debug(f"Creating {chr_n_grams}-grams.")
        data, labels = n_grams(data, n_grams=chr_n_grams, task=task)
        logging.debug(f"Fitting Logistic Regressor...")
        trainer = LogisticRegressorTrainer(seed, n_jobs)
        model, hyperparameters = trainer.fit(data, labels, hyperparameters_distributions)
    elif algorithm == "svm":
        logging.debug(f"Creating {chr_n_grams}-grams.")
        data, labels = n_grams(data, n_grams=chr_n_grams, task=task)
        logging.debug(f"Fitting Linear SVM...")
        trainer = LinearSVMTrainer(seed, n_jobs)
        model, hyperparameters = trainer.fit(data, labels, hyperparameters_distributions)
    logging.debug(f"Model fit.")

    ############
    # Validate #
    ############
    logging.info("Validating...")
    validation = validate(model, data, labels)

    config = {
        "task": task,
        "model": algorithm,
        "model_file": f"{output}.pickle",
        "model_configuration": f"{output}.model.json",
        "number_of_authors_selected": nr_authors,
        "positive_samples": sampling_size,
        "negative_samples": negative_sampling_size,
        "seed": seed
    }

    ################
    # Dump to disk #
    ################
    logging.info("Dumping info...")
    with open(output + ".model.json", "w") as log:
        json.dump(hyperparameters, log)
    with open(output + ".pickle", "wb") as log:
        pickle.dump(model, log)
    with open(output + ".train.validation.json", "w") as log:
        json.dump(validation, log)
    with open(output + ".json", "w") as log:
        json.dump(config, log)


if __name__ == "__main__":
    fire.Fire(train)
