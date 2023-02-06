from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import random
from datetime import datetime
from typing import Optional, Tuple

import numpy
import pandas
import torch
from transformers import AutoTokenizer, BertModel

from models.linear import LinearSVMTrainer, LogisticRegressorTrainer
from models.preprocessing import n_grams, preprocess_for_task
from models.transformer import VictoriaLoader
from validation import validate
from xai.probing import TransformerProber

__CW = os.path.dirname(os.path.abspath(__file__)) + "/"
__DATA_FOLDER = __CW + "../data/"
__DATASET_FOLDER = __DATA_FOLDER + "datasets/"
__MODELS_FOLDER = __DATA_FOLDER + "models/"
__OUTPUT_FOLDER = __DATA_FOLDER + "output/"


def train(dataset: str, algorithm: str, task: str,
          selected_author: Optional[int] = None,
          nr_authors: int = 10, positive_sampling_size: int | float = 1., negative_sampling_size: int | float = 1.,
          chr_n_grams: int = 3,
          hyperparameters: Optional[dict] = None,
          output: Optional[str] = None,
          seed: int = 42, n_jobs: int = 1, logging_level: str = "info") -> Tuple[object, dict, dict]:
    """
    Train a model with the given `algorithm` to perform `task` on the given `dataset`.
    Args:
        dataset: The dataset, currently only "victoria" is supported.
        algorithm: One of "svm", "lr".
        task: One of "sav" (Same-Author Verification), "av" (Authorship Verification), and "aa" (Authorship Attribution)
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

    if os.path.exists(output + ".cfg"):
        logging.error(f"{output}.cfg already exists, provide another output name.")
        sys.exit(-1)
    if os.path.exists(output + ".pickle"):
        logging.error(f"{output}.pickle already exists, provide another output name.")
        sys.exit(-1)
    hyperparameters_distributions = hyperparameters if hyperparameters is not None else dict()


    # set seed
    random.seed(seed)
    numpy.random.seed(seed)
    logging.debug(f"Reading dataset...")
    data = pandas.read_csv(__DATASET_FOLDER + "Gungor_2018_VictorianAuthorAttribution_data-train.csv",
                           encoding="latin-1")
    logging.debug(f"Done")
    authors = numpy.unique(data.author).squeeze()
    if selected_author is None:
        random_authors = numpy.random.choice(authors, nr_authors, replace=False) if nr_authors > 0\
                                                                                    else list(range(authors.size))
    else:
        random_authors = numpy.random.choice(authors, nr_authors - 1, replace=False) if nr_authors > 0\
                                                                                        else list(range(authors.size))
        random_authors = random_authors.tolist() + [selected_author]
    data = data[data["author"].isin(random_authors)]

    logging.info("Running...")
    logging.info(f"\tTask: {task}")
    logging.info(f"\tModel: {algorithm}")
    logging.info(f"\tModel file: {output}.pickle")
    logging.info(f"\tValidation file: {output}.validation.json")
    logging.info(f"\tModel parameters: {output}.model.json")
    logging.info(f"\tNumber of authors selected: {len(random_authors)}")
    logging.info(f"\tPositive samples: {positive_sampling_size}")
    logging.info(f"\tNegative samples: {negative_sampling_size}")
    logging.info(f"\tSeed: {seed}")
    logging.info(f"\tParallelism degree: {n_jobs}")


    #########
    # Train #
    #########
    # TODO: adjust with single model
    logging.debug(f"Preparing for task...")
    data = preprocess_for_task(data, task, positive_sampling_size, negative_sampling_size,
                               scale_labels=algorithm in ["lr", "svm"],
                               seed=seed)
    if algorithm == "lr":
        logging.debug(f"Creating {chr_n_grams}-grams on a {data.shape} frame.")
        data, labels, _ = n_grams(data, ngrams=chr_n_grams, task=task)
        logging.debug(f"Fitting Logistic Regressor...")
        trainer = LogisticRegressorTrainer(seed, n_jobs)
        model, optimal_hyperparameters = trainer.fit(data, labels, hyperparameters_distributions)
    elif algorithm == "svm":
        logging.debug(f"Creating {chr_n_grams}-grams on a {data.shape} frame.")
        data, labels, _ = n_grams(data, ngrams=chr_n_grams, task=task)
        logging.debug(f"Fitting Linear SVM...")
        trainer = LinearSVMTrainer(seed, n_jobs)
        model, optimal_hyperparameters = trainer.fit(data, labels, hyperparameters_distributions)
    elif algorithm == "transformer":
        logging.debug(f"Fitting Transformer...")
        num_labels = numpy.unique(data.label.values).size
        model = BertModel.from_pretrained("bert-base-uncased", num_labels=num_labels).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        victoria_dataset = VictoriaLoader(data.iloc[:1000], task, tokenizer, "cuda")

        prober = TransformerProber(model, tokenizer)
        logging.debug("Probing...")
        probing_results = prober.probe(victoria_dataset, probe_type="pos")


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
        "positive_samples": positive_sampling_size,
        "negative_samples": negative_sampling_size,
        "seed": seed
    }

    ################
    # Dump to disk #
    ################
    logging.info("Dumping info...")
    if algorithm != "transformer":
        with open(output + ".hyperparameters.json", "w") as log:
            json.dump(optimal_hyperparameters, log)
    else:
        torch.save(model.state_dict(), output + ".state_dict.pt")
    with open(output + ".pickle", "wb") as log:
        pickle.dump(model, log)
    with open(output + ".validation.json", "w") as log:
        json.dump(validation, log)
    with open(output + ".json", "w") as log:
        json.dump(config, log)

    return model, optimal_hyperparameters, validation
