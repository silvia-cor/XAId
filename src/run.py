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
from sklearn.feature_selection import f_classif

from models.linear import LinearSVMTrainer, LogisticRegressorTrainer
from models.preprocessing import n_grams, preprocess_for_task
from models.transformer import TransformerTrainer, VictoriaLoader
from validation import validate
from xai.probing import TransformerProber

__CW = os.path.dirname(os.path.abspath(__file__)) + "/"
__DATA_FOLDER = __CW + "../data/"
__DATASET_FOLDER = __DATA_FOLDER + "datasets/"
__MODELS_FOLDER = __DATA_FOLDER + "models/"
__OUTPUT_FOLDER = __DATA_FOLDER + "output/"

def probe(dataset: str, task: str,
          selected_author: Optional[int] = None,
          nr_authors: int = 10, positive_sampling_size: int | float = 1., negative_sampling_size: int | float = 1.,
          probe_type: str = "pos", k: int = 5, output: Optional[str] = None,
          seed: int = 42, n_jobs: int = 1, logging_level: str = "info") -> Dict:
    """Train a model with the given `algorithm` to perform `task` on the given `dataset`.

    Args:
        dataset: The dataset, currently only "victoria" is supported.
        task: One of "sav" (Same-Author Verification), "av" (Authorship Verification), and "aa" (Authorship Attribution)
        nr_authors: Number of authors to randomly select. Applies to AA tasks.
        positive_sampling_size: Number (if integer) or percentage (if float) of positive samples for adaptation to AV tasks.
                        Defaults to 1.0 (use all samples).
        negative_sampling_size: Number (if integer) or percentage (if float) of negative samples for adaptation to
                                AV tasks. If percentage, the percentage is computed according to `sampling_size`.
                                To use all samples, set to -1. Defaults to 1.0 (use as many negative samples as
                                positive ones).
        probe_type: The type of probe. Defaults to "pos".
        k: Number of probes to use. Defaults to 5.
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

    logging.info("Running...")
    logging.info(f"\tProbe file: {output}.probe.json")
    logging.info(f"\tProbe type: {probe_type}")
    logging.info(f"\tK: {k}")
    logging.info(f"\tNumber of authors selected: {nr_authors}")
    logging.info(f"\tPositive samples: {positive_sampling_size}")
    logging.info(f"\tNegative samples: {negative_sampling_size}")
    logging.info(f"\tSeed: {seed}")
    logging.info(f"\tParallelism degree: {n_jobs}")

    # set seed
    random.seed(seed)
    numpy.random.seed(seed)
    logging.debug(f"Reading dataset...")
    data = pandas.read_csv(__DATASET_FOLDER + "Gungor_2018_VictorianAuthorAttribution_data-train.csv",
                           encoding="latin-1")
    logging.debug(f"Done")
    authors = numpy.unique(data.author).squeeze()
    if selected_author is None:
        random_authors = numpy.random.choice(authors, nr_authors, replace=False) if nr_authors > 0 \
                                                                                else list(range(authors.size))
    else:
        random_authors = [selected_author]
    # data = data.groupby("author").head(20)
    data = data[data["author"].isin(random_authors)]
    logging.debug(f"\tSelected {nr_authors} authors.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", output_hidden_states=True)
    model.load_state_dict(torch.load("../data/models/victoria_bert.pt", map_location=torch.device(device)))
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    data.columns = ["text", "label"]
    victoria_dataset = VictoriaLoader(data, task, tokenizer, device)

    prober = TransformerProber(model, tokenizer)
    logging.debug("\tProbing...")
    probing_results, probes = prober.probe(victoria_dataset, probe_type=probe_type, k=k)

    with open(f"{output}.probe.{probe_type}.json", "w") as log:
        json.dump(probing_results, log)
    for author in probes:
        for p in probes[author]:
            with open(f"author{author}_probe_{probe_type}_{p}.pickle", "wb") as log:
                pickle.dump(p, log)

    return probing_results


def train(dataset: str, algorithm: str, task: str, analyzer: str = "char",
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
    logging.info(f"\tParallelism degree: {n_jobs}")

    # set seed
    random.seed(seed)
    numpy.random.seed(seed)
    logging.debug(f"Reading dataset...")
    data = pandas.read_csv(__DATASET_FOLDER + "Gungor_2018_VictorianAuthorAttribution_data-train.csv",
                           encoding="latin-1")
    logging.debug(f"Done")
    authors = numpy.unique(data.author).squeeze()
    random_authors = numpy.random.choice(authors, nr_authors, replace=False) if nr_authors > 0\
                                                                                else list(range(authors.size))
    data = data[data["author"].isin(random_authors)]
    logging.debug(f"\t\tSelected {nr_authors} authors.")

    #########
    # Train #
    #########
    logging.debug(f"Preparing for task...")
    data = preprocess_for_task(data, task, positive_sampling_size, negative_sampling_size,
                               scale_labels=algorithm in ["lr", "svm"],
                               seed=seed)
    if algorithm == "lr":
        logging.debug(f"Creating {chr_n_grams}-grams.")
        data, labels, _ = n_grams(data, ngrams=chr_n_grams, task=task, analyzer=analyzer)
        logging.debug(f"Feature selection on n_grams...")
        data = SelectKBest(f_classif, k=50).fit_transform(data, labels)
        logging.debug(f"Fitting Logistic Regressor...")
        trainer = LogisticRegressorTrainer(seed, n_jobs)
        model, optimal_hyperparameters = trainer.fit(data, labels, hyperparameters_distributions)
    elif algorithm == "svm":
        logging.debug(f"Creating {chr_n_grams}-grams.")
        data, labels, _ = n_grams(data, ngrams=chr_n_grams, task=task, analyzer=analyzer)
        logging.debug(f"Feature selection on n_grams...")
        data = SelectKBest(f_classif, k=50).fit_transform(data, labels)
        logging.debug(f"Fitting Linear SVM...")
        trainer = LinearSVMTrainer(seed, n_jobs)
        model, optimal_hyperparameters = trainer.fit(data, labels, hyperparameters_distributions)
    elif algorithm == "transformer":
        logging.debug(f"Fitting Transformer...")
        trainer = TransformerTrainer()
        trainer.model = trainer.model.load_state_dict(torch.load("../../data/models/victoria_bert.pt"))
        model = trainer.fit(data, data, task)


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