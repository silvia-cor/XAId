from __future__ import annotations

import json
import logging
import os
from typing import Optional
import random
import numpy

import fire
from datetime import datetime
from train import train
from models.preprocessing import preprocess_for_task, n_grams, create_MedLatin
import pickle
from models.transformer import Transformer
from validation import validate
from xai.feature_importance import irof, local_explanation
from xai.records import counter_factual_examples
from xai.probing import probe
import torch

__CW = os.path.dirname(os.path.abspath(__file__)) + "/"
__DATA_FOLDER = __CW + "../data/"
__DATASET_FOLDER = __DATA_FOLDER + "datasets/"
__MODELS_FOLDER = __DATA_FOLDER + "models/"
__OUTPUT_FOLDER = __DATA_FOLDER + "output/"


def run(dataset: str, algorithm: str, task: str,
        positive_sampling_size: int | float = 1., negative_sampling_size: int | float = 1.,
        char_n_grams: int = 3,
        hyperparameters: Optional[str] = None,
        output: Optional[str] = None, probe_type=None, k=5, min_length=5, max_length=10,
        seed: int = 42, n_jobs: int = 1, logging_level: str = "info"):
    """
    Train a model with the given `algorithm` to perform `task` on the given `dataset`.
    Args:
        :param dataset: The dataset, currently only "medlatin" is supported.
        :param algorithm: The algorithm for the model, currently "svm", "lr" and "transformer" are supported.
        :param task: The task for the model; one of "sav" (Same-Author Verification), "av" (Authorship Verification), and "aa" (Authorship Attribution).
        :param positive_sampling_size: Number (if integer) or percentage (if float) of positive samples for adaptation to AV tasks.
                        Defaults to 1.0 (use all samples).
        :param negative_sampling_size: Number (if integer) or percentage (if float) of negative samples for adaptation to
                                AV tasks. If percentage, the percentage is computed according to `sampling_size`.
                                To use all samples, set to -1. Defaults to 1.0 (use as many negative samples as
                                positive ones).
        :param char_n_grams: Value for character n-grams for model training. Defaults to 3.
        :param hyperparameters: Path to additional hyperparameters, to be stored in a json file.
        :param output: Output file for configuration. The script generates the following files: `output.config.json` (holding run configuration),
                `output.parameters.json` (holding the parameters and hyperparameters of the trained model),
                `output.pickle` (holding the trained model), and a `output.validation.json` (holding validation results).
        :param probe_type: Type of probe for the probing experiment.
        :param k: Number of probes to create for the probing experiment.
        :param min_length: Minimum chain length for "pos" probes for the probing experiment.
        :param max_length: Maximum chain length for "pos" probes for the probing experiment.
        :param seed: Random seed for the experiments. Defaults to 42.
        :param n_jobs: Parallelism degree, defaults to 1. Use -1 to use all available resources.
        :param logging_level: Logging level, defaults to "info".
    """

    if dataset != "medlatin":
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

    config = {
        "task": task,
        "model": algorithm,
        "model_file": f"{output}.pickle",
        "positive_samples": positive_sampling_size,
        "negative_samples": negative_sampling_size,
        "seed": seed
    }
    if probe_type is not None:
        config['probe_type'] = probe_type
    if probe_type in ['pos', 'sq']:
        config['k'] = k
        config['min_length'] = min_length
        config['max_length'] = max_length

    logging.info("Running...")
    logging.info(f"\tTask: {task}")
    logging.info(f"\tModel: {algorithm}")
    logging.info(f"\tModel file: {output}.pickle")
    logging.info(f"\tValidation file: {output}.validation.json")
    logging.info(f"\tPositive samples: {positive_sampling_size}")
    logging.info(f"\tNegative samples: {negative_sampling_size}")
    if probe_type is not None:
        logging.info(f"\tProbing type: {probe_type}")
    if probe_type in ['pos', 'sq']:
        logging.info(f"\tProbing for k chains: {k}")
        logging.info(f"\tMin length for chains: {min_length}")
        logging.info(f"\tMax length for chains: {max_length}")
    logging.info(f"\tSeed: {seed}")
    logging.info(f"\tParallelism degree: {n_jobs}")  # NB: putting -1 rise a Fire error

    # set seed
    random.seed(seed)
    numpy.random.seed(seed)
    logging.debug(f"Creating dataset...")
    # the only dataset currently supported is MedLatin
    data = create_MedLatin(__DATA_FOLDER + 'MedLatin')

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
    if os.path.exists(output + ".config.json"):
        logging.debug(f"\tTrained model found!")
        if algorithm != 'transformer':
            with open(output + ".pickle", "rb") as f:
                model, vectorizer, selector = pickle.load(f)
        else:
            num_labels = len(numpy.unique(train_df.label.values)) if task == "aa" else 2
            model = Transformer(state_dict=output + ".state_dict.pt", num_labels=num_labels)
    else:
        if algorithm != 'transformer':
            model, optimal_hyperparameters, vectorizer, selector = \
                train(train_df, algorithm, task, char_n_grams, hyperparameters=hyperparameters, seed=seed,
                      n_jobs=n_jobs)
            with open(output + ".parameters.json", "w") as log:
                json.dump(optimal_hyperparameters, log)
            with open(output + ".pickle", "wb") as log:
                pickle.dump((model, vectorizer, selector), log)
        else:
            model = train(train_df, algorithm, task, seed=seed)
            torch.save(model.model.state_dict(), output + ".state_dict.pt")

    ############
    # Validate #
    ############
    logging.info("Validating...")
    if algorithm == 'transformer':
        test_data = test_df
    else:
        test_data, _ = n_grams(test_df, char_n_grams, task, vectorizer)
        test_data = selector.transform(test_data)
    validation = validate(model, test_data, test_df.label.values, algorithm, task)

    ################
    # Dump to disk #
    ################
    logging.info("Dumping info...")
    with open(output + ".validation.json", "w") as log:
        json.dump(validation, log)
    with open(output + ".config.json", "w") as log:
        json.dump(config, log)

    ################
    # XAId #
    ################

    if task == 'sav' and algorithm == 'svm':
        train_data, _ = n_grams(train_df, char_n_grams, task, vectorizer)
        logging.info("Performing XAI: feature importance...")
        # IROF
        irof(model, vectorizer, selector, test_data, test_df.label.values, algorithm, task, seed, output)
        # local explanation
        n_samples = 2
        selected_samples = random.sample([sample for i, sample in enumerate(test_data)
                                          if test_df.label.values[i] == 0], n_samples) + \
                           random.sample([sample for i, sample in enumerate(test_data)
                                          if test_df.label.values[i] == 1], n_samples)
        selected_labels = [0] * n_samples + [1] * n_samples
        local_explanation(model, vectorizer, selector, selected_samples, selected_labels, output)
    if task == 'av':
        logging.info("Performing XAI: factuals and counter-factuals...")
        if algorithm == 'svm':
            counter_factual_examples(model, train_df, test_df, test_data, test_df.label.values, algorithm, task, seed,
                                     output, n_factuals=3, char_n_grams=char_n_grams,
                                     vectorizer=vectorizer, selector=selector)
        else:
            counter_factual_examples(model, train_df, test_df, test_data, test_df.label.values, algorithm, task, seed,
                                     output, n_factuals=3)
    if task == 'aa' and algorithm == 'transformer':
        logging.info("Performing XAI: probing...")
        probe(model, train_df, task, output, probe_type=probe_type, k=k, min_len=min_length, max_len=max_length,
              seed=seed, n_jobs=n_jobs)


if __name__ == "__main__":
    fire.Fire()
