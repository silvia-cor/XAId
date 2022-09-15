from __future__ import annotations

import json
import logging
import os
import sys
from typing import Optional

import fire

from train import train

__CW = os.path.dirname(os.path.abspath(__file__)) + "/"
__DATA_FOLDER = __CW + "../data/"
__DATASET_FOLDER = __DATA_FOLDER + "datasets/"
__MODELS_FOLDER = __DATA_FOLDER + "models/"
__OUTPUT_FOLDER = __DATA_FOLDER + "output/"


def cli_train(dataset: str, algorithm: str, task: str,
              nr_authors: int = 10, positive_sampling_size: int | float = 1., negative_sampling_size: int | float = 1.,
              chr_n_grams: int = 3,
              hyperparameters: Optional[str] = None,
              output: Optional[str] = None,
              seed: int = 42, n_jobs: int = 1, logging_level: str = "info"):
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
        hyperparameters: Path to additional hyperparameters, to be stored in a json file.
        output: Output file for configuration. The script generates a `output.cfg` (holding run configuration), a
                `output.pickle` holding the trained model, and a `output.results.json` holding validation results.
        seed: Random seed for the experiments. Defaults to 42.
        n_jobs: Parallelism degree, defaults to 1. Use -1 to use all available resources.
        logging_level: Logging level, defaults to "info".
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

    train(dataset, algorithm, task,
          nr_authors,
          positive_sampling_size=positive_sampling_size, negative_sampling_size=negative_sampling_size,
          chr_n_grams=chr_n_grams,
          hyperparameters=hyperparameters_distributions,
          output=output,
          seed=seed, n_jobs=n_jobs, logging_level=logging_level)


if __name__ == "__main__":
    fire.Fire(train)
