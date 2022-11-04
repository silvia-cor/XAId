from __future__ import annotations

from abc import abstractmethod
from typing import Tuple

import numpy
from sklearn.linear_model import LogisticRegression

import ray
from tune_sklearn import TuneGridSearchCV as SearchAlgorithm
from sklearn.svm import LinearSVC


class LinearTrainer:
    def __init__(self, seed: int = 42, n_jobs: int = 1):
        self.seed = seed
        self.n_jobs = n_jobs
        ray.init(num_cpus=n_jobs, num_gpus=0)

    @abstractmethod
    def fit(self, data: numpy.ndarray, labels: numpy.ndarray, hyperparameters) -> Tuple[LinearTrainer, dict]:
        pass


class LogisticRegressorTrainer(LinearTrainer):
    def __init__(self, seed: int = 42, n_jobs: int = 1):
        super().__init__(seed, n_jobs)

    def fit(self, data: numpy.ndarray, labels: numpy.ndarray, hyperparameters) -> Tuple[LogisticRegression, dict]:
        """
        Train a logistic regressor.
        Args:
            data: Training data.
            labels: Training labels.
            hyperparameters: Hyperparameters dictionary for the logistic regressor.
        Returns:
            A trained Logistic Regressor, its parameters, and an evaluation dictionary
        """
        # default search hyperparameters
        multi_class = "multinomial" if numpy.unique(labels).size > 2 else "ovr"
        if len(hyperparameters) == 0:
            hyperparameters["C"] = [0.001, 0.01, 0.1, 1, 10]
            hyperparameters["penalty"] = ["l2"]

        search = SearchAlgorithm(LogisticRegression(multi_class=multi_class,
                                                     max_iter=2,
                                                     warm_start=True),
                                 param_grid=hyperparameters,
                                 cv=5, early_stopping=True,
                                 max_iters=2,
                                 refit=True,
                                 n_jobs=self.n_jobs)
        search.fit(data, labels)

        regressor = search.best_estimator
        configuration = {
            "C": regressor.C,
            "coefficients": regressor.coef_.tolist(),
            "intercept": regressor.intercept_.tolist()
        }

        return regressor, configuration


class LinearSVMTrainer(LinearTrainer):
    def __init__(self, seed: int = 42, n_jobs: int = 1):
        super().__init__(seed, n_jobs)

    def fit(self, data: numpy.ndarray, labels: numpy.ndarray, hyperparameters) -> Tuple[LogisticRegression, dict]:
        """
        Train a linear SVM.
        Args:
            data: Training data.
            labels: Training labels.
            hyperparameters: Hyperparameters dictionary for the logistic regressor.
        Returns:
            A trained Linear SVM and its parameters.
        """
        # default search hyperparameters
        if len(hyperparameters) == 0:
            hyperparameters["C"] = [0.001, 0.01, 0.1, 1, 10]

        multi_class = "crammer_singer" if numpy.unique(labels).size > 2 else "ovr"
        search = SearchAlgorithm(LinearSVC(multi_class=multi_class,
                                           max_iter=2),
                                 param_grid=hyperparameters,
                                 cv=3, max_iters=100000,
                                 n_jobs=self.n_jobs)
        search.fit(data, labels)

        svm = search.best_estimator

        configuration = {
            "C": svm.C,
            "coefficients": svm.coef_.tolist(),
            "intercept": svm.intercept_.tolist()
        }

        return svm, configuration
