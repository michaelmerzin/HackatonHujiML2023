from __future__ import annotations
from typing import NoReturn

import sklearn.ensemble
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from base_estimator import BaseEstimator
import numpy as np


class AgodaCancellationEstimator(BaseEstimator):
    """
    An estimator for solving the Agoda Cancellation challenge
    """

    def __init__(self) -> AgodaCancellationEstimator:
        """
        Instantiate an estimator for solving the Agoda Cancellation challenge

        Parameters
        ----------


        Attributes
        ----------

        """
        super().__init__()
        self.models = list()

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an estimator for given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----

        """
        # self.model = LinearRegression().fit(X, y)
        # 0.28 loss
        # self.models.append(LogisticRegression(random_state=0, max_iter=40, solver="saga").fit(X, y))
        # 0.3 loss - 13
        # self.models.append(KNeighborsClassifier(n_neighbors=13).fit(X, y))
        # 0.21 - 40 estimators
        # self.models.append(AdaBoostClassifier(n_estimators=40, random_state=0).fit(X, y))
        # 0.22 loss
        # best k = 14
        self.models.append(DecisionTreeClassifier(max_depth=2).fit(X, y))
        # 0.26 loss
        # self.models.append(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 10, 2), random_state=1).fit(X, y))

        # self.models.append(LinearRegression().fit(X, y))

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        sums = np.zeros(len(X))
        for m in self.models:
            sums += m.predict(X)
        sums /= len(self.models)
        sums = np.round(sums)

        return sums

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under loss function
        """

        return sklearn.metrics.f1_score(y, self.predict(X))
