# A quick example of using the tracking API
import os
import time
import mlflow
import joblib

import numpy as np
import xgboost as xgb

from abc import ABC
from typing import Dict, Tuple, Any
from random import random, randint

from mlflow import log_metric, log_param, log_artifacts
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow import keras
from tensorflow.keras import layers


class RunBase(ABC):
    """Abstract class for a run, only for demo purpose."""
    def __init__(
        self,
        experiment_name: str,
        X: np.ndarray = [],
        y: np.ndarray = [],
        data_config: Dict = {},
        params: Dict = {},
        **kwargs,
    ):
        """A simple base class that handles the data preparation and basic
        loggings.

        Args:
            experiment_name (str, optional): Name of the experiment.
            X (np.ndarray, optional): Input of data for training.
                Defaults to [].
            y (np.ndarray, optional): Label of the data for training.
                Defaults to [].
            data_config (Dict, optional): Configuration about the data.
                Defaults to {}.
            params (Dict, optional): Configuration (parameters) of the model.
                Defaults to {}.
        """
        self.data_config = data_config
        self.params = params
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name=self.experiment_name)

        N = int(0.8 * len(X))
        self.X_train, self.y_train = X[:N], y[:N]
        self.X_val, self.y_val = X[N:], y[N:]

    def train(self):
        with mlflow.start_run() as run:
            print(f"Running {self.experiment_name}...")
            for k, v in self.data_config.items():
                log_param(k, v)
            # some basic statistic about the data
            # log_param('n_features', self.X_train.shape[1])
            log_param('n_training_rows', len(self.X_train))
            log_param('n_validataion_rows', len(self.X_val))
            self._fit()
            print(f'Model trained, with run_id of {run.info.run_id}')
        return

    def _fit(self):
        raise NotImplementedError

class RunXGB(RunBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _fit(self):
        dtrain = xgb.DMatrix(data=self.X_train, label=self.y_train)
        dval = xgb.DMatrix(data=self.X_val, label=self.y_val)

        self.params.update({
            'objective': 'binary:logistic',
            'eval_metric': ['auc', 'logloss'],
        })
        mlflow.xgboost.autolog()
        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            evals=[(dtrain, 'train'), (dval, 'validation')],
            early_stopping_rounds=5
        )
        return


class RunKeras(RunBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _fit(self):
        mlflow.tensorflow.autolog(every_n_iter=1)
        keras_model = keras.Sequential([
            layers.Dense(
                32,
                input_dim=self.X_train.shape[1],
                activation="relu",
                name="layer1"),
            layers.Dense(8, activation="relu", name="layer2"),
            layers.Dense(1, activation='sigmoid', name="layer3"),
        ])
        keras_model.compile(
            optimizer='sgd',
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC'])
        _ = keras_model.fit(
            self.X_train,
            self.y_train,
            epochs=20,
            batch_size=128,
            validation_data=(self.X_val, self.y_val))
        return


def example():
    """A quick ML example, with MLflow."""
    # Create a random dataset for classification and log the parameters
    print('Making data...')
    random_seed = int(time.time_ns() // 1e9)
    data_config = dict(
        n_samples=100000,
        n_features=200,
        n_informative=120,
        n_classes=2,
        random_state=random_seed)
    X, y = make_classification(**data_config)
    print('Data made.')

    XGBModel_1 = RunXGB(
        'xgboost logistic regression',
        X=X,
        y=y,
        data_config=data_config,
        params={
            'n_estimators': 20,
            'max_depth': 5,
            'eta': 1})
    XGBModel_1.train()

    XGBModel_2 = RunXGB(
        'xgboost logistic regression',
        X=X,
        y=y,
        data_config=data_config,
        params={
            'n_estimators': 50,
            'max_depth': 3,
            'eta': 0.2})
    XGBModel_2.train()

    KerasModel = RunKeras(
        'Keras logistic regression',
        X=X,
        y=y,
        data_config=data_config,
        params={})
    KerasModel.train()


if __name__ == "__main__":
    mlflow.set_tracking_uri('http://localhost:5000')
    example()
