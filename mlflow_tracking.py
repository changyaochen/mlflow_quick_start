# A quick example of using the tracking API
import os
import time
import mlflow
import joblib

from typing import Tuple
from random import random, randint

from mlflow import log_metric, log_param, log_artifacts
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow import keras
from tensorflow.keras import layers, metrics

def train_and_log():
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

    experiment_name = 'Logistic regression'
    experiment = mlflow.get_experiment_by_name(name=experiment_name)
    if not experiment:
        experiment_id = mlflow.create_experiment(name=experiment_name)
        experiment = mlflow.get_experiment(experiment_id)

    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run() as run:
        print("Training...")
        for k, v in data_config.items():
            log_param(k, v)

        # train the model and log the metrics
        model = LogisticRegression()
        model.fit(X, y)
        log_metric('accuracy', accuracy_score(y, model.predict(X)))
        log_metric('auc', roc_auc_score(y, model.predict_proba(X)[:, 1]))

        # save the model artifact to both local and remote store
        model_path = os.path.join(
            'artifacts', experiment.experiment_id, run.info.run_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        joblib.dump(model, os.path.join(model_path, 'model.joblib'))
        log_artifacts(local_dir=model_path)
        print(f'Model trained, with run_id of {run.info.run_id}')

    # make a keras model
    experiment_name = 'Keras'
    experiment = mlflow.get_experiment_by_name(name=experiment_name)
    if not experiment:
        experiment_id = mlflow.create_experiment(name=experiment_name)
        experiment = mlflow.get_experiment(experiment_id)

    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run() as run:
        keras_model = keras.Sequential([
            layers.Dense(
                32,
                input_dim=X.shape[1],
                activation="relu",
                name="layer1"),
            layers.Dense(8, activation="relu", name="layer2"),
            layers.Dense(2, activation='softmax', name="layer3"),
        ])
        keras_model.compile(
            optimizer='sgd',
            loss='binary_crossentropy',
            metrics=['accuracy'])
        # autolog your metrics, parameters, and model
        N = int(0.8 * len(X))
        X_train, y_train = X[:N], y[:N]
        X_val, y_val = X[N:], y[N:]

        mlflow.tensorflow.autolog(every_n_iter=1)
        _ = keras_model.fit(
            X_train,
            y_train,
            epochs=20,
            batch_size=128,
            validation_data=(X_val, y_val))


if __name__ == "__main__":
    mlflow.set_tracking_uri('http://localhost:5000')
    train_and_log()
