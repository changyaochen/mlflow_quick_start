# A quick example of using the tracking API
import os
import time
import mlflow
import joblib
from random import random, randint

from mlflow import log_metric, log_param, log_artifacts
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


def train_and_log(experiment_name: str):
    """A quick ML example, with MLflow."""
    # Create a random dataset for classification and log the parameters
    experiment = mlflow.get_experiment_by_name(name=experiment_name)
    if not experiment:
        experiment_id = mlflow.create_experiment(name=experiment_name)
        experiment = mlflow.get_experiment(experiment_id)

    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run() as run:
        print("Training...")
        random_seed = int(time.time_ns() // 1e9)
        data_config = dict(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            n_classes=2,
            random_state=random_seed)
        for k, v in data_config.items():
            log_param(k, v)

        X, y = make_classification(**data_config)

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


if __name__ == "__main__":
    mlflow.set_tracking_uri('http://localhost:5000')
    train_and_log(experiment_name='Logistic regression')
