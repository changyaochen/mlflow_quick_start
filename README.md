# mlflow_quick_start
A quick start to launch MLflow service.

To run it locally, change the `.env.example` file to `.env`, and modify the values within accordingly.

The backend database used is `postgres`, ran as a local container. One also needs to set up the artifact store. Currently, we choose AWS s3, therefore, one needs to have the local `~/.aws/credentials` file configurated properly.

Once the configuration is, run:
~~~sh
$ docker-compose up -d
~~~
and the MLflow UI will be ready at `localhost:5000`.

To test the tracking API, run:
~~~py
python mlflow_tracking.py
~~~
and the experiment should show up in the MLflow UI.