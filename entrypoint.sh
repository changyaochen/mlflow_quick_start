#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

mlflow server \
    --backend-store-uri $MLFLOW_BACKEND_STORE_URI \
    --default-artifact-root ${S3_PATH} \
    --host 0.0.0.0 \
    --port 5000