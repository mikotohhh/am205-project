#!/bin/bash

export BASE_DIR=$(pwd)
export IMAGE_NAME="cutest"

# Build the Docker image
docker build -t $IMAGE_NAME .

# Run the Docker container
docker run --name mycutest -dt cutest \
-v "$BASE_DIR":/app \

docker exec -it mycutest bash -c 'pipenv run bash'

