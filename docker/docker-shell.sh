#!/bin/bash

export BASE_DIR=$(pwd)
export IMAGE_NAME="cutest"

# Build the Docker image
docker build -t $IMAGE_NAME .

# Run the Docker container with volume binding
docker run --name mycutest -dt \
    -v "$BASE_DIR:/app" \
    $IMAGE_NAME

# Ensure proper permissions (optional, if needed)
# docker exec -it mycutest bash -c 'chown -R app:app /app'

# Start the shell with pipenv
docker exec -it mycutest bash -c 'pipenv run bash'