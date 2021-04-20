#! /usr/bin/env sh

# Use this script to build a production image and push to a container registry

# Exit in case of error
set -e

TAG=${TAG?Variable not set} \
sh ./scripts/build.sh

docker-compose -f docker-compose.yml push
