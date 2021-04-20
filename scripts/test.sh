#! /usr/bin/env sh

# Exit in case of error
set -e

DOMAIN=backend \
SMTP_HOST="" \
TRAEFIK_PUBLIC_NETWORK_IS_EXTERNAL=false \
INSTALL_DEV=true \
USE_TEST_DB=true \
docker-compose \
-f docker-compose.yml \
config > docker-stack.yml

docker-compose -f docker-stack.yml build
docker-compose -f docker-stack.yml down -v --remove-orphans # Remove possibly previous broken stacks left hanging after an error
docker-compose -f docker-compose.override.yml -f docker-stack.yml up -d

# Hack to wait until migrations are run
sleep 5

# Create test DB
docker-compose -f docker-stack.yml exec db sh -c 'psql -U ${POSTGRES_USER} -c "DROP DATABASE IF EXISTS ${POSTGRES_DB}_test;" && psql  -U ${POSTGRES_USER} -c "CREATE DATABASE ${POSTGRES_DB}_test WITH TEMPLATE ${POSTGRES_DB};"'

docker-compose -f docker-stack.yml exec -T backend bash /app/tests-start.sh "$@"

# Tear down test DB
docker-compose -f docker-stack.yml exec db sh -c 'psql -U ${POSTGRES_USER} -c "DROP DATABASE IF EXISTS ${POSTGRES_DB}_test;"'

docker-compose -f docker-stack.yml down -v --remove-orphans
