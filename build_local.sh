#! /usr/bin/env sh

# Exit in case of error
set -e

docker-compose --env-file .env.local build
docker-compose --env-file .env.local down -v --remove-orphans # Remove possibly previous broken stacks left hanging after an error
docker-compose --env-file .env.local up -d
docker-compose exec backend bash -c "alembic upgrade head" # Database migrations
docker-compose exec backend bash -c "chmod +x ./app/initial_data.py && ./app/initial_data.py"  # Web admin account
docker-compose --env-file .env.local down

