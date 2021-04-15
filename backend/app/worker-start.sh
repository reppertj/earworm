#! /usr/bin/env bash
set -e

python /app/app/celeryworker_pre_start.py

if [ "$RUN" == "" ]
  then
    celery --app app.worker worker --loglevel=INFO -Q main-queue -c 1 --without-gossip -O fair --prefetch-multiplier 1
  else
    eval $RUN
fi
