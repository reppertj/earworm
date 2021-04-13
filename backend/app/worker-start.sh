#! /usr/bin/env bash
set -e

python /app/app/celeryworker_pre_start.py

if [ "$RUN" == "" ]
  then
    celery -A app.worker worker --loglevel=INFO -Q main-queue
  else
    eval $RUN
fi
