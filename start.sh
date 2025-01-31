#!/bin/bash

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080 