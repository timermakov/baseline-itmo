#!/bin/bash

echo "OPENAI_API_KEY=${OPENAI_API_KEY}"
echo "SERPER_API_KEY=${SERPER_API_KEY}"

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080 --timeout 120