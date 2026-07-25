#!/bin/bash
# Script to pull Ollama models inside the container

# Wait for Ollama service to be ready
echo "Waiting for Ollama service..."
until curl -s http://localhost:11434/api/tags > /dev/null; do
    sleep 2
done

echo "Ollama service is up! Pulling phi3:mini..."
docker exec -it $(docker ps -qf "ancestor=ollama/ollama") ollama pull phi3:mini

echo "Model pulled successfully."
