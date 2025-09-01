#!/bin/bash

# Wait for OceanBase container to be ready
# This script is similar to obdiag's wait_observer_run.sh

set -e

CONTAINER_NAME=${1:-"langchain_oceanbase_test"}
TIMEOUT=${2:-600}  # 10 minutes default timeout
SLEEP_INTERVAL=${3:-10}  # 10 seconds default sleep interval

echo "Waiting for OceanBase container '${CONTAINER_NAME}' to be ready..."
echo "Timeout: ${TIMEOUT} seconds, Check interval: ${SLEEP_INTERVAL} seconds"

start_time=$(date +%s)
elapsed=0

while [ $elapsed -lt $TIMEOUT ]; do
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    remaining=$((TIMEOUT - elapsed))
    
    echo "Elapsed: ${elapsed}s, Remaining: ${remaining}s"
    
    # Check if container is still running
    if ! docker ps | grep -q "${CONTAINER_NAME}"; then
        echo "ERROR: Container '${CONTAINER_NAME}' is not running!"
        echo "Container status:"
        docker ps -a | grep "${CONTAINER_NAME}" || true
        echo "Container logs:"
        docker logs "${CONTAINER_NAME}" || true
        exit 1
    fi
    
    # Check for boot success message
    if docker logs "${CONTAINER_NAME}" 2>&1 | grep -q "boot success!"; then
        echo "SUCCESS: OceanBase is ready!"
        
        # Additional verification - try to connect
        echo "Verifying database connection..."
        if docker exec "${CONTAINER_NAME}" obclient -h127.0.0.1 -P2881 -uroot@test -p'' -e "SELECT 1;" >/dev/null 2>&1; then
            echo "SUCCESS: Database connection verified!"
            exit 0
        else
            echo "WARNING: Boot success detected but connection failed, continuing to wait..."
        fi
    fi
    
    sleep $SLEEP_INTERVAL
done

echo "ERROR: OceanBase did not become ready within ${TIMEOUT} seconds!"
echo "Final container status:"
docker ps -a | grep "${CONTAINER_NAME}" || true
echo "Final container logs:"
docker logs "${CONTAINER_NAME}" || true
exit 1
