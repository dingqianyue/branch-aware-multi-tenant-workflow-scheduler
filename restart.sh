#!/bin/bash

echo "Cleaning up and restarting services..."

# Kill existing processes
pkill -f celery
pkill -f uvicorn
pkill -f "npm run dev"

# Fix Redis
redis-cli CONFIG SET stop-writes-on-bgsave-error no
redis-cli FLUSHALL

# Clean logs
rm -f logs/*.log

# Restart
./start.sh
