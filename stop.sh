#!/bin/bash

# ============================================
# Workflow Scheduler - Stop Script
# ============================================
# This script stops all running services
# Usage: ./stop.sh

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}ℹ ${1}${NC}"
}

print_success() {
    echo -e "${GREEN}✓ ${1}${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ ${1}${NC}"
}

echo ""
echo -e "${BLUE}════════════════════════════════════════${NC}"
echo -e "${BLUE}  Stopping Workflow Scheduler${NC}"
echo -e "${BLUE}════════════════════════════════════════${NC}"
echo ""

# Kill processes by port
kill_by_port() {
    local port=$1
    local name=$2
    
    local pid=$(lsof -ti:$port 2>/dev/null)
    if [ -n "$pid" ]; then
        kill -15 $pid 2>/dev/null && sleep 2
        if kill -0 $pid 2>/dev/null; then
            kill -9 $pid 2>/dev/null
        fi
        print_success "$name stopped (was on port $port)"
    else
        print_info "$name not running"
    fi
}

# Kill processes by name
kill_by_name() {
    local process_name=$1
    local display_name=$2
    
    local pids=$(pgrep -f "$process_name")
    if [ -n "$pids" ]; then
        echo "$pids" | xargs kill -15 2>/dev/null && sleep 2
        # Force kill if still running
        pgrep -f "$process_name" | xargs kill -9 2>/dev/null || true
        print_success "$display_name stopped"
    else
        print_info "$display_name not running"
    fi
}

# Stop FastAPI (port 8000)
kill_by_port 8000 "FastAPI backend"

# Stop Frontend (port 3000)
kill_by_port 3000 "Next.js frontend"

# Stop Celery workers
kill_by_name "celery.*worker" "Celery worker"

# Stop any remaining Python processes from workflow-backend
kill_by_name "workflow-backend.*python" "Backend processes"

# Stop any remaining Node processes from workflow-frontend
kill_by_name "workflow-frontend.*node" "Frontend processes"

echo ""
print_success "All services stopped"
echo ""

# Optional: Stop Redis if it was started by brew
read -p "Stop Redis too? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v brew &> /dev/null; then
        brew services stop redis 2>/dev/null && print_success "Redis stopped" || print_info "Redis not managed by brew"
    else
        print_warning "Redis must be stopped manually"
    fi
fi

echo ""
