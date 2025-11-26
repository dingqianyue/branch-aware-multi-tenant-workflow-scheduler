#!/bin/bash

# ============================================
# Workflow Scheduler - Master Startup Script
# ============================================
# This script starts all services for the workflow scheduler
# Usage: ./start.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${BLUE}ℹ ${1}${NC}"
}

print_success() {
    echo -e "${GREEN}✓ ${1}${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ ${1}${NC}"
}

print_error() {
    echo -e "${RED}✗ ${1}${NC}"
}

print_header() {
    echo ""
    echo -e "${BLUE}════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}════════════════════════════════════════${NC}"
    echo ""
}

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

print_header "Workflow Scheduler Startup"

# ============================================
# 1. Check Prerequisites
# ============================================
print_info "Checking prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    exit 1
fi
print_success "Python 3 found: $(python3 --version)"

# Check Node.js
if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed"
    exit 1
fi
print_success "Node.js found: $(node --version)"

# Check Redis
if ! command -v redis-cli &> /dev/null; then
    print_error "Redis is not installed. Install with: brew install redis"
    exit 1
fi

# Test Redis connection
if ! redis-cli ping &> /dev/null; then
    print_warning "Redis is not running. Starting Redis..."
    if command -v brew &> /dev/null; then
        # macOS
        brew services start redis &> /dev/null || true
        sleep 2
    else
        # Linux
        redis-server --daemonize yes
        sleep 2
    fi

    if redis-cli ping &> /dev/null; then
        print_success "Redis started"
    else
        print_error "Redis failed to start"
        exit 1
    fi
else
    print_success "Redis is running"
fi

# ============================================
# 2. Setup Environment
# ============================================
print_header "Setting Up Environment"
./setup.sh

# ============================================
# 4. Create Log Directory
# ============================================
mkdir -p "$SCRIPT_DIR/logs"

# ============================================
# 5. Start Services
# ============================================
print_header "Starting Services"

# Function to cleanup on exit
cleanup() {
    print_warning "\nShutting down services..."
    
    # Kill all background jobs
    jobs -p | xargs -r kill 2>/dev/null || true
    
    # Wait for jobs to finish
    wait 2>/dev/null || true
    
    print_success "All services stopped"
    exit 0
}

# Trap Ctrl+C and call cleanup
trap cleanup SIGINT SIGTERM

# Start Celery Worker
print_info "Starting Celery worker..."
cd "$SCRIPT_DIR/workflow-backend"
source venv/bin/activate
celery -A app.celery_app worker --loglevel=info --pool=solo \
    > "$SCRIPT_DIR/logs/celery.log" 2>&1 &
CELERY_PID=$!
sleep 3

if kill -0 $CELERY_PID 2>/dev/null; then
    print_success "Celery worker started (PID: $CELERY_PID)"
else
    print_error "Celery worker failed to start. Check logs/celery.log"
    exit 1
fi

# Start FastAPI Backend
print_info "Starting FastAPI backend..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 \
    > "$SCRIPT_DIR/logs/fastapi.log" 2>&1 &
FASTAPI_PID=$!
sleep 3

if kill -0 $FASTAPI_PID 2>/dev/null; then
    print_success "FastAPI backend started (PID: $FASTAPI_PID)"
else
    print_error "FastAPI backend failed to start. Check logs/fastapi.log"
    cleanup
    exit 1
fi

# Wait for backend to be ready
print_info "Waiting for backend to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/ > /dev/null 2>&1; then
        print_success "Backend is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        print_error "Backend failed to respond. Check logs/fastapi.log"
        cleanup
        exit 1
    fi
    sleep 1
done

# Start Frontend
print_info "Starting Next.js frontend..."
cd "$SCRIPT_DIR/workflow-frontend"
npm run dev > "$SCRIPT_DIR/logs/frontend.log" 2>&1 &
FRONTEND_PID=$!
sleep 5

if kill -0 $FRONTEND_PID 2>/dev/null; then
    print_success "Frontend started (PID: $FRONTEND_PID)"
else
    print_error "Frontend failed to start. Check logs/frontend.log"
    cleanup
    exit 1
fi

# Wait for frontend to be ready
print_info "Waiting for frontend to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:3000/ > /dev/null 2>&1; then
        print_success "Frontend is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        print_error "Frontend failed to respond. Check logs/frontend.log"
        cleanup
        exit 1
    fi
    sleep 1
done

# ============================================
# 6. Display Status
# ============================================
print_header "All Services Running!"

echo ""
echo -e "${GREEN}🚀 Workflow Scheduler is ready!${NC}"
echo ""
echo -e "${BLUE}Services:${NC}"
echo -e "  • Celery Worker:  PID $CELERY_PID"
echo -e "  • FastAPI:        PID $FASTAPI_PID (http://localhost:8000)"
echo -e "  • Frontend:       PID $FRONTEND_PID (http://localhost:3000)"
echo ""
echo -e "${BLUE}URLs:${NC}"
echo -e "  • Frontend:       ${GREEN}http://localhost:3000${NC}"
echo -e "  • API Docs:       ${GREEN}http://localhost:8000/docs${NC}"
echo -e "  • API Stats:      ${GREEN}http://localhost:8000/scheduler/stats${NC}"
echo ""
echo -e "${BLUE}Logs:${NC}"
echo -e "  • Celery:         logs/celery.log"
echo -e "  • FastAPI:        logs/fastapi.log"
echo -e "  • Frontend:       logs/frontend.log"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo ""

# Keep script running and wait for all background jobs
wait
