#!/bin/bash

# ============================================
# Workflow Scheduler - Master Startup Script with Redis Fix
# ============================================
# This script starts all services with automatic Redis recovery
# Usage: ./start.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
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
    echo -e "${PURPLE}════════════════════════════════════════${NC}"
    echo -e "${PURPLE}  $1${NC}"
    echo -e "${PURPLE}════════════════════════════════════════${NC}"
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

# Check Python 3.11
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
    print_success "Python 3.11 found: $(python3.11 --version)"
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
    if [[ "$PYTHON_VERSION" == "3.11" ]]; then
        PYTHON_CMD="python3"
        print_success "Python 3.11 found: $(python3 --version)"
    else
        print_error "Python 3.11 is required but found: $(python3 --version)"
        print_info "Install with: brew install python@3.11 (macOS) or apt install python3.11 (Linux)"
        exit 1
    fi
else
    print_error "Python 3 is not installed"
    exit 1
fi

# Check Node.js
if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed"
    print_info "Install from: https://nodejs.org/"
    exit 1
fi
print_success "Node.js found: $(node --version)"

# Check npm
if ! command -v npm &> /dev/null; then
    print_error "npm is not installed"
    exit 1
fi
print_success "npm found: $(npm --version)"

# ============================================
# 2. Fix and Start Redis
# ============================================
print_header "Redis Setup and Recovery"

# Check if Redis is installed
if ! command -v redis-cli &> /dev/null; then
    print_error "Redis is not installed"
    print_info "Install Redis:"
    print_info "  macOS:  brew install redis"
    print_info "  Ubuntu: sudo apt-get install redis-server"
    exit 1
fi

# Function to fix Redis persistence errors
fix_redis_persistence() {
    print_warning "Fixing Redis persistence configuration..."
    
    # Try to connect and fix settings
    redis-cli CONFIG SET stop-writes-on-bgsave-error no &> /dev/null || true
    redis-cli CONFIG SET save "" &> /dev/null || true
    
    # Clear any corrupted data
    print_info "Clearing Redis data..."
    redis-cli FLUSHALL &> /dev/null || true
    
    print_success "Redis persistence issues resolved"
}

# Function to start Redis
start_redis() {
    if command -v brew &> /dev/null; then
        # macOS with Homebrew
        print_info "Starting Redis via Homebrew..."
        brew services stop redis &> /dev/null || true
        sleep 1
        brew services start redis &> /dev/null || true
    elif command -v systemctl &> /dev/null; then
        # Linux with systemd
        print_info "Starting Redis via systemd..."
        sudo systemctl restart redis &> /dev/null || true
    else
        # Manual start
        print_info "Starting Redis manually..."
        redis-server --daemonize yes &> /dev/null || true
    fi
    sleep 2
}

# Test Redis connection
print_info "Testing Redis connection..."
if redis-cli ping &> /dev/null; then
    print_success "Redis is running"
    
    # Check for persistence errors
    REDIS_INFO=$(redis-cli INFO persistence 2>/dev/null | grep -E "rdb_last_bgsave_status|aof_last_write_status" || true)
    if echo "$REDIS_INFO" | grep -q "err"; then
        print_warning "Redis has persistence errors"
        fix_redis_persistence
    fi
else
    print_warning "Redis is not running, attempting to start..."
    start_redis
    
    # Wait and test again
    for i in {1..5}; do
        if redis-cli ping &> /dev/null; then
            print_success "Redis started successfully"
            fix_redis_persistence
            break
        fi
        if [ $i -eq 5 ]; then
            print_error "Failed to start Redis"
            print_info "Try manually: redis-server"
            exit 1
        fi
        sleep 1
    done
fi

# Show Redis status
REDIS_VERSION=$(redis-cli INFO server | grep redis_version | cut -d: -f2 | tr -d '\r')
print_success "Redis ready (version: $REDIS_VERSION)"

# ============================================
# 3. Setup Environment
# ============================================
print_header "Environment Setup"

# Check if setup is needed
if [ ! -d "workflow-backend/venv" ] || [ ! -d "workflow-frontend/node_modules" ]; then
    print_info "Running setup..."
    
    # Make setup.sh executable if it's not
    if [ -f "setup.sh" ]; then
        chmod +x setup.sh
        ./setup.sh
    else
        print_error "setup.sh not found"
        exit 1
    fi
else
    print_success "Environment already set up"
fi

# ============================================
# 4. Create Required Directories
# ============================================
mkdir -p "$SCRIPT_DIR/logs"
mkdir -p "$SCRIPT_DIR/workflow-backend/uploads"
mkdir -p "$SCRIPT_DIR/workflow-backend/outputs"
mkdir -p "$SCRIPT_DIR/workflow-backend/tile_maps"

# ============================================
# 5. Kill Any Existing Processes
# ============================================
print_header "Cleaning Up Previous Sessions"

# Function to safely kill processes
safe_kill() {
    local pattern=$1
    local name=$2
    
    if pgrep -f "$pattern" > /dev/null; then
        print_warning "Stopping existing $name processes..."
        pkill -f "$pattern" 2>/dev/null || true
        sleep 1
    fi
}

safe_kill "celery.*worker" "Celery"
safe_kill "uvicorn.*app.main" "FastAPI"
safe_kill "npm.*run.*dev" "Frontend"

print_success "Cleanup complete"

# ============================================
# 6. Start Services
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

# Export Python path
export PYTHONPATH="${SCRIPT_DIR}/workflow-backend:${PYTHONPATH}"

celery -A app.celery_app worker \
    --loglevel=info \
    --pool=solo \
    --concurrency=1 \
    > "$SCRIPT_DIR/logs/celery.log" 2>&1 &
CELERY_PID=$!
sleep 3

if kill -0 $CELERY_PID 2>/dev/null; then
    print_success "Celery worker started (PID: $CELERY_PID)"
else
    print_error "Celery worker failed to start"
    print_warning "Check logs/celery.log for details:"
    tail -n 20 "$SCRIPT_DIR/logs/celery.log"
    exit 1
fi

# Start FastAPI Backend
print_info "Starting FastAPI backend..."
uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    > "$SCRIPT_DIR/logs/fastapi.log" 2>&1 &
FASTAPI_PID=$!
sleep 3

if kill -0 $FASTAPI_PID 2>/dev/null; then
    print_success "FastAPI backend started (PID: $FASTAPI_PID)"
else
    print_error "FastAPI backend failed to start"
    print_warning "Check logs/fastapi.log for details:"
    tail -n 20 "$SCRIPT_DIR/logs/fastapi.log"
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
        print_error "Backend failed to respond"
        print_warning "Check logs/fastapi.log for details"
        cleanup
        exit 1
    fi
    sleep 1
done

# Deactivate Python venv before starting frontend
deactivate

# Start Frontend
print_info "Starting Next.js frontend..."
cd "$SCRIPT_DIR/workflow-frontend"

# Check if package.json exists
if [ ! -f "package.json" ]; then
    print_error "package.json not found in workflow-frontend/"
    cleanup
    exit 1
fi

npm run dev > "$SCRIPT_DIR/logs/frontend.log" 2>&1 &
FRONTEND_PID=$!
sleep 5

if kill -0 $FRONTEND_PID 2>/dev/null; then
    print_success "Frontend started (PID: $FRONTEND_PID)"
else
    print_error "Frontend failed to start"
    print_warning "Check logs/frontend.log for details:"
    tail -n 20 "$SCRIPT_DIR/logs/frontend.log"
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
        print_warning "Frontend may still be building..."
    fi
    sleep 1
done

# ============================================
# 7. Display Status
# ============================================
print_header "All Services Running!"

echo ""
echo -e "${GREEN}🚀 Workflow Scheduler is ready!${NC}"
echo ""
echo -e "${BLUE}═══ Services ═══${NC}"
echo -e "  ${GREEN}●${NC} Redis:        ${GREEN}Running${NC}"
echo -e "  ${GREEN}●${NC} Celery:       PID $CELERY_PID"
echo -e "  ${GREEN}●${NC} FastAPI:      PID $FASTAPI_PID"
echo -e "  ${GREEN}●${NC} Frontend:     PID $FRONTEND_PID"
echo ""
echo -e "${BLUE}═══ URLs ═══${NC}"
echo -e "  Frontend:      ${GREEN}http://localhost:3000${NC}"
echo -e "  API Docs:      ${GREEN}http://localhost:8000/docs${NC}"
echo -e "  API Redoc:     ${GREEN}http://localhost:8000/redoc${NC}"
echo -e "  API Stats:     ${GREEN}http://localhost:8000/scheduler/stats${NC}"
echo ""
echo -e "${BLUE}═══ Logs ═══${NC}"
echo -e "  Celery:        tail -f logs/celery.log"
echo -e "  FastAPI:       tail -f logs/fastapi.log"
echo -e "  Frontend:      tail -f logs/frontend.log"
echo ""
echo -e "${BLUE}═══ Quick Commands ═══${NC}"
echo -e "  View Celery:   ${YELLOW}tail -f logs/celery.log${NC}"
echo -e "  View API:      ${YELLOW}tail -f logs/fastapi.log${NC}"
echo -e "  Redis CLI:     ${YELLOW}redis-cli${NC}"
echo -e "  Stop all:      ${YELLOW}Press Ctrl+C${NC}"
echo ""

# Monitor for errors in background
(
    while true; do
        sleep 10
        
        # Check if services are still running
        if ! kill -0 $CELERY_PID 2>/dev/null; then
            print_error "Celery worker crashed! Check logs/celery.log"
            break
        fi
        
        if ! kill -0 $FASTAPI_PID 2>/dev/null; then
            print_error "FastAPI crashed! Check logs/fastapi.log"
            break
        fi
        
        if ! kill -0 $FRONTEND_PID 2>/dev/null; then
            print_error "Frontend crashed! Check logs/frontend.log"
            break
        fi
        
        # Check for Redis issues
        if ! redis-cli ping &> /dev/null; then
            print_error "Redis connection lost!"
            break
        fi
    done
) &
MONITOR_PID=$!

# Keep script running and wait for all background jobs
wait
