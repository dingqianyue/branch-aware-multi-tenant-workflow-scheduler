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
# 2. Setup Backend
# ============================================
print_header "Setting Up Backend"

cd "$SCRIPT_DIR/workflow-backend"

# Check if venv exists
if [ ! -d "venv" ]; then
    print_warning "Virtual environment not found. Creating..."

    # Use Python 3.11 directly (required for InstanSeg)
    # Try multiple possible locations (Linux and macOS)
    PYTHON_CMD=""

    # Check common locations for python3.11
    for py_candidate in \
        "/usr/bin/python3.11" \
        "/bin/python3.11" \
        "/usr/local/bin/python3.11" \
        "/opt/homebrew/bin/python3.11" \
        "$(which python3.11 2>/dev/null)" \
        "/usr/bin/python3.10" \
        "/bin/python3.10" \
        "/usr/local/bin/python3.10" \
        "/opt/homebrew/bin/python3.10" \
        "$(which python3.10 2>/dev/null)"; do

        if [ -n "$py_candidate" ] && [ -x "$py_candidate" ]; then
            PYTHON_CMD="$py_candidate"
            break
        fi
    done

    if [ -z "$PYTHON_CMD" ]; then
        print_error "Python 3.11 or 3.10 not found. InstanSeg requires Python < 3.12"
        if command -v brew &> /dev/null; then
            print_error "Install with: brew install python@3.11"
        else
            print_error "Install with: apt-get install python3.11"
        fi
        exit 1
    fi

    print_info "Using $PYTHON_CMD for venv (compatible with InstanSeg)"
    $PYTHON_CMD -m venv venv
    print_success "Virtual environment created with $PYTHON_CMD"
fi

# Activate venv and install dependencies
print_info "Installing/updating backend dependencies..."
source venv/bin/activate

# Verify venv Python version
PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_info "Using Python $PYTHON_VERSION from venv"

# Always upgrade pip first (suppress notice)
python -m pip install --upgrade pip --quiet 2>/dev/null || true

# Check if packages are installed
if ! python -c "import fastapi" &> /dev/null; then
    print_info "Installing backend packages..."
    pip install -q "numpy==2.0.0"
    pip install -q fastapi uvicorn celery redis openslide-python \
                networkx opencv-python-headless scikit-image pillow \
                websockets python-multipart aiofiles requests
    # Install torch separately (large download)
    pip install -q torch torchvision
    print_success "Backend packages installed"
else
    print_success "Backend packages already installed"
fi

# Install InstanSeg (REQUIRED)
# InstanSeg requires Python >= 3.10 and < 3.12
if ! python -c "import instanseg" &> /dev/null; then
    PYTHON_MINOR=$(python -c 'import sys; print(sys.version_info.minor)')
    PYTHON_MAJOR=$(python -c 'import sys; print(sys.version_info.major)')

    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 10 ] && [ "$PYTHON_MINOR" -lt 12 ]; then
        print_info "Installing InstanSeg (required)..."
        if pip install -q git+https://github.com/instanseg/instanseg.git 2>&1 | grep -v "notice"; then
            print_success "InstanSeg installed successfully"
        else
            print_error "InstanSeg installation failed"
            exit 1
        fi
    else
        print_error "InstanSeg requires Python 3.10 or 3.11 (found $PYTHON_VERSION)"
        print_error "Please create venv with correct Python version"
        exit 1
    fi
else
    print_success "InstanSeg already installed"
fi

# ============================================
# 3. Setup Frontend
# ============================================
print_header "Setting Up Frontend"

cd "$SCRIPT_DIR/workflow-frontend"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    print_info "Installing frontend dependencies..."
    npm install --silent
    print_success "Frontend packages installed"
else
    print_success "Frontend packages already installed"
fi

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
