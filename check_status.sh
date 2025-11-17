#!/bin/bash
# Quick status check and log viewer

echo "═══════════════════════════════════════════"
echo "  Workflow Scheduler Status Check"
echo "═══════════════════════════════════════════"
echo ""

# Check if services are running
echo "Checking services..."
echo ""

# Check Celery
if pgrep -f "celery.*worker" > /dev/null; then
    echo "✓ Celery worker is running"
    CELERY_PID=$(pgrep -f "celery.*worker" | head -1)
    echo "  PID: $CELERY_PID"
else
    echo "✗ Celery worker is NOT running"
fi

echo ""

# Check FastAPI
if lsof -ti:8000 > /dev/null 2>&1; then
    echo "✓ FastAPI backend is running on port 8000"
    FASTAPI_PID=$(lsof -ti:8000 | head -1)
    echo "  PID: $FASTAPI_PID"
else
    echo "✗ FastAPI backend is NOT running on port 8000"
fi

echo ""

# Check Frontend
if lsof -ti:3000 > /dev/null 2>&1; then
    echo "✓ Frontend is running on port 3000"
    FRONTEND_PID=$(lsof -ti:3000 | head -1)
    echo "  PID: $FRONTEND_PID"
else
    echo "✗ Frontend is NOT running on port 3000"
fi

echo ""
echo "═══════════════════════════════════════════"
echo "  Recent Logs"
echo "═══════════════════════════════════════════"
echo ""

# Check for log files
if [ -f "logs/celery.log" ]; then
    echo "━━━ Last 20 lines of Celery log ━━━"
    tail -20 logs/celery.log
    echo ""
fi

if [ -f "logs/fastapi.log" ]; then
    echo "━━━ Last 20 lines of FastAPI log ━━━"
    tail -20 logs/fastapi.log
    echo ""
fi

# Check for errors in logs
if [ -f "logs/celery.log" ]; then
    ERROR_COUNT=$(grep -i "error\|exception\|failed" logs/celery.log | wc -l)
    if [ "$ERROR_COUNT" -gt 0 ]; then
        echo "⚠ Found $ERROR_COUNT errors in Celery log"
        echo "━━━ Recent errors ━━━"
        grep -i "error\|exception\|failed" logs/celery.log | tail -10
        echo ""
    fi
fi

# Test API endpoint
echo "━━━ Testing API endpoint ━━━"
if curl -s http://localhost:8000/ > /dev/null; then
    echo "✓ Backend API is responding"
    curl -s http://localhost:8000/scheduler/stats | python3 -m json.tool 2>/dev/null || echo "Stats endpoint returned data"
else
    echo "✗ Backend API is NOT responding"
fi

echo ""
echo "═══════════════════════════════════════════"
echo ""
echo "To view live logs:"
echo "  Celery:   tail -f logs/celery.log"
echo "  FastAPI:  tail -f logs/fastapi.log"
echo "  Frontend: tail -f logs/frontend.log"
echo ""
