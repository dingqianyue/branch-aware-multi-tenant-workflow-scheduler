#!/bin/bash
# This script installs the necessary dependencies for the workflow-backend,
# ensuring numpy is installed first to prevent version conflicts.

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python 3.11 is installed
if ! command -v python3.11 &> /dev/null; then
    echo -e "${RED}Error: Python 3.11 is not installed${NC}"
    echo "Please install Python 3.11 first:"
    echo "  macOS: brew install python@3.11"
    echo "  Ubuntu: sudo apt install python3.11 python3.11-venv"
    exit 1
fi

echo -e "${GREEN}Found Python 3.11: $(python3.11 --version)${NC}"

# Backend setup
echo "Setting up backend..."
cd workflow-backend

# Remove old virtual environment if it exists
if [ -d "venv" ]; then
    echo -e "${YELLOW}Removing existing virtual environment...${NC}"
    rm -rf venv
fi

# Create a virtual environment with Python 3.11
echo "Creating virtual environment with Python 3.11..."
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify Python version
PYTHON_VERSION=$(python --version)
echo -e "${GREEN}Virtual environment Python: $PYTHON_VERSION${NC}"

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies with specific versions to avoid conflicts
echo "Installing Python dependencies with compatible versions..."

# Install numpy first (compatible with both opencv and torch)
pip install "numpy>=1.26.4,<2.0"

# Install PyTorch and torchvision first (they have strict numpy requirements)
pip install torch torchvision

# Install opencv-python-headless version compatible with numpy<2.0
# Version 4.10.0.84 is the last version that works with numpy<2.0
pip install opencv-python-headless==4.10.0.84

# Install other dependencies
pip install fastapi uvicorn celery redis openslide-python \
    networkx scikit-image pillow \
    websockets python-multipart aiofiles requests

# Install InstanSeg
echo "Installing InstanSeg..."
pip install git+https://github.com/instanseg/instanseg.git

echo -e "${GREEN}Backend setup complete.${NC}"

# Deactivate virtual environment
deactivate

# Frontend setup
echo "Setting up frontend..."
cd ../workflow-frontend

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo -e "${RED}Error: npm is not installed${NC}"
    echo "Please install Node.js and npm first"
    exit 1
fi

npm install

echo -e "${GREEN}Frontend setup complete.${NC}"
echo -e "${GREEN}✓ Setup finished successfully!${NC}"
