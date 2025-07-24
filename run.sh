#!/bin/bash

# Deep RL Trading Application Launcher
# Usage: ./run.sh [api|dashboard|both]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
MODE="both"
PORT_API=5001
PORT_DASHBOARD=5000

# Parse command line arguments
if [ "$#" -gt 0 ]; then
    MODE="$1"
fi

# Function to check if Python is available
check_python() {
    if ! command -v python &> /dev/null; then
        echo -e "${RED}Error: Python is not installed or not in PATH${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Python available: $(python --version)${NC}"
}

# Function to check dependencies
check_dependencies() {
    echo -e "${YELLOW}Checking dependencies...${NC}"
    
    if [ ! -f "requirements.txt" ]; then
        echo -e "${RED}Error: requirements.txt not found${NC}"
        exit 1
    fi
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo -e "${YELLOW}Creating virtual environment...${NC}"
        python -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate || source venv/Scripts/activate
    
    # Install/upgrade dependencies
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -r requirements.txt --quiet
    
    echo -e "${GREEN}✓ Dependencies installed${NC}"
}

# Function to start API server
start_api() {
    echo -e "${GREEN}Starting Trading Prediction API on port ${PORT_API}...${NC}"
    cd API-App
    export FLASK_DEBUG=false
    export PORT=$PORT_API
    python app.py &
    API_PID=$!
    echo "API Server PID: $API_PID"
    cd ..
}

# Function to start dashboard
start_dashboard() {
    echo -e "${GREEN}Starting Data Visualization Dashboard on port ${PORT_DASHBOARD}...${NC}"
    cd App
    export FLASK_DEBUG=false
    export PORT=$PORT_DASHBOARD
    python app.py &
    DASHBOARD_PID=$!
    echo "Dashboard PID: $DASHBOARD_PID"
    cd ..
}

# Function to cleanup processes
cleanup() {
    echo -e "\n${YELLOW}Shutting down servers...${NC}"
    if [ ! -z "$API_PID" ]; then
        kill $API_PID 2>/dev/null || true
        echo "API server stopped"
    fi
    if [ ! -z "$DASHBOARD_PID" ]; then
        kill $DASHBOARD_PID 2>/dev/null || true
        echo "Dashboard stopped"
    fi
    echo -e "${GREEN}Cleanup complete${NC}"
    exit 0
}

# Trap signals for cleanup
trap cleanup SIGINT SIGTERM

# Main execution
echo -e "${GREEN}🚀 Deep RL Trading Application Launcher${NC}"
echo -e "Mode: ${YELLOW}$MODE${NC}"
echo ""

check_python
check_dependencies

# Activate virtual environment
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null || true

case $MODE in
    "api")
        start_api
        echo -e "\n${GREEN}✓ API server running at http://localhost:${PORT_API}${NC}"
        echo -e "  Health check: ${YELLOW}curl http://localhost:${PORT_API}/health${NC}"
        ;;
    "dashboard")
        start_dashboard
        echo -e "\n${GREEN}✓ Dashboard running at http://localhost:${PORT_DASHBOARD}${NC}"
        ;;
    "both")
        start_api
        sleep 2
        start_dashboard
        echo -e "\n${GREEN}✓ Both servers running:${NC}"
        echo -e "  API: ${YELLOW}http://localhost:${PORT_API}${NC}"
        echo -e "  Dashboard: ${YELLOW}http://localhost:${PORT_DASHBOARD}${NC}"
        ;;
    *)
        echo -e "${RED}Invalid mode: $MODE${NC}"
        echo "Usage: $0 [api|dashboard|both]"
        exit 1
        ;;
esac

echo -e "\n${YELLOW}Press Ctrl+C to stop all servers${NC}"

# Wait for all background jobs
wait