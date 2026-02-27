#!/bin/bash
# =============================================================
# GA-MSSR Mac Mini Setup Script
#
# Run this ONCE after copying the project to the Mac Mini.
# It will:
#   1. Detect the current user and project path
#   2. Create a Python virtual environment
#   3. Install all dependencies
#   4. Fix paths in the launchd plist
#   5. Create logs/ and live/state/ directories
#   6. Register and start the launchd service
#   7. Disable Mac Mini sleep
#
# Usage:
#   cd /path/to/ga-mssr
#   bash scripts/setup_mac_mini.sh
# =============================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================================"
echo "  GA-MSSR Mac Mini Setup"
echo "============================================================"
echo ""

# --- Detect environment ---
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MAC_USER="$(whoami)"
HOME_DIR="$HOME"
PLIST_NAME="com.ga-mssr.live"
PLIST_SRC="$PROJECT_DIR/com.ga-mssr.live.plist"
PLIST_DST="$HOME_DIR/Library/LaunchAgents/$PLIST_NAME.plist"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"

echo -e "  User:        ${GREEN}$MAC_USER${NC}"
echo -e "  Project dir: ${GREEN}$PROJECT_DIR${NC}"
echo -e "  Home dir:    ${GREEN}$HOME_DIR${NC}"
echo ""

# --- Check for .env file ---
if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo -e "${RED}ERROR: .env file not found!${NC}"
    echo "  Copy .env.example to .env and fill in your Bybit API keys:"
    echo "  cp $PROJECT_DIR/.env.example $PROJECT_DIR/.env"
    exit 1
fi
echo -e "${GREEN}[OK]${NC} .env file found"

# --- Check Python 3 is available ---
if command -v python3 &> /dev/null; then
    SYSTEM_PYTHON="$(command -v python3)"
    PY_VERSION="$(python3 --version 2>&1)"
    echo -e "${GREEN}[OK]${NC} $PY_VERSION at $SYSTEM_PYTHON"
else
    echo -e "${RED}ERROR: Python 3 not found!${NC}"
    echo "  Install Python 3.10+ via Homebrew: brew install python@3.12"
    exit 1
fi

# --- Create virtual environment ---
echo ""
echo "--- Setting up Python virtual environment ---"
if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}[SKIP]${NC} .venv already exists"
else
    echo "Creating .venv..."
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}[OK]${NC} Virtual environment created"
fi

# --- Install dependencies ---
echo ""
echo "--- Installing dependencies ---"
"$VENV_DIR/bin/pip" install --upgrade pip -q
"$VENV_DIR/bin/pip" install -e "$PROJECT_DIR" -q
"$VENV_DIR/bin/pip" install "ccxt>=4.0" "APScheduler>=3.10,<4.0" "python-dotenv>=1.0" "pygad>=3.0" -q
echo -e "${GREEN}[OK]${NC} All dependencies installed"

# --- Create required directories ---
echo ""
echo "--- Creating directories ---"
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/live/state"
echo -e "${GREEN}[OK]${NC} logs/ and live/state/ directories ready"

# --- Fix paths in the launchd plist ---
echo ""
echo "--- Configuring launchd plist ---"

# Generate the plist with correct paths for THIS machine
cat > "$PLIST_SRC" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$PLIST_NAME</string>

    <key>ProgramArguments</key>
    <array>
        <string>$PYTHON_BIN</string>
        <string>$PROJECT_DIR/scripts/run_live.py</string>
    </array>

    <key>WorkingDirectory</key>
    <string>$PROJECT_DIR</string>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>$PROJECT_DIR/logs/launchd_stdout.log</string>

    <key>StandardErrorPath</key>
    <string>$PROJECT_DIR/logs/launchd_stderr.log</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin</string>
    </dict>

    <key>ThrottleInterval</key>
    <integer>30</integer>
</dict>
</plist>
EOF
echo -e "${GREEN}[OK]${NC} Plist generated with correct paths"

# --- Copy plist to LaunchAgents ---
mkdir -p "$HOME_DIR/Library/LaunchAgents"
cp "$PLIST_SRC" "$PLIST_DST"
echo -e "${GREEN}[OK]${NC} Plist copied to ~/Library/LaunchAgents/"

# --- Quick test: verify imports work ---
echo ""
echo "--- Testing Python environment ---"
"$PYTHON_BIN" -c "
from live.config import load_config
config = load_config()
print(f'  Symbol:     {config.symbol}')
print(f'  Trade size: {config.trade_size}')
print(f'  Leverage:   {config.leverage}x')
print(f'  Testnet:    {config.testnet}')
print(f'  API key:    {config.api_key[:6]}...')
"
echo -e "${GREEN}[OK]${NC} Config loads correctly"

# --- Mainnet safety check ---
echo ""
TESTNET_VAL=$(grep BYBIT_TESTNET "$PROJECT_DIR/.env" | cut -d= -f2)
if [ "$TESTNET_VAL" = "false" ]; then
    echo -e "${YELLOW}WARNING: Configured for MAINNET (real money)${NC}"
else
    echo -e "${GREEN}Configured for TESTNET${NC}"
fi

# --- Register and start the launchd service ---
echo ""
echo "--- Registering launchd service ---"

# Unload if already loaded (ignore errors)
launchctl unload "$PLIST_DST" 2>/dev/null || true

echo ""
echo -e "${YELLOW}The bot needs the mainnet confirmation prompt bypassed for launchd.${NC}"
echo "To start the service, run:"
echo ""
echo -e "  ${GREEN}launchctl load $PLIST_DST${NC}"
echo ""
echo "To check status:"
echo -e "  ${GREEN}launchctl list | grep ga-mssr${NC}"
echo ""
echo "To view logs:"
echo -e "  ${GREEN}tail -f $PROJECT_DIR/logs/ga_mssr_live.log${NC}"
echo ""
echo "To stop the service:"
echo -e "  ${GREEN}launchctl unload $PLIST_DST${NC}"

# --- Disable sleep ---
echo ""
echo "--- Sleep prevention ---"
echo -e "${YELLOW}IMPORTANT:${NC} Prevent the Mac Mini from sleeping:"
echo "  System Settings > Energy > Prevent automatic sleeping when display is off: ON"
echo "  Or run: sudo pmset -a disablesleep 1"

echo ""
echo "============================================================"
echo -e "  ${GREEN}Setup complete!${NC}"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Verify .env has correct API keys"
echo "  2. Run: launchctl load $PLIST_DST"
echo "  3. Check: tail -f $PROJECT_DIR/logs/ga_mssr_live.log"
echo ""
