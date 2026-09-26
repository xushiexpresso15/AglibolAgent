#!/usr/bin/env bash
# Aglibol Agent Official One-Line Installer for Linux and macOS
# Usage: curl -fsSL https://raw.githubusercontent.com/Aglibol/AglibolAgent/main/install.sh | bash

set -e

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}"
cat << "EOF"
    _          _ _ _           _       _                    _   
   / \   __ _ | (_) |__   ___ | |     / \   __ _  ___ _ __ | |_ 
  / _ \ / _` || | | '_ \ / _ \| |    / _ \ / _` |/ _ \ '_ \| __|
 / ___ \ (_| || | | |_) | (_) | |   / ___ \ (_| |  __/ | | | |_ 
/_/   \_\__, ||_|_|_.__/ \___/|_|  /_/   \_\__, |\___|_| |_|\__|
        |___/                             |___/                
EOF
echo -e "  Aglibol Agent — Personal Multi-Agent AI Framework (Ollama-Native)${NC}\n"

INSTALL_DIR="${HOME}/.aglibol"
BIN_DIR="${HOME}/.local/bin"
mkdir -p "${INSTALL_DIR}" "${BIN_DIR}"

# 1. Check for uv (fastest Python package manager)
if command -v uv >/dev/null 2>&1; then
    echo -e "${GREEN}✔ Found uv! Installing Aglibol Agent CLI via uv tool...${NC}"
    uv tool install --force aglibol-agent
elif command -v uvx >/dev/null 2>&1; then
    echo -e "${GREEN}✔ Found uvx! Aglibol Agent can be run instantly via: uvx --from aglibol-agent aglibol${NC}"
else
    # Check for Python >= 3.11
    PYTHON=""
    for py in python3 python; do
        if command -v "$py" >/dev/null 2>&1; then
            if "$py" -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
                PYTHON="$py"
                break
            fi
        fi
    done

    if [ -z "$PYTHON" ]; then
        echo -e "${YELLOW}ℹ Python 3.11+ not found. Installing standalone 'uv' environment...${NC}"
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="${HOME}/.cargo/bin:${HOME}/.local/bin:${PATH}"
        uv tool install aglibol-agent
    else
        echo -e "${GREEN}✔ Found Python ($($PYTHON --version))${NC}"
        echo -e "${CYAN}→ Creating dedicated virtual environment in ${INSTALL_DIR}/env...${NC}"
        "$PYTHON" -m venv "${INSTALL_DIR}/env"
        "${INSTALL_DIR}/env/bin/pip" install --upgrade pip --quiet
        "${INSTALL_DIR}/env/bin/pip" install aglibol-agent --quiet
        ln -sf "${INSTALL_DIR}/env/bin/aglibol" "${BIN_DIR}/aglibol"
        ln -sf "${INSTALL_DIR}/env/bin/aglibol-agent" "${BIN_DIR}/aglibol-agent"
    fi
fi

# Ensure ~/.local/bin is in PATH
if [[ ":$PATH:" != *":${BIN_DIR}:"* ]]; then
    echo -e "\n${YELLOW}⚠ Notice: ${BIN_DIR} is not currently in your PATH.${NC}"
    echo "Add it to your shell configuration:"
    echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo ""
fi

echo -e "\n${GREEN}======================================================${NC}"
echo -e "${GREEN}✔ Aglibol Agent installed successfully!${NC}"
echo -e "Run the interactive developer environment with:"
echo -e "  ${CYAN}aglibol${NC}"
echo -e "${GREEN}======================================================${NC}\n"
