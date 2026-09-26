#!/usr/bin/env bash
# Aglibol Agent Official Uninstaller for Linux and macOS
# Usage: curl -fsSL https://raw.githubusercontent.com/xushiexpresso15/AglibolAgent/main/uninstall.sh | bash
# Pass --purge to also delete all persistent sessions, checkpoints, and cache (~/.aglibol)

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}=== Aglibol Agent Uninstaller ===${NC}\n"

INSTALL_DIR="${HOME}/.aglibol"
BIN_DIR="${HOME}/.local/bin"
PURGE=false

for arg in "$@"; do
    if [ "$arg" == "--purge" ] || [ "$arg" == "-p" ]; then
        PURGE=true
    fi
done

# 1. Uninstall via uv if present
if command -v uv >/dev/null 2>&1; then
    if uv tool list 2>/dev/null | grep -q "aglibol-agent"; then
        echo -e "${CYAN}→ Removing uv tool 'aglibol-agent'...${NC}"
        uv tool uninstall aglibol-agent || true
    fi
fi

# 2. Uninstall via pipx if present
if command -v pipx >/dev/null 2>&1; then
    if pipx list 2>/dev/null | grep -q "aglibol-agent"; then
        echo -e "${CYAN}→ Removing pipx package 'aglibol-agent'...${NC}"
        pipx uninstall aglibol-agent || true
    fi
fi

# 3. Remove symlinks in ~/.local/bin
for bin in "${BIN_DIR}/aglibol" "${BIN_DIR}/aglibol-agent"; do
    if [ -L "$bin" ] || [ -f "$bin" ]; then
        echo -e "${CYAN}→ Removing binary link ${bin}...${NC}"
        rm -f "$bin"
    fi
done

# 4. Remove standalone virtual environment
if [ -d "${INSTALL_DIR}/env" ]; then
    echo -e "${CYAN}→ Removing dedicated virtual environment in ${INSTALL_DIR}/env...${NC}"
    rm -rf "${INSTALL_DIR}/env"
fi

# 5. Handle user data (~/.aglibol)
if [ -d "${INSTALL_DIR}" ]; then
    if [ "$PURGE" = true ]; then
        echo -e "${RED}→ Purging all data, sessions, and configuration in ${INSTALL_DIR}...${NC}"
        rm -rf "${INSTALL_DIR}"
        echo -e "${GREEN}✔ All files and data completely removed.${NC}"
    else
        echo -e "${YELLOW}ℹ User sessions, checkpoints, and cache were preserved in ${INSTALL_DIR}.${NC}"
        echo -e "  To completely remove all persistent data, run:"
        echo -e "  ${CYAN}rm -rf ${INSTALL_DIR}${NC}"
    fi
fi

echo -e "\n${GREEN}✔ Aglibol Agent has been successfully uninstalled from your system.${NC}\n"
