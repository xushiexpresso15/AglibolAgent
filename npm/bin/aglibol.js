#!/usr/bin/env node

/**
 * Aglibol Agent NPX Zero-Install Launcher
 * Automatically provisions and invokes Aglibol Agent via uvx, pipx, or native Python.
 */

const { spawnSync, spawn } = require("child_process");
const fs = require("fs");
const path = require("path");
const os = require("os");

const args = process.argv.slice(2);

function commandExists(cmd) {
  try {
    const isWin = os.platform() === "win32";
    const checkCmd = isWin ? "where.exe" : "which";
    const result = spawnSync(checkCmd, [cmd], { stdio: "ignore", shell: false });
    return result.status === 0;
  } catch {
    return false;
  }
}

function runCommand(cmd, cmdArgs) {
  const isWin = os.platform() === "win32";
  let targetCmd = cmd;
  if (isWin && !targetCmd.includes(".") && !targetCmd.includes("\\")) {
    const check = spawnSync("where.exe", [targetCmd], { encoding: "utf-8", shell: false });
    if (check.stdout) {
      targetCmd = check.stdout.trim().split(/\r?\n/)[0];
    }
  }

  const child = spawn(targetCmd, cmdArgs, {
    stdio: "inherit",
    shell: false,
  });

  child.on("exit", (code) => {
    process.exit(code || 0);
  });

  child.on("error", (err) => {
    console.error(`\x1b[31m[Error] Failed to execute ${cmd}: ${err.message}\x1b[0m`);
    process.exit(1);
  });
}

function findPython() {
  const candidates = ["python3", "python", "py"];
  for (const py of candidates) {
    if (commandExists(py)) {
      try {
        const out = spawnSync(py, ["-c", "import sys; print(sys.version_info >= (3, 11))"], {
          encoding: "utf-8",
        });
        if (out.stdout && out.stdout.trim() === "True") {
          return py;
        }
      } catch {}
    }
  }
  return null;
}

function main() {
  // 1. If 'aglibol' is already installed globally, use it directly
  if (commandExists("aglibol")) {
    runCommand("aglibol", args);
    return;
  }
  if (commandExists("aglibol-agent")) {
    runCommand("aglibol-agent", args);
    return;
  }

  // 2. If 'uvx' is available (Astral uv package runner - ultra fast), use it
  if (commandExists("uvx")) {
    runCommand("uvx", ["--from", "aglibol-agent", "aglibol", ...args]);
    return;
  }

  // 3. If 'pipx' is available, use it
  if (commandExists("pipx")) {
    runCommand("pipx", ["run", "--spec", "aglibol-agent", "aglibol", ...args]);
    return;
  }

  // 4. Fallback to native python >= 3.11
  const python = findPython();
  if (python) {
    // Check if aglibol module is installed
    const hasModule = spawnSync(python, ["-m", "aglibol", "--version"], { stdio: "ignore" });
    if (hasModule.status === 0) {
      runCommand(python, ["-m", "aglibol", ...args]);
      return;
    }

    console.log("\x1b[36m[*] Aglibol Agent not found locally. Installing into user environment...\x1b[0m");
    const install = spawnSync(python, ["-m", "pip", "install", "--user", "aglibol-agent"], {
      stdio: "inherit",
    });

    if (install.status === 0) {
      runCommand(python, ["-m", "aglibol", ...args]);
      return;
    }
  }

  // 5. If no suitable Python is found, provide helpful instructions
  console.error("\x1b[31m[Error] Aglibol Agent requires Python >= 3.11 or 'uv'.\x1b[0m");
  console.error("\nPlease install Python or uv to continue:");
  console.error("  • Install uv (recommended, instant):");
  if (os.platform() === "win32") {
    console.error("      powershell -ExecutionPolicy ByPass -c \"irm https://astral.sh/uv/install.ps1 | iex\"");
  } else {
    console.error("      curl -LsSf https://astral.sh/uv/install.sh | sh");
  }
  console.error("  • Or install Python >= 3.11 from https://www.python.org/downloads/");
  process.exit(1);
}

main();
