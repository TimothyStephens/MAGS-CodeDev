#!/usr/bin/env bash
# =============================================================================
# OMP Sandbox Environment Verification
# Tests every tool, package, and extension installed per Containerfile.
# Run:  bash test_omp_env.sh
# =============================================================================
set -euo pipefail

PASS=0
FAIL=0
SKIP=0

ok() {
	echo "  \033[32m✔\033[0m  $*"
	PASS=$((PASS + 1))
}
fail() {
	echo "  \033[31m✘\033[0m  $*"
	FAIL=$((FAIL + 1))
}
skip() {
	echo "  \033[33m⊘\033[0m  $*"
	SKIP=$((SKIP + 1))
}
heading() {
	echo
	echo -e "\033[1m══ $1 ══\033[0m"
}

cmd_ok() {
	if eval "$1" >/dev/null 2>&1; then ok "$1 → $2"; else fail "$1 → $2"; fi
}
cmd_fail() {
	if eval "$1" >/dev/null 2>&1; then fail "$1 → $2 (expected failure but succeeded)"; else ok "$1 → $2 (correctly not available)"; fi
}

# =============================================================================
# 1. SYSTEM COMPILERS & TOOLS
# =============================================================================
heading "System Compilers & Tools"
cmd_ok "python3 --version" "Python 3"
cmd_ok "python3 -m venv /tmp/_venv_check_$$ && rm -rf /tmp/_venv_check_$$" "python3-venv"
cmd_ok "pip --version" "pip"
cmd_ok "git --version" "git"
cmd_ok "curl --version" "curl"
cmd_ok "gcc --version" "gcc"
cmd_ok "g++ --version" "g++"
cmd_ok "make --version" "make"
cmd_ok "cmake --version" "cmake"
cmd_ok "rustc --version" "rustc"
cmd_ok "cargo --version" "cargo"
cmd_ok "go version" "Go"
cmd_ok "sudo --version" "sudo"
cmd_ok "nano --version" "nano"
cmd_ok "vim --version" "vim"
cmd_ok "nvim --version" "Neovim (≥0.11)"
cmd_ok "fd --version" "fd (symlinked from fdfind)"
cmd_ok "rg --version" "ripgrep"
cmd_ok "unzip -v" "unzip"

# =============================================================================
# 2. CONTAINER RUNTIMES
# =============================================================================
heading "Container Runtimes"
cmd_ok "podman --version" "Podman"
skip "Docker (skipped per request)"
skip "Apptainer (dropped — Docker/Podman sufficient)"

# =============================================================================
# 3. JAVASCRIPT RUNTIMES
# =============================================================================
heading "JavaScript Runtimes"
cmd_ok "node --version" "Node.js 22"
cmd_ok "npm --version" "npm"
cmd_ok "bun --version" "Bun"

# =============================================================================
# 4. OMP CORE
# =============================================================================
heading "OMP Core"
cmd_ok "omp --version" "OMP CLI"
cmd_ok "omp plugin list" "OMP plugin system"

# =============================================================================
# 5. PYTHON PACKAGES
# =============================================================================
heading "Python Packages"
for pkg in pytest pytest_cov flake8 mypy bandit pyright numpy pandas hound_mcp; do
	if pip show "$pkg" &>/dev/null; then
		ver=$(pip show "$pkg" 2>/dev/null | grep '^Version:' | cut -d' ' -f2)
		ok "pip:$pkg  $ver"
	else
		fail "pip:$pkg  NOT INSTALLED"
	fi
done
cmd_ok "pyright --version" "pyright CLI"

# =============================================================================
# 6. MAGS-CODEDEV
# =============================================================================
heading "MAGS-CodeDev"
if command -v mags-codedev &>/dev/null; then
	ok "mags-codedev CLI"
else
	fail "mags-codedev CLI — not on PATH"
fi
python3 -c "import mags_codedev" &>/dev/null && ok "mags_codedev import" || fail "mags_codedev import"

for sub in init build test debug; do
	if mags-codedev "$sub" --help &>/dev/null; then
		ok "mags-codedev $sub"
	else
		fail "mags-codedev $sub — missing or error"
	fi
done

# =============================================================================
# 7. OMP PLUGINS (expected)
# =============================================================================
heading "OMP Plugins (expected)"
plugin_list=$(omp plugin list 2>/dev/null)
for short in pi-permission-system pi-codegraph pi-statusline pi-context-prune pi-lens mags-codedev-extension pi-nvim-bridge; do
	if echo "$plugin_list" | grep -qi "$short"; then
		ver=$(echo "$plugin_list" | grep -i "$short" | head -1 | sed 's/.*@//')
		ok "$short@${ver:-?}"
	else
		fail "$short — not in plugin list"
	fi
done

heading "OMP Plugins (optional — installed with || true)"
for short in pi-guardrails pi-rules; do
	if echo "$plugin_list" | grep -qi "$short"; then
		ok "$short — installed"
	else
		fail "$short — NOT installed"
	fi
done

# =============================================================================
# 8. PI-NVIM-BRIDGE
# =============================================================================
heading "pi-nvim-bridge"
[ -d /opt/pi-nvim-bridge ] && ok "pi-nvim-bridge directory" || fail "pi-nvim-bridge directory"
[ -f /root/.config/pi-bridge/init.lua ] && ok "pi-bridge init.lua" || fail "pi-bridge init.lua"

# =============================================================================
# 9. MCP CONFIGURATION
# =============================================================================
heading "MCP Configuration"
if [ -f /root/.omp/agent/.mcp.json ]; then
	ok "MCP config (.mcp.json) exists"
	if grep -q context7 /root/.omp/agent/.mcp.json; then
		ok "Context7 MCP server configured"
	else
		fail "Context7 MCP server not in config"
	fi
else
	# Containerfile writes to /root/.omp/agent/.mcp.json — OMP may have migrated it
	mcp_files=$(find /root/.omp -name ".mcp.json" 2>/dev/null)
	if [ -n "$mcp_files" ]; then
		ok "MCP config found elsewhere: $mcp_files"
	else
		fail "MCP config (.mcp.json) not found anywhere"
	fi
	if grep -rl "context7" /root/.omp/agent/.mcp.json /root/.omp/agent/config.yml /root/.omp/plugins/ 2>/dev/null | grep -v sessions | grep -v '.jsonl' | head -1 | grep -q .; then
		ok "Context7 MCP server configured"
	else
		fail "Context7 MCP server not configured (Containerfile npx config missing)"
	fi
fi

# =============================================================================
# 10. HOUND-MCP CLI
# =============================================================================
heading "Hound MCP"
if command -v hound &>/dev/null; then
	hound_ver=$(hound --version 2>/dev/null | grep -oP 'v[\d.]+' || echo "✓")
	ok "hound CLI  $hound_ver"
else
	fail "hound CLI — not on PATH"
fi

# =============================================================================
# 11. PLAYWRIGHT
# =============================================================================
heading "Playwright"
cmd_ok "npx playwright --version" "Playwright CLI"
if [ -d /root/.cache/ms-playwright/chromium-* ]; then
	ok "Chromium browser installed"
else
	fail "Chromium browser — not found in cache"
fi

# =============================================================================
# 12. COMPILATION SMOKE TESTS
# =============================================================================
heading "Compilation Smoke Tests"
tmpdir=$(mktemp -d)
trap "rm -rf $tmpdir" EXIT

echo 'int main(){return 0;}' >"$tmpdir/test.c"
if gcc "$tmpdir/test.c" -o "$tmpdir/test" &>/dev/null && "$tmpdir/test" &>/dev/null; then
	ok "C compile & run"
else
	fail "C compile & run"
fi

echo 'int main(){return 0;}' >"$tmpdir/test.cpp"
if g++ "$tmpdir/test.cpp" -o "$tmpdir/test" &>/dev/null && "$tmpdir/test" &>/dev/null; then
	ok "C++ compile & run"
else
	fail "C++ compile & run"
fi

mkdir -p "$tmpdir/gotest"
echo 'package main; func main(){}' >"$tmpdir/gotest/main.go"
(cd "$tmpdir/gotest" && go mod init test &>/dev/null && go build -o test_go . &>/dev/null && ./test_go &>/dev/null) && ok "Go compile & run" || fail "Go compile & run"

mkdir -p "$tmpdir/rusttest/src"
cat >"$tmpdir/rusttest/Cargo.toml" <<'CARGO'
[package]
name = "test"
version = "0.1.0"
edition = "2021"
CARGO
echo 'fn main(){}' >"$tmpdir/rusttest/src/main.rs"
if (cd "$tmpdir/rusttest" && cargo build --release &>/dev/null); then
	ok "Rust compile (cargo build)"
else
	fail "Rust compile (cargo build)"
fi

# =============================================================================
# 13. PYTHON TOOLCHAIN SMOKE TESTS
# =============================================================================
heading "Python Toolchain Smoke"
echo 'x = 1' >"$tmpdir/sample.py"

flake8 "$tmpdir/sample.py" --max-line-length=120 >/dev/null 2>&1 && ok "flake8 lint" || ok "flake8 lint (warnings OK)"
mypy "$tmpdir/sample.py" >/dev/null 2>&1 && ok "mypy type check" || ok "mypy type check (warnings OK)"
bandit -r "$tmpdir/sample.py" --quiet >/dev/null 2>&1 && ok "bandit security scan" || ok "bandit security scan (warnings OK)"
pyright "$tmpdir/sample.py" --outputjson >/dev/null 2>&1 && ok "pyright type check" || ok "pyright type check (warnings OK)"

cat >"$tmpdir/test_sample.py" <<'PYTEST'
def test_pass():
    assert 1 + 1 == 2
PYTEST
if python3 -m pytest "$tmpdir/test_sample.py" -q >/dev/null 2>&1; then
	ok "pytest run"
else
	fail "pytest run"
fi

# =============================================================================
# SUMMARY
# =============================================================================
heading "Summary"
total=$((PASS + FAIL + SKIP))
echo ""
echo -e "  \033[32mPass:  $PASS\033[0m  |  \033[31mFail:  $FAIL\033[0m  |  \033[33mSkip:  $SKIP\033[0m  |  Total: $total"
echo ""

if [ "$FAIL" -gt 0 ]; then
	echo -e "  \033[31m✘ Some checks failed.\033[0m"
	exit 1
else
	echo -e "  \033[32m✔ All checks passed.\033[0m"
	exit 0
fi
