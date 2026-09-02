#!/bin/bash
# Stress test runner for MAGs-CodeDev
# Usage: ./tests/stress_test.sh
#
# First run: bootstraps tests/stress_test/ (git init, copies manifest).
# Subsequent runs: picks up where it left off.
# Cleanup: rm -rf tests/stress_test

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEST_DIR="${SCRIPT_DIR}/stress_test"
MANIFEST_SRC="${SCRIPT_DIR}/stress_test_manifest.json"

echo "============================================"
echo "MAGs-CodeDev Stress Test"
echo "Work dir: ${TEST_DIR}"
echo "============================================"
echo ""

# Bootstrap on first run
if [ ! -d "${TEST_DIR}" ]; then
	echo "Bootstrapping ${TEST_DIR}..."
	mkdir -p "${TEST_DIR}/.mags-codedev"
	cp "${MANIFEST_SRC}" "${TEST_DIR}/.mags-codedev/manifest.json"
	cd "${TEST_DIR}"
	git init
	echo "Bootstrap complete."
	echo ""
fi

cd "${TEST_DIR}"
mags-codedev build 2>&1

echo ""
echo "============================================"
echo "Stress test run complete."
echo "To clean: rm -rf ${TEST_DIR}"
echo "============================================"
