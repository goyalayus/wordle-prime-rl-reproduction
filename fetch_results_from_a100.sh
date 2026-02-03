#!/bin/bash
# Fetch wordle test results from A100
# Usage: bash fetch_results_from_a100.sh

set -e

echo "Connecting to A100 and fetching results..."

# SSH and copy the results file
scp s_01kgh1r0496amc9yj1r2r7snk8@ssh.lightning.ai:~/wordle-prime-rl-reproduction/wordle_test_results.json ./

echo "Results fetched to wordle_test_results.json"
echo "Open viewer.html in your browser to view the results"
