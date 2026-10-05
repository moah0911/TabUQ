#!/bin/bash
# TabM-UQ Experiment Status Checker
# Usage: bash check_status.sh

echo "=========================================="
echo "TabM-UQ Experiment Status"
echo "=========================================="
echo ""

# Check running processes
echo "Running Processes:"
ps aux | grep "experiments.train\|run_all.py" | grep -v grep | awk '{print "  PID: " $2 " | CPU: " $3 "% | MEM: " $4 "% | Time: " $11 " | " $12}'

# Count completed results
echo ""
echo "Completed Results:"
if [ -d "results/raw" ]; then
    count=$(ls results/raw/*_uq.json 2>/dev/null | wc -l)
    echo "  $count / 33 total (11 datasets × 3 seeds)"
    ls results/raw/*_uq.json 2>/dev/null | sed 's|results/raw/|  - |' | sed 's|_uq.json||'
else
    echo "  0 / 33"
fi

# Check log
echo ""
echo "Latest Log Entries:"
if [ -f "results/k32_no_embeddings.log" ]; then
    tail -10 results/k32_no_embeddings.log
elif [ -f "results/full_run.log" ]; then
    tail -10 results/full_run.log
else
    echo "  No log file found"
fi

echo ""
echo "=========================================="
