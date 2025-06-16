#!/bin/bash
# Run the analyzer and capture output
echo "Starting OMEGA-4 analyzer with test mode..."
echo "Press 'T' in the window to start panel testing"
echo "Output will be saved to test_output.log"

python3 omega4_main.py --width 1400 --height 900 --bars 1024 2>&1 | tee test_output.log &
PID=$!

echo "Analyzer running with PID $PID"
echo "Press Ctrl+C to stop"

# Wait for user to stop
trap "kill $PID 2>/dev/null; echo 'Stopped'; exit" INT
wait $PID