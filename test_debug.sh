#!/bin/bash
echo "Starting OMEGA-4 with debug output enabled..."
echo "Debug output will update every second in the terminal"
echo "Press 'D' to toggle debug on/off"
echo ""

# Run for 10 seconds and capture output
timeout 10 python3 omega4_main.py --width 1400 --height 900 --bars 1024 2>&1 | tee debug_output.log

echo ""
echo "Test completed. Output saved to debug_output.log"
echo "Showing last debug frame:"
echo ""

# Extract the last complete debug frame
awk '/^=+$/{buf=""}; {buf=buf"\n"$0}; END{print buf}' debug_output.log | tail -60