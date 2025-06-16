#!/bin/bash
# Quick verification of fixes

echo "Starting OMEGA-4 for 15 seconds..."
echo "Please play music with vocals"
echo "Press 'D' in the window to generate debug snapshot"
echo "----------------------------------------"

timeout 15 python3 omega4_main.py 2>&1 | grep -E "(BANDS|Has Voice|Brilliance|dB\s*$|Band [0-9]:)" | head -50