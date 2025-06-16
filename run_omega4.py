#!/usr/bin/env python3
"""Runner script for OMEGA-4"""

import sys
import omega4_main

if __name__ == "__main__":
    # Pass command line arguments to main
    sys.argv[0] = "omega4_main.py"  # Set program name for argparse
    omega4_main.main()