#!/usr/bin/env python3
"""Start the mixinto web interface."""
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from mixinto.web.app import main

if __name__ == "__main__":
    main()
