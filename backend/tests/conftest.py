"""Configuration for pytest."""

import sys
import os

# Add backend directory to path so 'app' package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
