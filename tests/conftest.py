# tests/conftest.py
import sys
import os

# Add the parent directory to the Python path
# This allows pytest to find the 'src' directory
# and its modules.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

