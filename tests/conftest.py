"""Shared pytest fixtures for EmotionLens tests."""

import sys
import os

# Ensure src/ is importable from test files
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
