"""
Pytest configuration for signal-kit Python tests.
"""

import sys
from pathlib import Path

# Add the python directory to the path so we can import signal_kit
project_root = Path(__file__).parent.parent.parent
python_dir = project_root / "python"

if python_dir.exists() and str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))
