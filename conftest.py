import sys
from pathlib import Path

# Add the project root to sys.path so 'pipeline' and 'query' modules are importable.
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
