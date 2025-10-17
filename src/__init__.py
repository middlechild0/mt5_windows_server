"""
Initialize the src package
"""
from pathlib import Path

# Create empty __init__.py files in all subdirectories
def create_init_files():
    """Create __init__.py files in all subdirectories."""
    root = Path(__file__).parent
    for dir_path in [root] + list(root.glob("**/")):
        init_file = dir_path / "__init__.py"
        if not init_file.exists():
            init_file.touch()

create_init_files()