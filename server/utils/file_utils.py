"""
Core data handling utilities for file operations, data validation, and type checking.
"""
import os
import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union, Optional

def ensure_dir(path: Union[str, Path]) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path as string or Path object
    """
    if isinstance(path, str):
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

def validate_directory_structure(config: dict) -> bool:
    """
    Validates that all required directories exist and are accessible.
    Creates missing directories if needed.
    """
    def create_if_missing(path: str):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            return f"Created: {path}"
        return f"Exists: {path}"

    results = []
    
    # Validate all directory paths from config
    for category, paths in config.items():
        if isinstance(paths, dict):
            for name, path in paths.items():
                if isinstance(path, str) and "dir" in name.lower():
                    results.append(create_if_missing(path))

    return "\n".join(results)

def archive_old_data(source_path: str, archive_path: str, days_threshold: int = 30) -> None:
    """
    Archives data older than the specified threshold.
    """
    current_time = datetime.now()
    
    for root, _, files in os.walk(source_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            
            if (current_time - file_time).days > days_threshold:
                relative_path = os.path.relpath(root, source_path)
                archive_dir = os.path.join(archive_path, relative_path)
                os.makedirs(archive_dir, exist_ok=True)
                
                shutil.move(file_path, os.path.join(archive_dir, file))

def create_backup(source_path: str, backup_dir: str, backup_name: Optional[str] = None) -> str:
    """
    Creates a backup of the specified source path.
    """
    if backup_name is None:
        backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    backup_path = os.path.join(backup_dir, backup_name)
    
    if os.path.isfile(source_path):
        os.makedirs(backup_dir, exist_ok=True)
        shutil.copy2(source_path, backup_path)
    else:
        shutil.copytree(source_path, backup_path)
    
    return backup_path

def load_json_safe(file_path: str) -> Dict:
    """
    Safely loads a JSON file with error handling.
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {str(e)}")
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {file_path}")

def save_json_safe(data: Union[Dict, List], file_path: str, indent: int = 2) -> None:
    """
    Safely saves data to a JSON file with error handling.
    """
    try:
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
            
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent)
    except Exception as e:
        raise IOError(f"Failed to save JSON to {file_path}: {str(e)}")

def clean_old_files(directory: str, pattern: str = "*", days_threshold: int = 30) -> List[str]:
    """
    Removes files older than the specified threshold.
    Returns list of removed files.
    """
    removed_files = []
    current_time = datetime.now()
    
    for root, _, files in os.walk(directory):
        for file in files:
            if Path(file).match(pattern):
                file_path = os.path.join(root, file)
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if (current_time - file_time).days > days_threshold:
                    os.remove(file_path)
                    removed_files.append(file_path)
    
    return removed_files