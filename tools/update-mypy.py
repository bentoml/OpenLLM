#!/usr/bin/env python3
import concurrent.futures
import configparser
import os
from typing import List


# Function to find .pyi files in a given directory
def pyi_in_subdir(directory: str, git_root: str) -> List[str]:
  pyi_files = []
  for root, _, files in os.walk(directory):
    for file in files:
      if file.endswith('.pyi') or file == '_typing_compat.py':
        full_path = os.path.join(root, file)
        # Convert to relative path with respect to the git root
        relative_path = os.path.relpath(full_path, git_root)
        pyi_files.append(relative_path)
  return pyi_files


def find_pyi_files(git_root: str) -> List[str]:
  # List all subdirectories
  subdirectories = [
    os.path.join(git_root, name) for name in os.listdir(git_root) if os.path.isdir(os.path.join(git_root, name))
  ]

  # Use a thread pool to execute searches concurrently
  with concurrent.futures.ThreadPoolExecutor() as executor:
    # Map of future to subdirectory
    future_to_subdir = {executor.submit(pyi_in_subdir, subdir, git_root): subdir for subdir in subdirectories}

    all_pyi_files = set()
    for future in concurrent.futures.as_completed(future_to_subdir):
      pyi_files = future.result()
      all_pyi_files.update(pyi_files)

  return list(all_pyi_files)


# Function to update mypy.ini file
def update_mypy_ini(pyi_files: List[str], mypy_ini_path: str) -> int:
  config = configparser.ConfigParser()
  config.read(mypy_ini_path)

  # Existing files from mypy.ini
  existing_files = config.get('mypy', 'files', fallback='').split(', ')

  # Add new .pyi files if they are not already in the list
  updated_files = existing_files + [f for f in pyi_files if f not in existing_files]

  # Update the 'files' entry
  config['mypy']['files'] = ', '.join(updated_files)

  # Write changes back to mypy.ini
  with open(mypy_ini_path, 'w') as configfile:
    config.write(configfile)
  return 0


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MYPY_CONFIG = os.path.join(ROOT, 'mypy.ini')

if __name__ == '__main__':
  raise SystemExit(update_mypy_ini(find_pyi_files(ROOT), MYPY_CONFIG))
