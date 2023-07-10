"""Simple function to call to get the current InteractiveShell instance
"""
from IPython.terminal.interactiveshell import InteractiveShell

def get_ipython() -> None | InteractiveShell: ...
