"""OpenLLM CLI Extension.

The following directory contains all possible extensions for OpenLLM CLI
For adding new extension, just simply name that ext to `<name_ext>.py` and define
a ``click.command()`` with the following format:

```python
import click

@click.command(<name_ext>)
...
def cli(...): # <- this is important here, it should always name CLI in order for the extension resolver to know how to import this extensions.
```

NOTE: Make sure to keep this file blank such that it won't mess with the import order.
"""
