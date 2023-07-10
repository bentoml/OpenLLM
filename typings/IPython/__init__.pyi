import sys

from .core.getipython import get_ipython as getipython

"""
IPython: tools for interactive and parallel computing in Python.

https://ipython.org
"""
if sys.version_info < (3, 9): ...
__all__ = ["start_ipython", "embed", "start_kernel", "embed_kernel"]
__author__ = ...
__license__ = ...
__version__ = ...
version_info = ...
__patched_cves__ = ...

def embed_kernel(module=..., local_ns=..., **kwargs):  # -> None:
    """Embed and start an IPython kernel in a given scope.

    If you don't want the kernel to initialize the namespace
    from the scope of the surrounding function,
    and/or you want to load full IPython configuration,
    you probably want `IPython.start_kernel()` instead.

    Parameters
    ----------
    module : types.ModuleType, optional
        The module to load into IPython globals (default: caller)
    local_ns : dict, optional
        The namespace to load into IPython user namespace (default: caller)
    **kwargs : various, optional
        Further keyword args are relayed to the IPKernelApp constructor,
        such as `config`, a traitlets :class:`Config` object (see :ref:`configure_start_ipython`),
        allowing configuration of the kernel (see :ref:`kernel_options`).  Will only have an effect
        on the first embed_kernel call for a given process.
    """
    ...

def start_ipython(argv=..., **kwargs):  # -> None:
    """Launch a normal IPython instance (as opposed to embedded)

    `IPython.embed()` puts a shell in a particular calling scope,
    such as a function or method for debugging purposes,
    which is often not desirable.

    `start_ipython()` does full, regular IPython initialization,
    including loading startup files, configuration, etc.
    much of which is skipped by `embed()`.

    This is a public API method, and will survive implementation changes.

    Parameters
    ----------
    argv : list or None, optional
        If unspecified or None, IPython will parse command-line options from sys.argv.
        To prevent any command-line parsing, pass an empty list: `argv=[]`.
    user_ns : dict, optional
        specify this dictionary to initialize the IPython user namespace with particular values.
    **kwargs : various, optional
        Any other kwargs will be passed to the Application constructor,
        such as `config`, a traitlets :class:`Config` object (see :ref:`configure_start_ipython`),
        allowing configuration of the instance (see :ref:`terminal_options`).
    """
    ...

def start_kernel(argv=..., **kwargs):  # -> None:
    """Launch a normal IPython kernel instance (as opposed to embedded)

    `IPython.embed_kernel()` puts a shell in a particular calling scope,
    such as a function or method for debugging purposes,
    which is often not desirable.

    `start_kernel()` does full, regular IPython initialization,
    including loading startup files, configuration, etc.
    much of which is skipped by `embed_kernel()`.

    Parameters
    ----------
    argv : list or None, optional
        If unspecified or None, IPython will parse command-line options from sys.argv.
        To prevent any command-line parsing, pass an empty list: `argv=[]`.
    user_ns : dict, optional
        specify this dictionary to initialize the IPython user namespace with particular values.
    **kwargs : various, optional
        Any other kwargs will be passed to the Application constructor,
        such as `config`, a traitlets :class:`Config` object (see :ref:`configure_start_ipython`),
        allowing configuration of the kernel (see :ref:`kernel_options`).
    """
    ...
