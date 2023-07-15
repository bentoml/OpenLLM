"""A dict subclass that supports attribute style access.

Can probably be replaced by types.SimpleNamespace from Python 3.3
"""
from typing import Any, Self

class Struct(dict[str, Any]):
    """A dict subclass with attribute style access.

    This dict subclass has a a few extra features:

    * Attribute style access.
    * Protection of class members (like keys, items) when using attribute
      style access.
    * The ability to restrict assignment to only existing keys.
    * Intelligent merging.
    * Overloaded operators.
    """

    _allownew: bool = True

    def __init__(self, *args: Any, **kw: Any):
        """Initialize with a dictionary, another Struct, or data.

        Parameters
        ----------
        *args : dict, Struct
            Initialize with one dict or Struct
        **kw : dict
            Initialize with key, value pairs.

        Examples
        --------
        >>> s = Struct(a=10,b=30)
        >>> s.a
        10
        >>> s.b
        30
        >>> s2 = Struct(s,c=30)
        >>> sorted(s2.keys())
        ['a', 'b', 'c']
        """
    def __setitem__(self, key: str, value: Any) -> None:
        """Set an item with check for allownew.

        Examples
        --------
        >>> s = Struct()
        >>> s['a'] = 10
        >>> s.allow_new_attr(False)
        >>> s['a'] = 10
        >>> s['a']
        10
        >>> try:
        ...     s['b'] = 20
        ... except KeyError:
        ...     print('this is not allowed')
        ...
        this is not allowed
        """
    def __setattr__(self, key: str, value: Any) -> None:
        """Set an attr with protection of class members.

        This calls :meth:`self.__setitem__` but convert :exc:`KeyError` to
        :exc:`AttributeError`.

        Examples
        --------
        >>> s = Struct()
        >>> s.a = 10
        >>> s.a
        10
        >>> try:
        ...     s.get = 10
        ... except AttributeError:
        ...     print("you can't set a class member")
        ...
        you can't set a class member
        """
    def __getattr__(self, key: str) -> Any:
        """Get an attr by calling :meth:`dict.__getitem__`.

        Like :meth:`__setattr__`, this method converts :exc:`KeyError` to
        :exc:`AttributeError`.

        Examples
        --------
        >>> s = Struct(a=10)
        >>> s.a
        10
        >>> type(s.get)
        <... 'builtin_function_or_method'>
        >>> try:
        ...     s.b
        ... except AttributeError:
        ...     print("I don't have that key")
        ...
        I don't have that key
        """
    def __iadd__(self, other: Struct) -> Self:
        """s += s2 is a shorthand for s.merge(s2).

        Examples
        --------
        >>> s = Struct(a=10,b=30)
        >>> s2 = Struct(a=20,c=40)
        >>> s += s2
        >>> sorted(s.keys())
        ['a', 'b', 'c']
        """
    def __add__(self, other: Struct) -> Struct:
        """s + s2 -> New Struct made from s.merge(s2).

        Examples
        --------
        >>> s1 = Struct(a=10,b=30)
        >>> s2 = Struct(a=20,c=40)
        >>> s = s1 + s2
        >>> sorted(s.keys())
        ['a', 'b', 'c']
        """
    def __sub__(self, other: Struct) -> Struct:
        """s1 - s2 -> remove keys in s2 from s1.

        Examples
        --------
        >>> s1 = Struct(a=10,b=30)
        >>> s2 = Struct(a=40)
        >>> s = s1 - s2
        >>> s
        {'b': 30}
        """
    def __isub__(self, other: Struct) -> Self:
        """Inplace remove keys from self that are in other.

        Examples
        --------
        >>> s1 = Struct(a=10,b=30)
        >>> s2 = Struct(a=40)
        >>> s1 -= s2
        >>> s1
        {'b': 30}
        """
    def dict(self) -> Self:
        """Get the dict representation of the struct."""
    def copy(self) -> Self:
        """Return a copy as a Struct.

        Examples
        --------
        >>> s = Struct(a=10,b=30)
        >>> s2 = s.copy()
        >>> type(s2) is Struct
        True
        """
    def hasattr(self, key: str) -> bool:
        """hasattr function available as a method.

        Implemented like has_key.

        Examples
        --------
        >>> s = Struct(a=10)
        >>> s.hasattr('a')
        True
        >>> s.hasattr('b')
        False
        >>> s.hasattr('get')
        False
        """
    def allow_new_attr(self, allow: bool = True) -> None:
        """Set whether new attributes can be created in this Struct.

        This can be used to catch typos by verifying that the attribute user
        tries to change already exists in this Struct.
        """
    def merge(
        self,
        __loc_data__: dict[str, Any] | Struct | None = ...,
        __conflict_solve: dict[str, Any] | None = ...,
        **kw: Any,
    ) -> Struct:
        """Merge two Structs with customizable conflict resolution.

        This is similar to :meth:`update`, but much more flexible. First, a
        dict is made from data+key=value pairs. When merging this dict with
        the Struct S, the optional dictionary 'conflict' is used to decide
        what to do.

        If conflict is not given, the default behavior is to preserve any keys
        with their current value (the opposite of the :meth:`update` method's
        behavior).

        Parameters
        ----------
        __loc_data__ : dict, Struct
            The data to merge into self
        __conflict_solve : dict
            The conflict policy dict.  The keys are binary functions used to
            resolve the conflict and the values are lists of strings naming
            the keys the conflict resolution function applies to.  Instead of
            a list of strings a space separated string can be used, like
            'a b c'.
        **kw : dict
            Additional key, value pairs to merge in

        Notes
        -----
        The `__conflict_solve` dict is a dictionary of binary functions which will be used to
        solve key conflicts.  Here is an example::

            __conflict_solve = dict(
                func1=['a','b','c'],
                func2=['d','e']
            )

        In this case, the function :func:`func1` will be used to resolve
        keys 'a', 'b' and 'c' and the function :func:`func2` will be used for
        keys 'd' and 'e'.  This could also be written as::

            __conflict_solve = dict(func1='a b c',func2='d e')

        These functions will be called for each key they apply to with the
        form::

            func1(self['a'], other['a'])

        The return value is used as the final merged value.

        As a convenience, merge() provides five (the most commonly needed)
        pre-defined policies: preserve, update, add, add_flip and add_s. The
        easiest explanation is their implementation::

            preserve = lambda old,new: old
            update   = lambda old,new: new
            add      = lambda old,new: old + new
            add_flip = lambda old,new: new + old  # note change of order!
            add_s    = lambda old,new: old + ' ' + new  # only for str!

        You can use those four words (as strings) as keys instead
        of defining them as functions, and the merge method will substitute
        the appropriate functions for you.

        For more complicated conflict resolution policies, you still need to
        construct your own functions.

        Examples
        --------
        This show the default policy:

        >>> s = Struct(a=10,b=30)
        >>> s2 = Struct(a=20,c=40)
        >>> s.merge(s2)
        >>> sorted(s.items())
        [('a', 10), ('b', 30), ('c', 40)]

        Now, show how to specify a conflict dict:

        >>> s = Struct(a=10,b=30)
        >>> s2 = Struct(a=20,b=40)
        >>> conflict = {'update':'a','add':'b'}
        >>> s.merge(s2,conflict)
        >>> sorted(s.items())
        [('a', 20), ('b', 70)]
        """
