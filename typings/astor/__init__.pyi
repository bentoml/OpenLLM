from __future__ import annotations

import ast
import typing as t

class SourceGenerator(ast.NodeVisitor):
    def __init__(
        self,
        indent_width: str,
        add_line_information: bool = ...,
        pretty_string: t.Callable[..., t.Any] = ...,
        len: t.Callable[[t.Any], int] = ...,
        isinstance: t.Callable[[t.Any, t.Any], bool] = ...,
        callable: t.Callable[[t.Any], bool] = ...,
    ) -> None: ...
    def newline(self, node: ast.AST | None = ..., extra: t.Any = ...) -> None: ...
    def write(*params: t.Any) -> None: ...
