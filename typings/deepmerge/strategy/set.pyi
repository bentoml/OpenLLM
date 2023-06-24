from typing import Any

from ..merger import Merger
from .core import StrategyList


class SetStrategies(StrategyList):
    NAME = ...

    @staticmethod
    def strategy_union(config: Any, path: str, base: StrategyList, nxt: StrategyList) -> StrategyList: ...
    @staticmethod
    def strategy_intersect(config: Any, path: str, base: StrategyList, nxt: StrategyList) -> StrategyList: ...
    @staticmethod
    def strategy_override(config: Merger, path: str, base: StrategyList, nxt: StrategyList) -> StrategyList: ...
