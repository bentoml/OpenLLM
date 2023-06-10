from typing import Any

from .core import StrategyList
from ..merger import Merger

class SetStrategies(StrategyList):
    NAME = ...

    @staticmethod
    def strategy_union(config: Any, path: str, base: StrategyList, nxt: StrategyList) -> StrategyList: ...
    @staticmethod
    def strategy_intersect(config: Any, path: str, base: StrategyList, nxt: StrategyList) -> StrategyList: ...
    @staticmethod
    def strategy_override(config: Merger, path: str, base: StrategyList, nxt: StrategyList) -> StrategyList: ...
