from ..merger import Merger
from .core import StrategyList


class DictStrategies(StrategyList):
    @staticmethod
    def strategy_merge(config: Merger, path: str, base: StrategyList, nxt: StrategyList) -> StrategyList: ...
    @staticmethod
    def strategy_override(config: Merger, path: str, base: StrategyList, nxt: StrategyList) -> StrategyList: ...
