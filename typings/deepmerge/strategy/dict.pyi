from .core import StrategyList
from ..merger import Merger
class DictStrategies(StrategyList):
    @staticmethod
    def strategy_merge(config: Merger, path: str, base: StrategyList, nxt: StrategyList) -> StrategyList: ...
    @staticmethod
    def strategy_override(config: Merger, path: str, base: StrategyList, nxt: StrategyList) -> StrategyList: ...
