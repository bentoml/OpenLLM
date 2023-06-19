from typing import List
from typing import Literal
from typing import Tuple
from typing import Union

from .merger import Merger


DEFAULT_TYPE_SPECIFIC_MERGE_STRATEGIES: List[
    Tuple[type, Union[Literal["append"], Literal["merge"], Literal["union"]]]
] = ...
always_merger: Merger = ...
merge_or_raise: Merger = ...
conservative_merger: Merger = ...
