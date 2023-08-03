import abc
from abc import abstractmethod

from _typeshed import Incomplete
from torch import nn

from .triton_utils.mixin import TritonModuleMixin as TritonModuleMixin

logger: Incomplete

class FusedBaseModule(nn.Module, TritonModuleMixin, metaclass=abc.ABCMeta):
    @classmethod
    @abstractmethod
    def inject_to_model(cls, *args, **kwargs): ...

class FusedBaseAttentionModule(FusedBaseModule, metaclass=abc.ABCMeta):
    @classmethod
    @abstractmethod
    def inject_to_model(cls, model, use_triton: bool = ..., group_size: int = ..., use_cuda_fp16: bool = ..., desc_act: bool = ..., trainable: bool = ..., **kwargs): ...
    @classmethod
    def warmup(cls, model, transpose: bool = ..., seqlen: int = ...) -> None: ...

class FusedBaseMLPModule(FusedBaseModule, metaclass=abc.ABCMeta):
    @classmethod
    @abstractmethod
    def inject_to_model(cls, model, use_triton: bool = ..., **kwargs): ...
