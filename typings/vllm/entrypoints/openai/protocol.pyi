from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Union

from pydantic import BaseModel
from vllm.utils import random_uuid as random_uuid

class ErrorResponse(BaseModel):
    object: str
    message: str
    type: str
    param: Optional[str]
    code: Optional[str]

class ModelPermission(BaseModel):
    id: str
    object: str
    created: int
    allow_create_engine: bool
    allow_sampling: bool
    allow_logprobs: bool
    allow_search_indices: bool
    allow_view: bool
    allow_fine_tuning: bool
    organization: str
    group: Optional[str]
    is_blocking: str

class ModelCard(BaseModel):
    id: str
    object: str
    created: int
    owned_by: str
    root: Optional[str]
    parent: Optional[str]
    permission: List[ModelPermission]

class ModelList(BaseModel):
    object: str
    data: List[ModelCard]

class UsageInfo(BaseModel):
    prompt_tokens: int
    total_tokens: int
    completion_tokens: Optional[int]

class ChatCompletionRequest(BaseModel):
    model: str
    messages: Union[str, List[Dict[str, str]]]
    temperature: Optional[float]
    top_p: Optional[float]
    n: Optional[int]
    max_tokens: Optional[int]
    stop: Optional[Union[str, List[str]]]
    stream: Optional[bool]
    presence_penalty: Optional[float]
    frequency_penalty: Optional[float]
    logit_bias: Optional[Dict[str, float]]
    user: Optional[str]
    best_of: Optional[int]
    top_k: Optional[int]
    ignore_eos: Optional[bool]
    use_beam_search: Optional[bool]

class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    suffix: Optional[str]
    max_tokens: Optional[int]
    temperature: Optional[float]
    top_p: Optional[float]
    n: Optional[int]
    stream: Optional[bool]
    logprobs: Optional[int]
    echo: Optional[bool]
    stop: Optional[Union[str, List[str]]]
    presence_penalty: Optional[float]
    frequency_penalty: Optional[float]
    best_of: Optional[int]
    logit_bias: Optional[Dict[str, float]]
    user: Optional[str]
    top_k: Optional[int]
    ignore_eos: Optional[bool]
    use_beam_search: Optional[bool]

class LogProbs(BaseModel):
    text_offset: List[int]
    token_logprobs: List[Optional[float]]
    tokens: List[str]
    top_logprobs: List[Optional[Dict[str, float]]]

class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[LogProbs]
    finish_reason: Optional[Literal["stop", "length"]]

class CompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo

class CompletionResponseStreamChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[LogProbs]
    finish_reason: Optional[Literal["stop", "length"]]

class CompletionStreamResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[CompletionResponseStreamChoice]

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length"]]

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo

class DeltaMessage(BaseModel):
    role: Optional[str]
    content: Optional[str]

class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]

class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
