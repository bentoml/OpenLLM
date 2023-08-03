from typing import Dict
from typing import List
from typing import Tuple

import torch
from _typeshed import Incomplete
from vllm.sampling_params import SamplingParams as SamplingParams
from vllm.sequence import SequenceData as SequenceData
from xformers.ops import AttentionBias as AttentionBias

class InputMetadata:
    seq_groups: Incomplete
    seq_data: Incomplete
    prompt_lens: Incomplete
    slot_mapping: Incomplete
    context_lens: Incomplete
    max_context_len: Incomplete
    block_tables: Incomplete
    num_prompts: Incomplete
    num_prompt_tokens: Incomplete
    num_generation_tokens: Incomplete
    num_valid_tokens: Incomplete
    max_num_blocks_per_seq: Incomplete
    attn_bias: Incomplete
    def __init__(self, seq_groups: List[Tuple[List[int], SamplingParams]], seq_data: Dict[int, SequenceData], prompt_lens: List[int], slot_mapping: torch.Tensor, context_lens: torch.Tensor, max_context_len: int, block_tables: torch.Tensor) -> None: ...
