from __future__ import annotations
import logging
import re
import typing as t

import openllm
from openllm_core._typing_compat import overload
from openllm_core.config.configuration_dolly_v2 import DEFAULT_PROMPT_TEMPLATE
from openllm_core.config.configuration_dolly_v2 import END_KEY
from openllm_core.config.configuration_dolly_v2 import RESPONSE_KEY
from openllm_core.config.configuration_dolly_v2 import get_special_token_id
if t.TYPE_CHECKING: import torch, transformers, tensorflow as tf
else:
  torch, transformers, tf = openllm.utils.LazyLoader('torch', globals(), 'torch'), openllm.utils.LazyLoader('transformers', globals(),
                                                                                                            'transformers'), openllm.utils.LazyLoader('tf', globals(), 'tensorflow')
logger = logging.getLogger(__name__)

@overload
def get_pipeline(model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer, _init: t.Literal[True] = True, **attrs: t.Any) -> transformers.Pipeline:
  ...

@overload
def get_pipeline(model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer, _init: t.Literal[False] = ..., **attrs: t.Any) -> type[transformers.Pipeline]:
  ...

def get_pipeline(model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer, _init: bool = False, **attrs: t.Any) -> type[transformers.Pipeline] | transformers.Pipeline:
  # Lazy loading the pipeline. See databricks' implementation on HuggingFace for more information.
  class InstructionTextGenerationPipeline(transformers.Pipeline):
    def __init__(self, *args: t.Any, do_sample: bool = True, max_new_tokens: int = 256, top_p: float = 0.92, top_k: int = 0, **kwargs: t.Any):
      super().__init__(*args, model=model, tokenizer=tokenizer, do_sample=do_sample, max_new_tokens=max_new_tokens, top_p=top_p, top_k=top_k, **kwargs)

    def _sanitize_parameters(self, return_full_text: bool | None = None, **generate_kwargs: t.Any) -> tuple[dict[str, t.Any], dict[str, t.Any], dict[str, t.Any]]:
      if t.TYPE_CHECKING: assert self.tokenizer is not None
      preprocess_params: dict[str, t.Any] = {}
      # newer versions of the tokenizer configure the response key as a special token.  newer versions still may
      # append a newline to yield a single token.  find whatever token is configured for the response key.
      tokenizer_response_key = next((token for token in self.tokenizer.additional_special_tokens if token.startswith(RESPONSE_KEY)), None)
      response_key_token_id = None
      end_key_token_id = None
      if tokenizer_response_key:
        try:
          response_key_token_id = get_special_token_id(self.tokenizer, tokenizer_response_key)
          end_key_token_id = get_special_token_id(self.tokenizer, END_KEY)
          # Ensure generation stops once it generates "### End"
          generate_kwargs['eos_token_id'] = end_key_token_id
        except ValueError:
          pass
      forward_params = generate_kwargs
      postprocess_params = {'response_key_token_id': response_key_token_id, 'end_key_token_id': end_key_token_id}
      if return_full_text is not None: postprocess_params['return_full_text'] = return_full_text
      return preprocess_params, forward_params, postprocess_params

    def preprocess(self, input_: str, **generate_kwargs: t.Any) -> t.Dict[str, t.Any]:
      if t.TYPE_CHECKING: assert self.tokenizer is not None
      prompt_text = DEFAULT_PROMPT_TEMPLATE.format(instruction=input_)
      inputs = self.tokenizer(prompt_text, return_tensors='pt')
      inputs['prompt_text'] = prompt_text
      inputs['instruction_text'] = input_
      return t.cast(t.Dict[str, t.Any], inputs)

    def _forward(self, input_tensors: dict[str, t.Any], **generate_kwargs: t.Any) -> transformers.utils.generic.ModelOutput:
      if t.TYPE_CHECKING: assert self.tokenizer is not None
      input_ids, attention_mask = input_tensors['input_ids'], input_tensors.get('attention_mask', None)
      if input_ids.shape[1] == 0: input_ids, attention_mask, in_b = None, None, 1
      else: in_b = input_ids.shape[0]
      generated_sequence = self.model.generate(input_ids=input_ids.to(self.model.device) if input_ids is not None else None,
                                               attention_mask=attention_mask.to(self.model.device) if attention_mask is not None else None,
                                               pad_token_id=self.tokenizer.pad_token_id,
                                               **generate_kwargs)
      out_b = generated_sequence.shape[0]
      if self.framework == 'pt':
        generated_sequence = generated_sequence.reshape(in_b, out_b // in_b, *generated_sequence.shape[1:])
      elif self.framework == 'tf':
        generated_sequence = tf.reshape(generated_sequence, (in_b, out_b // in_b, *generated_sequence.shape[1:]))
      instruction_text = input_tensors.pop('instruction_text')
      return {'generated_sequence': generated_sequence, 'input_ids': input_ids, 'instruction_text': instruction_text}

    def postprocess(self, model_outputs: dict[str, t.Any], response_key_token_id: int, end_key_token_id: int, return_full_text: bool = False) -> list[dict[t.Literal['generated_text'], str]]:
      if t.TYPE_CHECKING: assert self.tokenizer is not None
      _generated_sequence, instruction_text = model_outputs['generated_sequence'][0], model_outputs['instruction_text']
      generated_sequence: list[list[int]] = _generated_sequence.numpy().tolist()
      records: list[dict[t.Literal['generated_text'], str]] = []
      for sequence in generated_sequence:
        # The response will be set to this variable if we can identify it.
        decoded = None
        # If we have token IDs for the response and end, then we can find the tokens and only decode between them.
        if response_key_token_id and end_key_token_id:
          # Find where "### Response:" is first found in the generated tokens.  Considering this is part of the
          # prompt, we should definitely find it.  We will return the tokens found after this token.
          try:
            response_pos = sequence.index(response_key_token_id)
          except ValueError:
            response_pos = None
          if response_pos is None:
            logger.warning('Could not find response key %s in: %s', response_key_token_id, sequence)
          if response_pos:
            # Next find where "### End" is located.  The model has been trained to end its responses with this
            # sequence (or actually, the token ID it maps to, since it is a special token).  We may not find
            # this token, as the response could be truncated.  If we don't find it then just return everything
            # to the end.  Note that even though we set eos_token_id, we still see the this token at the end.
            try:
              end_pos = sequence.index(end_key_token_id)
            except ValueError:
              end_pos = None
            decoded = self.tokenizer.decode(sequence[response_pos + 1:end_pos]).strip()
        if not decoded:
          # Otherwise we'll decode everything and use a regex to find the response and end.
          fully_decoded = self.tokenizer.decode(sequence)
          # The response appears after "### Response:".  The model has been trained to append "### End" at the
          # end.
          m = re.search(r'#+\s*Response:\s*(.+?)#+\s*End', fully_decoded, flags=re.DOTALL)
          if m: decoded = m.group(1).strip()
          else:
            # The model might not generate the "### End" sequence before reaching the max tokens.  In this case,
            # return everything after "### Response:".
            m = re.search(r'#+\s*Response:\s*(.+)', fully_decoded, flags=re.DOTALL)
            if m: decoded = m.group(1).strip()
            else: logger.warning('Failed to find response in:\n%s', fully_decoded)
        # If the full text is requested, then append the decoded text to the original instruction.
        # This technically isn't the full text, as we format the instruction in the prompt the model has been
        # trained on, but to the client it will appear to be the full text.
        if return_full_text: decoded = f'{instruction_text}\n{decoded}'
        records.append({'generated_text': t.cast(str, decoded)})
      return records

  return InstructionTextGenerationPipeline() if _init else InstructionTextGenerationPipeline

class DollyV2(openllm.LLM['transformers.Pipeline', 'transformers.PreTrainedTokenizer']):
  __openllm_internal__ = True

  @property
  def import_kwargs(self) -> tuple[dict[str, t.Any], dict[str, t.Any]]:
    return {'device_map': 'auto' if torch.cuda.is_available() and torch.cuda.device_count() > 1 else None, 'torch_dtype': torch.bfloat16}, {}

  def load_model(self, *args: t.Any, **attrs: t.Any) -> transformers.Pipeline:
    return get_pipeline(transformers.AutoModelForCausalLM.from_pretrained(self._bentomodel.path, *args, **attrs), self.tokenizer, _init=True, return_full_text=self.config.return_full_text)

  def generate(self, prompt: str, **attrs: t.Any) -> list[dict[t.Literal['generated_text'], str]]:
    llm_config = self.config.model_construct_env(**attrs)
    with torch.inference_mode():
      return self.model(prompt, return_full_text=llm_config.return_full_text, generation_config=llm_config.to_generation_config())
