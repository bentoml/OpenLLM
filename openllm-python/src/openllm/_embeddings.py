# See https://github.com/bentoml/sentence-embedding-bento for more information.
from __future__ import annotations
import typing as t

import transformers

from huggingface_hub import snapshot_download

import bentoml
import openllm

from bentoml._internal.frameworks.transformers import API_VERSION
from bentoml._internal.frameworks.transformers import MODULE_NAME
from bentoml._internal.models.model import ModelOptions
from bentoml._internal.models.model import ModelSignature

if t.TYPE_CHECKING:
  import torch

_GENERIC_EMBEDDING_ID = 'sentence-transformers/all-MiniLM-L6-v2'
_BENTOMODEL_ID = 'sentence-transformers--all-MiniLM-L6-v2'

def get_or_download(ids: str = _BENTOMODEL_ID) -> bentoml.Model:
  try:
    return bentoml.transformers.get(ids)
  except bentoml.exceptions.NotFound:
    model_signatures = {
        k: ModelSignature(batchable=False)
        for k in ('forward', 'generate', 'contrastive_search', 'greedy_search', 'sample', 'beam_search', 'beam_sample', 'group_beam_search', 'constrained_beam_search', '__call__')
    }
    with bentoml.models.create(ids,
                               module=MODULE_NAME,
                               api_version=API_VERSION,
                               options=ModelOptions(),
                               context=openllm.utils.generate_context(framework_name='transformers'),
                               labels={
                                   'runtime': 'pt', 'framework': 'openllm'
                               },
                               signatures=model_signatures) as bentomodel:
      snapshot_download(_GENERIC_EMBEDDING_ID,
                        local_dir=bentomodel.path,
                        local_dir_use_symlinks=False,
                        ignore_patterns=['*.safetensors', '*.h5', '*.ot', '*.pdf', '*.md', '.gitattributes', 'LICENSE.txt'])
      return bentomodel

class GenericEmbeddingRunnable(bentoml.Runnable):
  SUPPORTED_RESOURCES = ('nvidia.com/gpu', 'cpu')
  SUPPORTS_CPU_MULTI_THREADING = True

  def __init__(self) -> None:
    self.device = 'cuda' if openllm.utils.device_count() > 0 else 'cpu'
    self._bentomodel = get_or_download()
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(self._bentomodel.path)
    self.model = transformers.AutoModel.from_pretrained(self._bentomodel.path)
    self.model.to(self.device)

  @bentoml.Runnable.method(batchable=True, batch_dim=0)
  def encode(self, sentences: list[str]) -> t.Sequence[openllm.EmbeddingsOutput]:
    import torch
    import torch.nn.functional as F
    encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.device)
    attention_mask = encoded_input['attention_mask']
    # Compute token embeddings
    with torch.no_grad():
      model_output = self.model(**encoded_input)
    # Perform pooling and normalize
    sentence_embeddings = F.normalize(self.mean_pooling(model_output, attention_mask), p=2, dim=1)
    return [openllm.EmbeddingsOutput(embeddings=sentence_embeddings.cpu().numpy(), num_tokens=int(torch.sum(attention_mask).item()))]

  @staticmethod
  def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    import torch
    # Mean Pooling - Take attention mask into account for correct averaging
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

__all__ = ['GenericEmbeddingRunnable']
