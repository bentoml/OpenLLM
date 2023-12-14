def prepare_logits_processor(config):
  import transformers

  generation_config = config.generation_config
  logits_processor = transformers.LogitsProcessorList()
  if generation_config['temperature'] >= 1e-5 and generation_config['temperature'] != 1.0:
    logits_processor.append(transformers.TemperatureLogitsWarper(generation_config['temperature']))
  if generation_config['repetition_penalty'] > 1.0:
    logits_processor.append(transformers.RepetitionPenaltyLogitsProcessor(generation_config['repetition_penalty']))
  if 1e-8 <= generation_config['top_p']:
    logits_processor.append(transformers.TopPLogitsWarper(generation_config['top_p']))
  if generation_config['top_k'] > 0:
    logits_processor.append(transformers.TopKLogitsWarper(generation_config['top_k']))
  return logits_processor


# NOTE: The ordering here is important. Some models have two of these and we have a preference for which value gets used.
SEQLEN_KEYS = ['max_sequence_length', 'seq_length', 'max_position_embeddings', 'max_seq_len', 'model_max_length']


def get_context_length(config):
  rope_scaling = getattr(config, 'rope_scaling', None)
  rope_scaling_factor = config.rope_scaling['factor'] if rope_scaling else 1.0
  for key in SEQLEN_KEYS:
    if getattr(config, key, None) is not None:
      return int(rope_scaling_factor * getattr(config, key))
  return 2048


def is_sentence_complete(output):
  return output.endswith(('.', '?', '!', '...', '。', '?', '!', '…', '"', "'", '”'))


def is_partial_stop(output, stop_str):
  for i in range(min(len(output), len(stop_str))):
    if stop_str.startswith(output[-i:]):
      return True
  return False
