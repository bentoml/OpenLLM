from __future__ import annotations

FRAMEWORK_TO_AUTOCLASS_MAPPING = {
    'pt': ('AutoModelForCausalLM', 'AutoModelForSeq2SeqLM'),
    'tf': ('TFAutoModelForCausalLM', 'TFAutoModelForSeq2SeqLM'),
    'flax': ('FlaxAutoModelForCausalLM', 'FlaxAutoModelForSeq2SeqLM'),
    'vllm': ('AutoModelForCausalLM', 'AutoModelForSeq2SeqLM')
}
HUB_ATTRS = ['cache_dir', 'code_revision', 'force_download', 'local_files_only', 'proxies', 'resume_download', 'revision', 'subfolder', 'use_auth_token']
