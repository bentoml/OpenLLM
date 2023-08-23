from __future__ import annotations
import functools, os, typing as t, pytest, openllm
from bentoml._internal.configuration.containers import BentoMLContainer
if t.TYPE_CHECKING: from pathlib import Path

HF_INTERNAL_T5_TESTING = 'hf-internal-testing/tiny-random-t5'

actions_xfail = functools.partial(
    pytest.mark.xfail, condition=os.getenv('GITHUB_ACTIONS') is not None, reason='Marking GitHub Actions to xfail due to flakiness and building environment not isolated.',
)
@actions_xfail
def test_general_build_with_internal_testing():
  bento_store = BentoMLContainer.bento_store.get()

  llm = openllm.AutoLLM.for_model('flan-t5', model_id=HF_INTERNAL_T5_TESTING)
  bento = openllm.build('flan-t5', model_id=HF_INTERNAL_T5_TESTING)

  assert llm.llm_type == bento.info.labels['_type']
  assert llm.config['env']['framework_value'] == bento.info.labels['_framework']

  bento = openllm.build('flan-t5', model_id=HF_INTERNAL_T5_TESTING)
  assert len(bento_store.list(bento.tag)) == 1
@actions_xfail
def test_general_build_from_local(tmp_path_factory: pytest.TempPathFactory):
  local_path = tmp_path_factory.mktemp('local_t5')
  llm = openllm.AutoLLM.for_model('flan-t5', model_id=HF_INTERNAL_T5_TESTING, ensure_available=True)

  if llm.bettertransformer:
    llm.__llm_model__ = llm.model.reverse_bettertransformer()

  llm.save_pretrained(local_path)

  assert openllm.build('flan-t5', model_id=local_path.resolve().__fspath__(), model_version='local')
@pytest.fixture()
def dockerfile_template(tmp_path_factory: pytest.TempPathFactory):
  file = tmp_path_factory.mktemp('dockerfiles') / 'Dockerfile.template'
  file.write_text("{% extends bento_base_template %}\n{% block SETUP_BENTO_ENTRYPOINT %}\n{{ super() }}\nRUN echo 'sanity from custom dockerfile'\n{% endblock %}")
  return file
@pytest.mark.usefixtures('dockerfile_template')
@actions_xfail
def test_build_with_custom_dockerfile(dockerfile_template: Path):
  assert openllm.build('flan-t5', model_id=HF_INTERNAL_T5_TESTING, dockerfile_template=str(dockerfile_template))
