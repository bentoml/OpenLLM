from __future__ import annotations

import sys, typing

import pytest, pexpect


@pytest.fixture
def pexpect_process() -> typing.Generator[pexpect.spawn[typing.Any], None, None]:
  child = pexpect.spawn(
    f'{sys.executable} -m openllm hello', encoding='utf-8', timeout=20, echo=False
  )
  try:
    yield child
  finally:
    try:
      child.sendcontrol('c')
      child.close(force=True)
    except:
      pass


def safe_expect(
  child: pexpect.spawn, pattern: str, timeout: int = 10, debug_msg: str = 'Expecting pattern'
) -> int:
  try:
    print(f"\n{debug_msg}: '{pattern}'")
    index = child.expect(pattern, timeout=timeout)
    print(f'Found match at index {index}')
    print(f'Before match: {child.before}')
    print(f'After match: {child.after}')
    return index
  except pexpect.TIMEOUT:
    print(f'TIMEOUT while {debug_msg}')
    print(f'Last output: {child.before}')
    raise
  except pexpect.EOF:
    print(f'EOF while {debug_msg}')
    print(f'Last output: {child.before}')
    raise


def test_hello_flow_to_deploy(pexpect_process: pexpect.spawn) -> None:
  child = pexpect_process

  try:
    safe_expect(child, 'Select a model', timeout=10, debug_msg='Waiting for model selection prompt')

    child.sendline('\x1b[B')
    child.sendline('\r')

    safe_expect(
      child, 'Select a version', timeout=10, debug_msg='Waiting for version selection prompt'
    )

    child.sendline('\r')

    safe_expect(
      child, 'Select an action', timeout=10, debug_msg='Waiting for action selection prompt'
    )

    child.sendline('\x1b[B')
    child.sendline('\x1b[B')

    child.sendline('\r')

    safe_expect(
      child, 'Select an instance type', timeout=10, debug_msg='Waiting for instance type prompt'
    )

    child.sendline('\r')

    child.expect('Error: .*HF_TOKEN', timeout=10)
  except Exception as e:
    pytest.fail(f'Test failed with exception: {e}')
