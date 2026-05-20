from __future__ import annotations

import asyncio

from openllm.common import async_run_command


async def _communicate(cmd: list[str]) -> tuple[int | None, str, str]:
  async with async_run_command(cmd, silent=True) as proc:
    stdout, stderr = await proc.communicate()
  return proc.returncode, stdout.decode().strip(), stderr.decode().strip()


def test_async_run_command_preserves_shell_metacharacters() -> None:
  returncode, stdout, stderr = asyncio.run(
    _communicate(['python', '-c', 'import sys; print(sys.argv[1])', 'literal-$(printf injected)'])
  )

  assert returncode == 0
  assert stdout == 'literal-$(printf injected)'
  assert stderr == ''
