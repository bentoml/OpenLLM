import asyncio
import os

from openllm.common import async_run_command


def test_async_run_command_does_not_invoke_shell(tmp_path):
  marker = tmp_path / 'shell_injection_marker.txt'
  if os.name == 'nt':
    separator = '&'
    payload = ['cmd', '/c', f'echo PWNED>{marker}']
  else:
    separator = ';'
    payload = ['sh', '-c', f'echo PWNED > {marker}']

  async def run() -> None:
    cmd = ['python', '-c', 'pass', separator, *payload]
    async with async_run_command(cmd, cwd=str(tmp_path), silent=True) as proc:
      await proc.wait()

  asyncio.run(run())

  assert not marker.exists()
