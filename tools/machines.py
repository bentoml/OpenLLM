from __future__ import annotations

import httpx,os,dataclasses,logging,time,argparse,typing as t

if (ENV := os.getenv("PAPERSPACE_API_KEY")) is None: raise RuntimeError('This script requires setting "PAPERSPACE_API_KEY"')
HEADERS = httpx.Headers({'Authorization': f'Bearer {ENV}', 'Accept': 'application/json'})
API_URL = 'https://api.paperspace.com/v1'

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

@dataclasses.dataclass
class Machine:
  id: str
  inner: httpx.Client = dataclasses.field(default_factory=lambda: httpx.Client(headers=HEADERS, base_url=API_URL, timeout=60), repr=False)

  def close(self): self.inner.close()
  def __del__(self): self.close()
  def __enter__(self): return self
  def __exit__(self, *_: t.Any) -> None: self.close()
  @property
  def metadata(self) -> dict[str, t.Any]: return self.inner.get(f'/machines/{self.id}').json()
  @property
  def status(self) -> t.Literal['off', 'ready', 'stopping', 'starting']: return self.metadata['state']
  def start(self) -> bool:
    response = self.inner.patch(f'/machines/{self.id}/start')
    if response.status_code == 400 or self.status == 'ready':
      logger.error('machine is already running')
      return False
    elif response.status_code != 200:
      logger.error('Error while starting machine "%s": %s', self.id, response.json())
    return True
  def stop(self) -> bool:
    response = self.inner.patch(f'/machines/{self.id}/stop')
    if response.status_code == 400 or self.status == 'off':
      logger.error('machine is already off')
      return False
    elif response.status_code != 200:
      logger.error('Error while stopping machine "%s": %s', self.id, response.json())
    return True

def main():
  parser = argparse.ArgumentParser()
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument('--start', metavar='ID')
  group.add_argument('--stop', metavar='ID')
  args = parser.parse_args()

  if args.start:
    with Machine(id=args.start) as machine:
      if machine.start():
        while machine.status != 'ready':
          logger.info('Waiting for machine "%s" to be ready...', machine.id)
          time.sleep(5)
      else:
        logger.error('Failed to start machine "%s"', machine.id)
        return 1
  elif args.stop:
    with Machine(id=args.stop) as machine:
      if machine.stop():
        while machine.status != 'ready':
          logger.info('Waiting for machine "%s" to stop...', machine.id)
          time.sleep(5)
      else:
        logger.error('Failed to stopmachine "%s"', machine.id)
        return 1
  return 0

if __name__ == "__main__": raise SystemExit(main())
