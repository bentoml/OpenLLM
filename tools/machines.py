from __future__ import annotations

import httpx,os,dataclasses,datetime,time,argparse,typing as t

if (ENV := os.getenv("PAPERSPACE_API_KEY")) is None: raise RuntimeError('This script requires setting "PAPERSPACE_API_KEY"')
HEADERS = httpx.Headers({'Authorization': f'Bearer {ENV}', 'Accept': 'application/json'})
API_URL = 'https://api.paperspace.com/v1'

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
      print('machine is already running')
      return False
    elif response.status_code != 200: raise ValueError(f'Error while starting machine: {response.json()}')
    return True
  def stop(self) -> bool:
    response = self.inner.patch(f'/machines/{self.id}/stop')
    if response.status_code == 400 or self.status == 'off':
      print('machine is already off')
      return False
    elif response.status_code != 200: raise ValueError(f'Error while stopping machine {response.json()}')
    return True
  @classmethod
  def ci(cls, template_id: str):
    client = httpx.Client(headers=HEADERS, base_url=API_URL, timeout=60)
    machines = client.get('/machines', params=dict(limit=1, name='openllm-ci')).json()
    if len(machines['items']) == 1:
      return cls(id=machines['items'][0]['id'], inner=client)
    response = client.post('/machines', json=dict(
      name=f'openllm-ci-{datetime.datetime.now().timestamp()}',
      machineType='A100-80G', templateId=template_id,
      networkId=os.getenv("PAPERSPACE_NETWORK_ID"),
      diskSize=500, region='ny2', publicIpType="dynamic", startOnCreate=True,
    ))
    if response.status_code != 200: raise ValueError(f'Failed while creating a machine: {response.json()}')
    return cls(id=response.json()['data']['id'], inner=client)
  def actions(self): return f'publicIp={self.metadata["publicIp"]}'

def main():
  parser = argparse.ArgumentParser()
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument('--start', metavar='ID')
  group.add_argument('--stop', metavar='ID')
  group.add_argument('--delete', metavar='ID')
  group.add_argument('--ci-template', metavar='ID')
  args = parser.parse_args()

  if args.ci_template:
    machine =  Machine.ci(args.ci_template)
    while machine.status != 'ready': time.sleep(5)
    print(machine.actions())
    machine.close()
  elif args.delete:
    with httpx.Client(headers=HEADERS, base_url=API_URL, timeout=60) as client:
      response = client.delete(f'/machines/{args.delete}')
      if response.status_code != 200:
        print('Error while deleting machine %s', response.json())
        return 1
  elif args.start:
    with Machine(id=args.start) as machine:
      if machine.start():
        while machine.status != 'ready':
          print('Waiting for machine "%s" to be ready...', machine.id)
          time.sleep(5)
      else:
        print('Failed to start machine "%s"', machine.id)
        return 1
  elif args.stop:
    with Machine(id=args.stop) as machine:
      if machine.stop():
        while machine.status != 'ready':
          print('Waiting for machine "%s" to stop...', machine.id)
          time.sleep(5)
      else:
        print('Failed to stop machine "%s"', machine.id)
        return 1
  return 0

if __name__ == "__main__": raise SystemExit(main())
