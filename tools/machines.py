from __future__ import annotations

import httpx, os, dataclasses, typing as t

if (ENV := os.getenv("PAPERSPACE_API_KEY")) is None: raise RuntimeError('This script requires setting "PAPERSPACE_API_KEY"')
HEADERS = httpx.Headers({'Authorization': f'Bearer {ENV}', 'Content-Type': 'application/json', 'Accept': 'application/json'})
API_URL = 'https://api.paperspace.com/v1'

@dataclasses.dataclass
class Machines:
  id: str
  inner: httpx.Client = dataclasses.field(default_factory=lambda: httpx.Client(headers=HEADERS, base_url=API_URL, timeout=60), repr=False)

  def close(self): self.inner.close()
  def __del__(self): self.close()
  @property
  def metadata(self) -> dict[str, t.Any]: return self.inner.get(f'/machines/{self.id}').json()
  @property
  def status(self) -> str: return self.metadata['state']
  def start(self) -> bool:
    response = self.inner.patch(f'/machines/{self.id}/start')
    if response.status_code != 200: breakpoint()
    return True
  def stop(self) -> bool:
    response = self.inner.patch(f'/machines/{self.id}/stop')
    if response.status_code != 200: breakpoint()
    return True
