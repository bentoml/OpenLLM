"""
This type stub file was generated by pyright.
"""

from .base import DictType

class Healthcheck(DictType):
    """
    Defines a healthcheck configuration for a container or service.

    Args:
        test (:py:class:`list` or str): Test to perform to determine
            container health. Possible values:

            - Empty list: Inherit healthcheck from parent image
            - ``["NONE"]``: Disable healthcheck
            - ``["CMD", args...]``: exec arguments directly.
            - ``["CMD-SHELL", command]``: Run command in the system's
              default shell.

            If a string is provided, it will be used as a ``CMD-SHELL``
            command.
        interval (int): The time to wait between checks in nanoseconds. It
            should be 0 or at least 1000000 (1 ms).
        timeout (int): The time to wait before considering the check to
            have hung. It should be 0 or at least 1000000 (1 ms).
        retries (int): The number of consecutive failures needed to
            consider a container as unhealthy.
        start_period (int): Start period for the container to
            initialize before starting health-retries countdown in
            nanoseconds. It should be 0 or at least 1000000 (1 ms).
    """

    def __init__(self, **kwargs) -> None: ...
    @property
    def test(self): ...
    @test.setter
    def test(self, value): ...
    @property
    def interval(self): ...
    @interval.setter
    def interval(self, value): ...
    @property
    def timeout(self): ...
    @timeout.setter
    def timeout(self, value): ...
    @property
    def retries(self): ...
    @retries.setter
    def retries(self, value): ...
    @property
    def start_period(self): ...
    @start_period.setter
    def start_period(self, value): ...
