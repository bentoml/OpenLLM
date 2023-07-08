from typing import Any

from .base import DictType

class LogConfigTypesEnum:
    _values = ...

class LogConfig(DictType):
    """
    Configure logging for a container, when provided as an argument to
    :py:meth:`~docker.api.container.ContainerApiMixin.create_host_config`.
    You may refer to the
    `official logging driver documentation <https://docs.docker.com/config/containers/logging/configure/>`_
    for more information.

    Args:
        type (str): Indicate which log driver to use. A set of valid drivers
            is provided as part of the :py:attr:`LogConfig.types`
            enum. Other values may be accepted depending on the engine version
            and available logging plugins.
        config (dict): A driver-dependent configuration dictionary. Please
            refer to the driver's documentation for a list of valid config
            keys.

    Example:

        >>> from docker.types import LogConfig
        >>> lc = LogConfig(type=LogConfig.types.JSON, config={
        ...   'max-size': '1g',
        ...   'labels': 'production_status,geo'
        ... })
        >>> hc = client.create_host_config(log_config=lc)
        >>> container = client.create_container('busybox', 'true',
        ...    host_config=hc)
        >>> client.inspect_container(container)['HostConfig']['LogConfig']
        {'Type': 'json-file', 'Config': {'labels': 'production_status,geo', 'max-size': '1g'}}
    """

    types = LogConfigTypesEnum
    def __init__(self, **kwargs: Any) -> None: ...
    @property
    def type(self): ...
    @type.setter
    def type(self, value: Any): ...
    @property
    def config(self): ...
    def set_config_value(self, key: str, value: Any) -> None:
        """Set a the value for ``key`` to ``value`` inside the ``config``
        dict.
        """
        ...
    def unset_config(self, key: str) -> None:
        """Remove the ``key`` property from the ``config`` dict."""
        ...

class Ulimit(DictType):
    """
    Create a ulimit declaration to be used with
    :py:meth:`~docker.api.container.ContainerApiMixin.create_host_config`.

    Args:

        name (str): Which ulimit will this apply to. The valid names can be
            found in '/etc/security/limits.conf' on a gnu/linux system.
        soft (int): The soft limit for this ulimit. Optional.
        hard (int): The hard limit for this ulimit. Optional.

    Example:

        >>> nproc_limit = docker.types.Ulimit(name='nproc', soft=1024)
        >>> hc = client.create_host_config(ulimits=[nproc_limit])
        >>> container = client.create_container(
                'busybox', 'true', host_config=hc
            )
        >>> client.inspect_container(container)['HostConfig']['Ulimits']
        [{'Name': 'nproc', 'Hard': 0, 'Soft': 1024}]

    """

    def __init__(self, **kwargs) -> None: ...
    @property
    def name(self): ...
    @name.setter
    def name(self, value): ...
    @property
    def soft(self): ...
    @soft.setter
    def soft(self, value): ...
    @property
    def hard(self): ...
    @hard.setter
    def hard(self, value): ...

class DeviceRequest(DictType):
    """
    Create a device request to be used with
    :py:meth:`~docker.api.container.ContainerApiMixin.create_host_config`.

    Args:

        driver (str): Which driver to use for this device. Optional.
        count (int): Number or devices to request. Optional.
            Set to -1 to request all available devices.
        device_ids (list): List of strings for device IDs. Optional.
            Set either ``count`` or ``device_ids``.
        capabilities (list): List of lists of strings to request
            capabilities. Optional. The global list acts like an OR,
            and the sub-lists are AND. The driver will try to satisfy
            one of the sub-lists.
            Available capabilities for the ``nvidia`` driver can be found
            `here <https://github.com/NVIDIA/nvidia-container-runtime>`_.
        options (dict): Driver-specific options. Optional.
    """

    def __init__(
        self,
        count: int | None = ...,
        driver: str | None = ...,
        device_ids: list[str] | None = ...,
        capabilities: list[list[str]] | None = ...,
        options: dict[str, str] | None = ...,
    ) -> None: ...
    @property
    def driver(self) -> str: ...
    @driver.setter
    def driver(self, value: str) -> None: ...
    @property
    def count(self) -> int: ...
    @count.setter
    def count(self, value: int) -> None: ...
    @property
    def device_ids(self): ...
    @device_ids.setter
    def device_ids(self, value): ...
    @property
    def capabilities(self): ...
    @capabilities.setter
    def capabilities(self, value): ...
    @property
    def options(self): ...
    @options.setter
    def options(self, value): ...

class HostConfig(dict):
    def __init__(
        self,
        version,
        binds=...,
        port_bindings=...,
        lxc_conf=...,
        publish_all_ports=...,
        links=...,
        privileged=...,
        dns=...,
        dns_search=...,
        volumes_from=...,
        network_mode=...,
        restart_policy=...,
        cap_add=...,
        cap_drop=...,
        devices=...,
        extra_hosts=...,
        read_only=...,
        pid_mode=...,
        ipc_mode=...,
        security_opt=...,
        ulimits=...,
        log_config=...,
        mem_limit=...,
        memswap_limit=...,
        mem_reservation=...,
        kernel_memory=...,
        mem_swappiness=...,
        cgroup_parent=...,
        group_add=...,
        cpu_quota=...,
        cpu_period=...,
        blkio_weight=...,
        blkio_weight_device=...,
        device_read_bps=...,
        device_write_bps=...,
        device_read_iops=...,
        device_write_iops=...,
        oom_kill_disable=...,
        shm_size=...,
        sysctls=...,
        tmpfs=...,
        oom_score_adj=...,
        dns_opt=...,
        cpu_shares=...,
        cpuset_cpus=...,
        userns_mode=...,
        uts_mode=...,
        pids_limit=...,
        isolation=...,
        auto_remove=...,
        storage_opt=...,
        init=...,
        init_path=...,
        volume_driver=...,
        cpu_count=...,
        cpu_percent=...,
        nano_cpus=...,
        cpuset_mems=...,
        runtime=...,
        mounts=...,
        cpu_rt_period=...,
        cpu_rt_runtime=...,
        device_cgroup_rules=...,
        device_requests=...,
        cgroupns=...,
    ) -> None: ...

def host_config_type_error(param, param_value, expected): ...
def host_config_version_error(param, version, less_than=...): ...
def host_config_value_error(param, param_value): ...
def host_config_incompatible_error(param, param_value, incompatible_param): ...

class ContainerConfig(dict):
    def __init__(
        self,
        version,
        image,
        command,
        hostname=...,
        user=...,
        detach=...,
        stdin_open=...,
        tty=...,
        ports=...,
        environment=...,
        volumes=...,
        network_disabled=...,
        entrypoint=...,
        working_dir=...,
        domainname=...,
        host_config=...,
        mac_address=...,
        labels=...,
        stop_signal=...,
        networking_config=...,
        healthcheck=...,
        stop_timeout=...,
        runtime=...,
    ) -> None: ...
