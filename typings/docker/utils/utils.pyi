"""
This type stub file was generated by pyright.
"""

URLComponents = ...

def create_ipam_pool(*args, **kwargs): ...
def create_ipam_config(*args, **kwargs): ...
def decode_json_header(header): ...
def compare_version(v1, v2):  # -> Literal[0, -1, 1]:
    """Compare docker versions

    >>> v1 = '1.9'
    >>> v2 = '1.10'
    >>> compare_version(v1, v2)
    1
    >>> compare_version(v2, v1)
    -1
    >>> compare_version(v2, v2)
    0
    """
    ...

def version_lt(v1, v2): ...
def version_gte(v1, v2): ...
def convert_port_bindings(port_bindings): ...
def convert_volume_binds(binds): ...
def convert_tmpfs_mounts(tmpfs): ...
def convert_service_networks(networks): ...
def parse_repository_tag(repo_name): ...
def parse_host(addr, is_win32=..., tls=...): ...
def parse_devices(devices): ...
def kwargs_from_env(ssl_version=..., assert_hostname=..., environment=...): ...
def convert_filters(filters): ...
def datetime_to_timestamp(dt):
    """Convert a UTC datetime to a Unix timestamp"""
    ...

def parse_bytes(s): ...
def normalize_links(links): ...
def parse_env_file(env_file):  # -> dict[Unknown, Unknown]:
    """
    Reads a line-separated environment file.
    The format of each line should be "key=value".
    """
    ...

def split_command(command): ...
def format_environment(environment): ...
def format_extra_hosts(extra_hosts, task=...): ...
def create_host_config(self, *args, **kwargs): ...
