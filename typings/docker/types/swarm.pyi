"""
This type stub file was generated by pyright.
"""

class SwarmSpec(dict):
    """
    Describe a Swarm's configuration and options. Use
    :py:meth:`~docker.api.swarm.SwarmApiMixin.create_swarm_spec`
    to instantiate.
    """

    def __init__(
        self,
        version,
        task_history_retention_limit=...,
        snapshot_interval=...,
        keep_old_snapshots=...,
        log_entries_for_slow_followers=...,
        heartbeat_tick=...,
        election_tick=...,
        dispatcher_heartbeat_period=...,
        node_cert_expiry=...,
        external_cas=...,
        name=...,
        labels=...,
        signing_ca_cert=...,
        signing_ca_key=...,
        ca_force_rotate=...,
        autolock_managers=...,
        log_driver=...,
    ) -> None: ...

class SwarmExternalCA(dict):
    """
    Configuration for forwarding signing requests to an external
    certificate authority.

    Args:
        url (string): URL where certificate signing requests should be
            sent.
        protocol (string): Protocol for communication with the external CA.
        options (dict): An object with key/value pairs that are interpreted
            as protocol-specific options for the external CA driver.
        ca_cert (string): The root CA certificate (in PEM format) this
            external CA uses to issue TLS certificates (assumed to be to
            the current swarm root CA certificate if not provided).



    """

    def __init__(self, url, protocol=..., options=..., ca_cert=...) -> None: ...
