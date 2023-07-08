"""
This type stub file was generated by pyright.
"""

from .. import utils

class ServiceApiMixin:
    @utils.minimum_version("1.24")
    def create_service(
        self,
        task_template,
        name=...,
        labels=...,
        mode=...,
        update_config=...,
        networks=...,
        endpoint_config=...,
        endpoint_spec=...,
        rollback_config=...,
    ):
        """
        Create a service.

        Args:
            task_template (TaskTemplate): Specification of the task to start as
                part of the new service.
            name (string): User-defined name for the service. Optional.
            labels (dict): A map of labels to associate with the service.
                Optional.
            mode (ServiceMode): Scheduling mode for the service (replicated
                or global). Defaults to replicated.
            update_config (UpdateConfig): Specification for the update strategy
                of the service. Default: ``None``
            rollback_config (RollbackConfig): Specification for the rollback
                strategy of the service. Default: ``None``
            networks (:py:class:`list`): List of network names or IDs or
                :py:class:`~docker.types.NetworkAttachmentConfig` to attach the
                service to. Default: ``None``.
            endpoint_spec (EndpointSpec): Properties that can be configured to
                access and load balance a service. Default: ``None``.

        Returns:
            A dictionary containing an ``ID`` key for the newly created
            service.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        ...
    @utils.minimum_version("1.24")
    @utils.check_resource("service")
    def inspect_service(self, service, insert_defaults=...):
        """
        Return information about a service.

        Args:
            service (str): Service name or ID.
            insert_defaults (boolean): If true, default values will be merged
                into the service inspect output.

        Returns:
            (dict): A dictionary of the server-side representation of the
                service, including all relevant properties.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        ...
    @utils.minimum_version("1.24")
    @utils.check_resource("task")
    def inspect_task(self, task):
        """
        Retrieve information about a task.

        Args:
            task (str): Task ID

        Returns:
            (dict): Information about the task.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        ...
    @utils.minimum_version("1.24")
    @utils.check_resource("service")
    def remove_service(self, service):  # -> Literal[True]:
        """
        Stop and remove a service.

        Args:
            service (str): Service name or ID

        Returns:
            ``True`` if successful.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        ...
    @utils.minimum_version("1.24")
    def services(self, filters=..., status=...):
        """
        List services.

        Args:
            filters (dict): Filters to process on the nodes list. Valid
                filters: ``id``, ``name`` , ``label`` and ``mode``.
                Default: ``None``.
            status (bool): Include the service task count of running and
                desired tasks. Default: ``None``.

        Returns:
            A list of dictionaries containing data about each service.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        ...
    @utils.minimum_version("1.25")
    @utils.check_resource("service")
    def service_logs(
        self, service, details=..., follow=..., stdout=..., stderr=..., since=..., timestamps=..., tail=..., is_tty=...
    ):
        """
        Get log stream for a service.
        Note: This endpoint works only for services with the ``json-file``
        or ``journald`` logging drivers.

        Args:
            service (str): ID or name of the service
            details (bool): Show extra details provided to logs.
                Default: ``False``
            follow (bool): Keep connection open to read logs as they are
                sent by the Engine. Default: ``False``
            stdout (bool): Return logs from ``stdout``. Default: ``False``
            stderr (bool): Return logs from ``stderr``. Default: ``False``
            since (int): UNIX timestamp for the logs staring point.
                Default: 0
            timestamps (bool): Add timestamps to every log line.
            tail (string or int): Number of log lines to be returned,
                counting from the current end of the logs. Specify an
                integer or ``'all'`` to output all log lines.
                Default: ``all``
            is_tty (bool): Whether the service's :py:class:`ContainerSpec`
                enables the TTY option. If omitted, the method will query
                the Engine for the information, causing an additional
                roundtrip.

        Returns (generator): Logs for the service.
        """
        ...
    @utils.minimum_version("1.24")
    def tasks(self, filters=...):
        """
        Retrieve a list of tasks.

        Args:
            filters (dict): A map of filters to process on the tasks list.
                Valid filters: ``id``, ``name``, ``service``, ``node``,
                ``label`` and ``desired-state``.

        Returns:
            (:py:class:`list`): List of task dictionaries.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        ...
    @utils.minimum_version("1.24")
    @utils.check_resource("service")
    def update_service(
        self,
        service,
        version,
        task_template=...,
        name=...,
        labels=...,
        mode=...,
        update_config=...,
        networks=...,
        endpoint_config=...,
        endpoint_spec=...,
        fetch_current_spec=...,
        rollback_config=...,
    ):
        """
        Update a service.

        Args:
            service (string): A service identifier (either its name or service
                ID).
            version (int): The version number of the service object being
                updated. This is required to avoid conflicting writes.
            task_template (TaskTemplate): Specification of the updated task to
                start as part of the service.
            name (string): New name for the service. Optional.
            labels (dict): A map of labels to associate with the service.
                Optional.
            mode (ServiceMode): Scheduling mode for the service (replicated
                or global). Defaults to replicated.
            update_config (UpdateConfig): Specification for the update strategy
                of the service. Default: ``None``.
            rollback_config (RollbackConfig): Specification for the rollback
                strategy of the service. Default: ``None``
            networks (:py:class:`list`): List of network names or IDs or
                :py:class:`~docker.types.NetworkAttachmentConfig` to attach the
                service to. Default: ``None``.
            endpoint_spec (EndpointSpec): Properties that can be configured to
                access and load balance a service. Default: ``None``.
            fetch_current_spec (boolean): Use the undefined settings from the
                current specification of the service. Default: ``False``

        Returns:
            A dictionary containing a ``Warnings`` key.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        ...
