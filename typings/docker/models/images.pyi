"""
This type stub file was generated by pyright.
"""

from .resource import Collection, Model

class Image(Model):
    """
    An image on the server.
    """

    def __repr__(self): ...
    @property
    def labels(self):  # -> dict[Any, Any]:
        """
        The labels of an image as dictionary.
        """
        ...
    @property
    def short_id(self):
        """
        The ID of the image truncated to 12 characters, plus the ``sha256:``
        prefix.
        """
        ...
    @property
    def tags(self):  # -> list[Unknown]:
        """
        The image's tags.
        """
        ...
    def history(self):
        """
        Show the history of an image.

        Returns:
            (str): The history of the image.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        ...
    def remove(self, force=..., noprune=...):
        """
        Remove this image.

        Args:
            force (bool): Force removal of the image
            noprune (bool): Do not delete untagged parents

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        ...
    def save(self, chunk_size=..., named=...):
        """
        Get a tarball of an image. Similar to the ``docker save`` command.

        Args:
            chunk_size (int): The generator will return up to that much data
                per iteration, but may return less. If ``None``, data will be
                streamed as it is received. Default: 2 MB
            named (str or bool): If ``False`` (default), the tarball will not
                retain repository and tag information for this image. If set
                to ``True``, the first tag in the :py:attr:`~tags` list will
                be used to identify the image. Alternatively, any element of
                the :py:attr:`~tags` list can be used as an argument to use
                that specific tag as the saved identifier.

        Returns:
            (generator): A stream of raw archive data.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.

        Example:

            >>> image = cli.images.get("busybox:latest")
            >>> f = open('/tmp/busybox-latest.tar', 'wb')
            >>> for chunk in image.save():
            >>>   f.write(chunk)
            >>> f.close()
        """
        ...
    def tag(self, repository, tag=..., **kwargs):
        """
        Tag this image into a repository. Similar to the ``docker tag``
        command.

        Args:
            repository (str): The repository to set for the tag
            tag (str): The tag name
            force (bool): Force

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.

        Returns:
            (bool): ``True`` if successful
        """
        ...

class RegistryData(Model):
    """
    Image metadata stored on the registry, including available platforms.
    """

    def __init__(self, image_name, *args, **kwargs) -> None: ...
    @property
    def id(self):
        """
        The ID of the object.
        """
        ...
    @property
    def short_id(self):
        """
        The ID of the image truncated to 12 characters, plus the ``sha256:``
        prefix.
        """
        ...
    def pull(self, platform=...):
        """
        Pull the image digest.

        Args:
            platform (str): The platform to pull the image for.
            Default: ``None``

        Returns:
            (:py:class:`Image`): A reference to the pulled image.
        """
        ...
    def has_platform(self, platform):  # -> bool:
        """
        Check whether the given platform identifier is available for this
        digest.

        Args:
            platform (str or dict): A string using the ``os[/arch[/variant]]``
                format, or a platform dictionary.

        Returns:
            (bool): ``True`` if the platform is recognized as available,
            ``False`` otherwise.

        Raises:
            :py:class:`docker.errors.InvalidArgument`
                If the platform argument is not a valid descriptor.
        """
        ...
    def reload(self): ...

class ImageCollection(Collection):
    model = Image
    def build(self, **kwargs):  # -> Model | tuple[Model, Iterator[Any]]:
        """
        Build an image and return it. Similar to the ``docker build``
        command. Either ``path`` or ``fileobj`` must be set.

        If you already have a tar file for the Docker build context (including
        a Dockerfile), pass a readable file-like object to ``fileobj``
        and also pass ``custom_context=True``. If the stream is also
        compressed, set ``encoding`` to the correct value (e.g ``gzip``).

        If you want to get the raw output of the build, use the
        :py:meth:`~docker.api.build.BuildApiMixin.build` method in the
        low-level API.

        Args:
            path (str): Path to the directory containing the Dockerfile
            fileobj: A file object to use as the Dockerfile. (Or a file-like
                object)
            tag (str): A tag to add to the final image
            quiet (bool): Whether to return the status
            nocache (bool): Don't use the cache when set to ``True``
            rm (bool): Remove intermediate containers. The ``docker build``
                command now defaults to ``--rm=true``, but we have kept the old
                default of `False` to preserve backward compatibility
            timeout (int): HTTP timeout
            custom_context (bool): Optional if using ``fileobj``
            encoding (str): The encoding for a stream. Set to ``gzip`` for
                compressing
            pull (bool): Downloads any updates to the FROM image in Dockerfiles
            forcerm (bool): Always remove intermediate containers, even after
                unsuccessful builds
            dockerfile (str): path within the build context to the Dockerfile
            buildargs (dict): A dictionary of build arguments
            container_limits (dict): A dictionary of limits applied to each
                container created by the build process. Valid keys:

                - memory (int): set memory limit for build
                - memswap (int): Total memory (memory + swap), -1 to disable
                    swap
                - cpushares (int): CPU shares (relative weight)
                - cpusetcpus (str): CPUs in which to allow execution, e.g.,
                    ``"0-3"``, ``"0,1"``
            shmsize (int): Size of `/dev/shm` in bytes. The size must be
                greater than 0. If omitted the system uses 64MB
            labels (dict): A dictionary of labels to set on the image
            cache_from (list): A list of images used for build cache
                resolution
            target (str): Name of the build-stage to build in a multi-stage
                Dockerfile
            network_mode (str): networking mode for the run commands during
                build
            squash (bool): Squash the resulting images layers into a
                single layer.
            extra_hosts (dict): Extra hosts to add to /etc/hosts in building
                containers, as a mapping of hostname to IP address.
            platform (str): Platform in the format ``os[/arch[/variant]]``.
            isolation (str): Isolation technology used during build.
                Default: `None`.
            use_config_proxy (bool): If ``True``, and if the docker client
                configuration file (``~/.docker/config.json`` by default)
                contains a proxy configuration, the corresponding environment
                variables will be set in the container being built.

        Returns:
            (tuple): The first item is the :py:class:`Image` object for the
                image that was built. The second item is a generator of the
                build logs as JSON-decoded objects.

        Raises:
            :py:class:`docker.errors.BuildError`
                If there is an error during the build.
            :py:class:`docker.errors.APIError`
                If the server returns any other error.
            ``TypeError``
                If neither ``path`` nor ``fileobj`` is specified.
        """
        ...
    def get(self, name):  # -> Model:
        """
        Gets an image.

        Args:
            name (str): The name of the image.

        Returns:
            (:py:class:`Image`): The image.

        Raises:
            :py:class:`docker.errors.ImageNotFound`
                If the image does not exist.
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        ...
    def get_registry_data(self, name, auth_config=...):  # -> RegistryData:
        """
        Gets the registry data for an image.

        Args:
            name (str): The name of the image.
            auth_config (dict): Override the credentials that are found in the
                config for this request.  ``auth_config`` should contain the
                ``username`` and ``password`` keys to be valid.

        Returns:
            (:py:class:`RegistryData`): The data object.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        ...
    def list(self, name=..., all=..., filters=...):  # -> list[Model]:
        """
        List images on the server.

        Args:
            name (str): Only show images belonging to the repository ``name``
            all (bool): Show intermediate image layers. By default, these are
                filtered out.
            filters (dict): Filters to be processed on the image list.
                Available filters:
                - ``dangling`` (bool)
                - `label` (str|list): format either ``"key"``, ``"key=value"``
                    or a list of such.

        Returns:
            (list of :py:class:`Image`): The images.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        ...
    def load(self, data):  # -> list[Model]:
        """
        Load an image that was previously saved using
        :py:meth:`~docker.models.images.Image.save` (or ``docker save``).
        Similar to ``docker load``.

        Args:
            data (binary): Image data to be loaded.

        Returns:
            (list of :py:class:`Image`): The images.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        ...
    def pull(self, repository, tag=..., all_tags=..., **kwargs):  # -> Model | list[Model]:
        """
        Pull an image of the given name and return it. Similar to the
        ``docker pull`` command.
        If ``tag`` is ``None`` or empty, it is set to ``latest``.
        If ``all_tags`` is set, the ``tag`` parameter is ignored and all image
        tags will be pulled.

        If you want to get the raw pull output, use the
        :py:meth:`~docker.api.image.ImageApiMixin.pull` method in the
        low-level API.

        Args:
            repository (str): The repository to pull
            tag (str): The tag to pull
            auth_config (dict): Override the credentials that are found in the
                config for this request.  ``auth_config`` should contain the
                ``username`` and ``password`` keys to be valid.
            platform (str): Platform in the format ``os[/arch[/variant]]``
            all_tags (bool): Pull all image tags

        Returns:
            (:py:class:`Image` or list): The image that has been pulled.
                If ``all_tags`` is True, the method will return a list
                of :py:class:`Image` objects belonging to this repository.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.

        Example:

            >>> # Pull the image tagged `latest` in the busybox repo
            >>> image = client.images.pull('busybox')

            >>> # Pull all tags in the busybox repo
            >>> images = client.images.pull('busybox', all_tags=True)
        """
        ...
    def push(self, repository, tag=..., **kwargs): ...
    def remove(self, *args, **kwargs): ...
    def search(self, *args, **kwargs): ...
    def prune(self, filters=...): ...
    def prune_builds(self, *args, **kwargs): ...

def normalize_platform(platform, engine_info): ...
