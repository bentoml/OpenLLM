from __future__ import annotations
from google.protobuf import __version__
if __version__.startswith("4"):
    from ._generated_pb4.service_pb2_grpc import *
else:
    from ._generated_pb3.service_pb2_grpc import *
