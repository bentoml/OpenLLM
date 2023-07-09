# Copyright 2023 BentoML Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The actual client implementation.

Use ``openllm.client`` instead.
This holds the implementation of the client, which is used to communicate with the
OpenLLM server. It is used to send requests to the server, and receive responses.
"""

from .runtimes.grpc import AsyncGrpcClient as AsyncGrpcClient
from .runtimes.grpc import GrpcClient as GrpcClient
from .runtimes.http import AsyncHTTPClient as AsyncHTTPClient
from .runtimes.http import HTTPClient as HTTPClient
