## OpenLLM generated services for {model_id}

Available endpoints:

- `GET /v1/models`: compatible with OpenAI's models list.
- `POST /v1/chat/completions`: compatible with OpenAI's chat completions client.
- `POST /v1/generate_stream`: low-level SSE-formatted streams. Users are responsible for correct prompt format strings.
- `POST /v1/generate`: low-level one-shot generation. Users are responsible for correct prompt format strings.
- `POST /v1/helpers/messages`: helpers endpoints to return fully formatted chat messages should the model is a chat model.
- `POST /v1/metadata (deprecated)`: returns compatible metadata for OpenLLM server.
