# LangChain + BentoML + OpenLLM


Run it locally:
```bash
export SERPAPI_API_KEY="__Your_SERP_API_key__"
export BENTOML_CONFIG_OPTIONS="api_server.traffic.timeout=900 runners.traffic.timeout=900"
bentoml serve
```

Build Bento:
```bash
bentoml build
```

Generate docker image:

```bash
bentoml containerize ...
docker run \
  -e SERPAPI_API_KEY="__Your_SERP_API_key__" \
  -e BENTOML_CONFIG_OPTIONS="api_server.traffic.timeout=900 runners.traffic.timeout=900" \
  -p 3000:3000 \
  ..image_name

```
