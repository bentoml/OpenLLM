# BentoChatTTS

[ChatTTS](https://github.com/2noise/ChatTTS) is a text-to-speech model designed specifically for dialogue scenario such as LLM assistant. 

## Prerequisites

- You have installed Python 3.9+ and `pip`. See the [Python downloads page](https://www.python.org/downloads/) to learn more.
- You have a basic understanding of key concepts in BentoML, such as Services. We recommend you read [Quickstart](https://docs.bentoml.com/en/latest/get-started/quickstart.html) first.
- (Optional) We recommend you create a virtual environment for dependency isolation for this project. See the [Conda documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or the [Python documentation](https://docs.python.org/3/library/venv.html) for details.

## Install Dependencies

```
pip install bentoml
pip install -r requirements.txt
```

If not already present, you need to install `libsox-dev` with your package manager first.

## Run

Start a Bento server with ChatTTS.

```
export CHAT_TTS_REPO=https://github.com/2noise/ChatTTS.git
bentoml serve
```

The server is now active at [http://localhost:3000](http://localhost:3000/). You can interact with it using the Swagger UI.

## Deploy

You can deploy the ChatTTS Bento service to BentoCloud.

[Sign up](https://www.bentoml.com/) if you haven't got a BentoCloud account.

Make sure you have [logged in to BentoCloud](https://docs.bentoml.com/en/latest/bentocloud/how-tos/manage-access-token.html), then run the following command to deploy it.

```bash
bentoml deploy .
```
