import os
import yaml
from typing import Annotated
import sys
import shutil
import io
import pathlib

import bentoml
from bentoml.validators import ContentType
from bento_constants import CONSTANT_YAML


CONSTANTS = yaml.safe_load(CONSTANT_YAML)

CHATTTS_PATH = os.path.join(os.path.dirname(__file__), "ChatTTS")

@bentoml.service(**CONSTANTS["service_config"])
class Main:

    @bentoml.on_deployment
    @staticmethod
    def on_deployment():
        CHAT_TTS_REPO = os.environ.get("CHAT_TTS_REPO")
        assert CHAT_TTS_REPO, "CHAT_TTS_REPO environment variable is not set"

        import dulwich
        import dulwich.errors
        import dulwich.porcelain

        if os.path.exists(CHATTTS_PATH):
            shutil.rmtree(CHATTTS_PATH)

        dulwich.porcelain.clone(
            CHAT_TTS_REPO,
            CHATTTS_PATH,
            checkout=True,
            depth=1,
        )

    def __init__(self) -> None:
        sys.path.append(CHATTTS_PATH)

        import ChatTTS

        self.chat = ChatTTS.Chat()
        self.chat.load_models(compile=False) # Set to True for better performance

    @bentoml.api
    def tts(self, text: str = "PUT YOUR TEXT HERE") -> Annotated[pathlib.Path, ContentType("audio/wav")]:
        import torch
        import torchaudio

        wavs = self.chat.infer([text])
        output_io = io.BytesIO()
        torchaudio.save(output_io, torch.from_numpy(wavs[0]), 24000, format="wav")
        return output_io.getvalue()
