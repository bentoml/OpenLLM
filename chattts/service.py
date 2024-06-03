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
        self.chat.load_models(compile=False)  # Set to True for better performance

    @bentoml.api
    def tts(
        self,
        text: str = "PUT YOUR TEXT HERE",
        speaker: str = "2",
    ) -> Annotated[pathlib.Path, ContentType("audio/wav")]:
        rhythm: bool = True
        temperature: float = 0.3
        top_P: float = 0.7
        top_K: int = 20

        import torch
        import torchaudio

        if speaker:
            seed = int(speaker, 16)
            torch.manual_seed(seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

        dim = self.chat.pretrain_models["gpt"].gpt.layers[0].mlp.gate_proj.in_features
        std, mean = self.chat.pretrain_models["spk_stat"].chunk(2)
        rand_spk = torch.randn(dim, device=std.device) * std + mean

        params_infer_code = {
            "spk_emb": rand_spk,
            "temperature": temperature,
            "top_P": top_P,
            "top_K": top_K,
        }

        wavs = self.chat.infer(
            [text],
            skip_refine_text=rhythm,
            params_infer_code=params_infer_code,
        )

        output_io = io.BytesIO()
        torchaudio.save(output_io, torch.from_numpy(wavs[0]), 24000, format="wav")
        return output_io.getvalue()
