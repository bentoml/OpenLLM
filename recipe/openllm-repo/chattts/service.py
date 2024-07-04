import io
import os
from typing import Annotated

import bentoml
import fastapi
import fastapi.staticfiles
import yaml
from bento_constants import CONSTANT_YAML
from bentoml.validators import ContentType
from fastapi.responses import FileResponse

CONSTANTS = yaml.safe_load(CONSTANT_YAML)
ROOT_DIR = os.path.join(os.path.dirname(__file__), "ui", "build")

ui_app = fastapi.FastAPI()
ui_app.mount(
    "/",
    fastapi.staticfiles.StaticFiles(directory=ROOT_DIR, html=True),
    name="static",
)


@bentoml.mount_asgi_app(ui_app, path="/")
@bentoml.service(
    **CONSTANTS["service_config"],
    http={
        "cors": {
            "enabled": True,
            "access_control_allow_origins": "*",
            "access_control_allow_methods": "POST, OPTIONS",
            "access_control_allow_headers": "Content-Type",
        },
    },
)
class Main:
    def __init__(self) -> None:
        import ChatTTS

        self.chat = ChatTTS.Chat()
        self.chat.load(compile=False)

    @bentoml.api
    def tts(
        self,
        text: str = "PUT YOUR TEXT HERE",
        speaker: int = 2,
    ) -> Annotated[bytes, ContentType("audio/wav")]:
        import ChatTTS

        rhythm: bool = True
        temperature: float = 0.3
        top_P: float = 0.7
        top_K: int = 20

        import torch
        import torchaudio

        if speaker:
            seed = int(speaker)
            torch.manual_seed(seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

        rand_spk = self.chat.sample_random_speaker()

        params_infer_code = ChatTTS.Chat.InferCodeParams(
            spk_emb=rand_spk,
            temperature=temperature,
            top_P=top_P,
            top_K=top_K,
        )

        wavs = self.chat.infer(
            [text],
            skip_refine_text=rhythm,
            params_infer_code=params_infer_code,
        )

        output_io = io.BytesIO()
        torchaudio.save(output_io, torch.from_numpy(wavs[0]), 24000, format="wav")  # type: ignore
        return output_io.getvalue()
