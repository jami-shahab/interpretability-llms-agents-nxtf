"""Wrapper for the Qwen2.5-Omni multimodal model."""  # noqa: N999

import json
import re

import torch
from qwen_omni_utils import process_mm_info
from transformers import (
    AutoConfig,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)

from .base import BaseModel


MODALITIES = {"text", "image", "video", "audio"}


class Qwen2_5OMNI(BaseModel):  # noqa: N801
    """Wrap the Qwen2.5-Omni model to handle multi-modal input and generate output."""

    def __init__(
        self,
        model_name="Qwen/Qwen2.5-Omni-7B",
        prompt=None,
        enable_flashattn=True,
        use_audio_in_video=True,
        return_audio=False,
    ):
        """Initialize the model and processor."""
        self.prompt = prompt
        self.return_audio = return_audio
        self.use_audio_in_video = use_audio_in_video

        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_name)

        config = AutoConfig.from_pretrained(model_name)
        config.attn_implementation = "eager"  # FORCE disable FlashAttention2

        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.disable_talker()

        self.model.eval()

        self.device = self.model.device
        self.dtype = self.model.dtype

    def prepare_input(self, inputs):
        """Prepare model input from multi-modal data.

        Processes text, audio, image, and video inputs and converts them into
        the format required by the Qwen2.5-Omni model. Creates conversation
        structure with system and user roles, applies chat template, and
        handles multi-modal preprocessing.
        """
        conversation = []

        for input_data in inputs:
            user_content = []

            for key, value in input_data.items():
                if key == "text":
                    user_content.append(
                        {
                            "type": "text",
                            "text": value,
                        }
                    )
                elif key in {"image", "video", "audio"}:
                    user_content.append({"type": key, key: value})

            conversation.append(
                [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": self.prompt,
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": user_content,  # user_content contains only question + media
                    },
                ]
            )

        if len(conversation) == 1:
            conversation = conversation[0]

        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        audios, images, videos = process_mm_info(
            conversation, use_audio_in_video=self.use_audio_in_video
        )

        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=self.use_audio_in_video,
        )

        return inputs.to(self.device)

    @torch.no_grad()
    def generate(self, inputs):
        """Generate text and audio response from the model given prepared inputs."""

        def normalize_timestamp(text):
            pattern = r"\[(.*?)\]"
            matches = re.findall(pattern, text)
            if not matches:
                return text

            def fix(ts):
                ts = ts.replace(" ", "")
                ts = ts.replace("-", " - ")
                return f"[{ts}]"

            for m in matches:
                text = text.replace(f"[{m}]", fix(m))

            return text

        if self.return_audio:
            text_ids, audio = self.model.generate(
                **inputs,
                use_audio_in_video=self.use_audio_in_video,
                max_new_tokens=128,
                do_sample=False,
            )
            text = self.processor.batch_decode(
                text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            audio = audio.reshape(-1).detach().cpu().numpy()  # need to change
        else:
            text_ids = self.model.generate(
                **inputs,
                use_audio_in_video=self.use_audio_in_video,
                return_audio=False,
                max_new_tokens=128,  # speed boost
                do_sample=False,  # deterministic & faster
            )
            text = self.processor.batch_decode(
                text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            audio = None
        text = [normalize_timestamp(t.split("assistant")[-1].strip()) for t in text]
        return text, audio


if __name__ == "__main__":
    prompt = None

    with open("data/Cooking-tutorials.json", "r") as f:
        sources = json.load(f)

    source = sources[0]
    question = source["question"]
    answer = source["answer"]
    timestamps = source["timestamps"]

    inputs = []
    for timestamp in timestamps:
        filename = timestamp.replace(".txt", "")
        inputs.append(
            {
                "text": question,
                "audio": f"data/Cooking-tutorials/original-audio/{filename}.m4a",
                "video": f"data/Cooking-tutorials/original-videos/{filename}.mp4",
            }
        )
        break
    model = Qwen2_5OMNI(model_name="Qwen/Qwen2.5-Omni-7B", prompt=prompt)
    inputs = model.prepare_input(inputs)
    text, audio = model.generate(inputs)

    print(text)
