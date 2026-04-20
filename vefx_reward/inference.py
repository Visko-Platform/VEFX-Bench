"""
VEFX-Reward: Video editing quality assessment inference API.

Usage:
    from vefx_reward import VEFXReward

    model = VEFXReward("xiangbog/VEFX-Reward-4B", device="cuda")
    scores = model.score("original.mp4", "edited.mp4", "add a hat to the person")
    # {'IF': 3.21, 'RQ': 2.85, 'EE': 3.54, 'Overall': 9.60}
"""

import json
import os
from collections.abc import Mapping
from typing import Optional

import numpy as np
import torch
from transformers import AutoProcessor

from .model import Qwen3VLRewardModelBT, ordinal_predict
from .prompt_template import build_prompt
from .vision_process import process_vision_info

# Default model hyperparameters (matching the released VEFX-Reward-4B)
DEFAULT_FPS = 4.0
DEFAULT_MAX_FRAME_PIXELS = 399360
DEFAULT_NUM_CLASSES = 4
DEFAULT_OUTPUT_DIM = 3
DIMS = ["IF", "RQ", "EE"]

SPECIAL_TOKENS = [
    "<|VQ_reward|>", "<|MQ_reward|>", "<|TA_reward|>",
    "<|IF_reward|>", "<|RQ_reward|>", "<|EE_reward|>",
]


class VEFXReward:
    """VEFX-Reward model for video editing quality assessment.

    Scores video edits on three dimensions (1–4 scale):
    - **IF** (Instructional Following): How well the edit follows the instruction.
    - **RQ** (Render Quality): Visual and temporal quality of the edited video.
    - **EE** (Edit Exclusivity): Whether only the intended region was modified.

    Args:
        model_path: HuggingFace model ID or local path
            (e.g., ``"xiangbog/VEFX-Reward-4B"``).
        device: Device string (default ``"cuda"``).
        dtype: Torch dtype (default ``torch.bfloat16``).
        fps: Frames per second for video sampling (default 4.0).
        max_frame_pixels: Maximum pixels per frame (default 399360).

    Example::

        model = VEFXReward("xiangbog/VEFX-Reward-4B")
        scores = model.score("original.mp4", "edited.mp4", "make it snowy")
        print(scores)
        # {'IF': 3.2, 'RQ': 2.9, 'EE': 3.5, 'Overall': 9.6}
    """

    def __init__(
        self,
        model_path: str = "xiangbog/VEFX-Reward-4B",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        fps: float = DEFAULT_FPS,
        max_frame_pixels: int = DEFAULT_MAX_FRAME_PIXELS,
    ):
        self.device = device
        self.dtype = dtype
        self.fps = fps
        self.max_frame_pixels = max_frame_pixels

        # Load config
        vefx_config_path = os.path.join(model_path, "vefx_config.json") if os.path.isdir(model_path) else None
        if vefx_config_path and os.path.exists(vefx_config_path):
            with open(vefx_config_path) as f:
                vefx_config = json.load(f)
        else:
            # Try to download from HF hub
            try:
                from huggingface_hub import hf_hub_download
                vefx_config_path = hf_hub_download(model_path, "vefx_config.json")
                with open(vefx_config_path) as f:
                    vefx_config = json.load(f)
            except Exception:
                vefx_config = {}

        self.num_classes = vefx_config.get("num_classes", DEFAULT_NUM_CLASSES)
        self.output_dim = vefx_config.get("output_dim", DEFAULT_OUTPUT_DIM)
        self.use_ordinal = vefx_config.get("use_ordinal", True)
        reward_token = vefx_config.get("reward_token", "special")

        # Load processor and add special tokens
        self.processor = AutoProcessor.from_pretrained(model_path, padding_side="right")
        existing_tokens = set(self.processor.tokenizer.get_vocab().keys())
        tokens_to_add = [t for t in SPECIAL_TOKENS if t not in existing_tokens]
        if tokens_to_add:
            self.processor.tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
        special_token_ids = self.processor.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

        # Load model
        self.model = Qwen3VLRewardModelBT.from_pretrained(
            model_path,
            dtype=dtype,
            output_dim=self.output_dim,
            reward_token=reward_token,
            special_token_ids=special_token_ids,
            use_ordinal=self.use_ordinal,
            num_classes=self.num_classes,
            use_cache=True,
        )
        self.model.resize_token_embeddings(len(self.processor.tokenizer))

        self.model.eval().to(self.device)
        print(f"VEFX-Reward loaded on {self.device} ({dtype})")

    def _prepare_input(self, data):
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            return data.to(device=self.device)
        return data

    def _build_batch(self, original_video: str, edited_video: str, instruction: str):
        """Build a single-sample batch from video paths and instruction."""
        content = [
            {
                "type": "video",
                "video": f"file://{os.path.abspath(original_video)}",
                "max_pixels": self.max_frame_pixels,
                "fps": self.fps,
                "sample_type": "uniform",
            },
            {
                "type": "video",
                "video": f"file://{os.path.abspath(edited_video)}",
                "max_pixels": self.max_frame_pixels,
                "fps": self.fps,
                "sample_type": "uniform",
            },
            {"type": "text", "text": build_prompt(instruction)},
        ]
        messages = [[{"role": "user", "content": content}]]
        image_inputs, video_inputs, video_metadata_list = process_vision_info(messages)
        video_inputs = [v.float() / 255.0 for v in video_inputs]

        texts = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        processor_kwargs = dict(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": False, "do_sample_frames": False},
        )
        if video_metadata_list:
            processor_kwargs["videos_kwargs"]["video_metadata"] = video_metadata_list
            processor_kwargs["videos_kwargs"]["return_metadata"] = True

        batch = self.processor(**processor_kwargs)
        return self._prepare_input(batch)

    def _logits_to_scores(self, logits: torch.Tensor) -> dict:
        """Convert raw ordinal logits to IF/RQ/EE scores."""
        logits_np = logits.float().cpu().numpy()
        if self.use_ordinal:
            num_dims = self.output_dim
            num_thresholds = self.num_classes - 1
            logits_reshaped = logits_np.reshape(1, num_dims, num_thresholds)
            hard, soft = ordinal_predict(logits_reshaped, self.num_classes)
            scores = {DIMS[j]: round(float(soft[0, j]), 3) for j in range(num_dims)}
        else:
            scores = {DIMS[j]: round(float(logits_np[0, j]), 3) for j in range(self.output_dim)}
        scores["Overall"] = round(sum(scores[d] for d in DIMS), 3)
        return scores

    @torch.no_grad()
    def score(
        self,
        original_video: str,
        edited_video: str,
        instruction: str,
    ) -> dict:
        """Score a single video edit.

        Args:
            original_video: Path to the original (source) video.
            edited_video: Path to the edited video.
            instruction: The editing instruction text.

        Returns:
            Dictionary with keys ``'IF'``, ``'RQ'``, ``'EE'``, ``'Overall'``.
            Each dimension is scored on a continuous 1–4 scale.
        """
        batch = self._build_batch(original_video, edited_video, instruction)
        logits = self.model(**batch, return_dict=True)["logits"]
        return self._logits_to_scores(logits)

    @torch.no_grad()
    def score_batch(
        self,
        original_videos: list[str],
        edited_videos: list[str],
        instructions: list[str],
    ) -> list[dict]:
        """Score multiple video edits (processed sequentially to avoid OOM).

        Args:
            original_videos: List of paths to original videos.
            edited_videos: List of paths to edited videos.
            instructions: List of editing instruction texts.

        Returns:
            List of score dictionaries, one per sample.
        """
        assert len(original_videos) == len(edited_videos) == len(instructions)
        results = []
        for orig, edit, inst in zip(original_videos, edited_videos, instructions):
            results.append(self.score(orig, edit, inst))
        return results
