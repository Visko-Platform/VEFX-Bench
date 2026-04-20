"""
VEFX-Reward: Qwen3-VL based reward model for video editing quality assessment.

Extends Qwen3VLForConditionalGeneration with an rm_head for ordinal regression,
scoring video edits on Instructional Following (IF), Render Quality (RQ),
and Edit Exclusivity (EE) on a 1–4 scale.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional
from transformers import Qwen3VLForConditionalGeneration


class Qwen3VLRewardModelBT(Qwen3VLForConditionalGeneration):
    """Qwen3-VL with a reward head for ordinal video edit quality scoring."""

    def __init__(self, config, output_dim=3, reward_token="special",
                 special_token_ids=None, use_ordinal=True, num_classes=4, **kwargs):
        if 'use_cache' in kwargs:
            config.use_cache = kwargs.pop('use_cache')
        super().__init__(config, **kwargs)
        self.output_dim = output_dim
        self.rm_head = nn.Linear(config.text_config.hidden_size, output_dim, bias=False)
        nn.init.normal_(self.rm_head.weight, mean=0.0, std=1.0 / config.text_config.hidden_size)
        self.reward_token = reward_token
        self.use_ordinal = use_ordinal
        self.num_classes = num_classes
        self.special_token_ids = special_token_ids
        if self.special_token_ids is not None:
            self.reward_token = "special"

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]  # [B, L, D]
        logits = self.rm_head(hidden_states)  # [B, L, output_dim]

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        pad_token_id = self.config.text_config.pad_token_id
        if pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.eq(input_ids, pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        if self.reward_token == "last":
            pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        elif self.reward_token == "mean":
            valid_lengths = torch.clamp(sequence_lengths, min=0, max=logits.size(1) - 1)
            pooled_logits = torch.stack([logits[i, :valid_lengths[i]].mean(dim=0) for i in range(batch_size)])
        elif self.reward_token == "special":
            special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for special_token_id in self.special_token_ids:
                special_token_mask = special_token_mask | (input_ids == special_token_id)
            pooled_logits = logits[special_token_mask, ...]
            num_matched = special_token_mask.sum(dim=1)
            num_dims = num_matched[0].item()
            pooled_logits = pooled_logits.view(batch_size, num_dims, -1)
            if self.use_ordinal:
                pooled_logits = pooled_logits.view(batch_size, -1)
            else:
                if self.output_dim == num_dims:
                    pooled_logits = pooled_logits.diagonal(dim1=1, dim2=2)
                pooled_logits = pooled_logits.view(batch_size, -1)
        else:
            raise ValueError(f"Invalid reward_token: {self.reward_token}")

        return {"logits": pooled_logits}


def ordinal_predict(logits: np.ndarray, num_classes: int):
    """
    Convert CORN ordinal logits to predicted scores.

    Args:
        logits: [B, D, K-1] raw threshold logits
        num_classes: K (number of ordinal classes)

    Returns:
        hard_preds: [B, D] integer predictions in {1..K}
        soft_preds: [B, D] continuous expected value E[Y]
    """
    probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid → P(Y>k | Y>=k)
    cum_probs = np.cumprod(probs, axis=-1)  # P(Y>k) = prod_{j<=k} P(Y>j|Y>=j)

    hard_preds = (cum_probs > 0.5).sum(axis=-1) + 1  # [B, D]

    cum_ext = np.concatenate([
        np.ones((*cum_probs.shape[:-1], 1)),
        cum_probs,
        np.zeros((*cum_probs.shape[:-1], 1)),
    ], axis=-1)
    p_class = cum_ext[..., :-1] - cum_ext[..., 1:]
    p_class = np.maximum(p_class, 0)
    class_values = np.arange(1, num_classes + 1)
    soft_preds = (p_class * class_values).sum(axis=-1)

    return hard_preds, soft_preds
