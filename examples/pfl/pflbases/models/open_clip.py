"""
Implementation of CLIP from the open clip repository.

https://github.com/mlfoundations/open_clip

[1]. Learning Transferable Visual Models From Natural Language Supervision, 21.

"""

import logging
from typing import List

import torch
from torch import nn
import numpy as np

import open_clip
from open_clip import tokenizer

# open_clip.list_pretrained()

from plato.config import Config


class CLIP(nn.Module):
    def __init__(self):
        # get the name of clip model
        # for example, convnext_base_w
        model_name = Config().trainer.model_name
        # get the name of pretrained dataset
        # for example, laion2b_s13b_b82k_augreg
        pretrained_data_name = Config().trainer.pretrained_dataset
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained_data_name
        )

        logging.info(
            "Pre-trained Model parameters:",
            f"{np.sum([int(np.prod(p.shape)) for p in self.model.parameters()]):,}",
        )

    def model_forward(self, images=None, text_prompts=None):
        """Forwarding the model to get embeddings."""
        text_tokens = tokenizer.tokenize(text_prompts)

        with torch.no_grad():
            image_features = self.model.encode_image(images).float()
            text_features = self.model.encode_text(text_tokens).float()

        return {"text_embeds": text_features, "image_embeds": image_features}

    def classification_forward(
        self,
        images: torch.Tensor,
        text_prompts: List[List[str]],
        is_attain_encodings: bool = True,
    ):
        """Forwarding the model for prediction."""

        model_outputs = self.model_forward(images, text_prompts)

        image_features = model_outputs["image_embeds"]
        text_features = model_outputs["text_embeds"]

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        top_probs = text_probs.cpu().topk(5, dim=-1)

        outputs = {"probs": top_probs}

        if is_attain_encodings:
            outputs["text_embeds"] = text_features
            outputs["image_embeds"] = image_features

        return outputs
