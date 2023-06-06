"""
Implementation of CLIP [1] model.

[1]. Learning Transferable Visual Models From Natural Language Supervision, 21.

Note that images are processed within the forward function by applying the self.processor
Therefore, there is no need to have a visual transform in the data loading part.
"""

import logging

from typing import List

import torch
from torch import nn
import numpy as np
from transformers import CLIPProcessor, CLIPModel

from plato.config import Config


class CLIP(nn.Module):
    def __init__(self):
        # get the name of clip model
        # for example, openai/clip-vit-base-patch32
        model_source = "openai"
        model_name = Config().trainer.model_name

        # set the name of pretrained model
        pretrained_model = model_source + "/" + model_name

        # define the model and processor
        self.model = CLIPModel.from_pretrained(pretrained_model)
        self.processor = CLIPProcessor.from_pretrained(pretrained_model)

        logging.info(
            "Pre-trained Model parameters:",
            f"{np.sum([int(np.prod(p.shape)) for p in self.model.parameters()]):,}",
        )

    def model_forward(self, images=None, text_prompts=None):
        """Forwarding the model to get embeddings."""
        # preprocess images within the forward part
        inputs = self.processor(
            text=text_prompts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        return self.model(**inputs)

    def classification_forward(
        self,
        images: torch.Tensor,
        text_prompts: List[List[str]],
        is_attain_encodings: bool = True,
    ):
        """Forwarding the model for prediction of classification."""
        model_outputs = self.model_forward(images, text_prompts)

        # the image-text similarity score
        logits_per_image = model_outputs.logits_per_image

        # take the softmax to get the label probabilities
        probs = logits_per_image.softmax(dim=1)

        outputs = {"probs": probs}

        if is_attain_encodings:
            outputs["text_embeds"] = model_outputs.text_embeds
            outputs["image_embeds"] = model_outputs.image_embeds

        return outputs
