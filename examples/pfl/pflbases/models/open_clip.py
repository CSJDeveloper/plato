"""
Implementation of CLIP from the open clip repository.

https://github.com/mlfoundations/open_clip

[1]. Learning Transferable Visual Models From Natural Language Supervision, 21.

Note that images are processed within the forward function by applying the self.processor
Therefore, there is no need to have a visual transform in the data loading part.
"""

import logging
from typing import List, Union

import torch
from torch import nn
import numpy as np

import open_clip
from open_clip import tokenizer

# open_clip.list_pretrained()

from plato.config import Config


class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        # get the name of clip model
        # for example, convnext_base_w
        model_name = Config().trainer.model_name

        # get the name of pretrained dataset
        # for example, laion2b_s13b_b82k_augreg
        pretrained_data_name = Config().trainer.pretrained_dataset

        # get the model and preprocessor
        self.model, _, self.preprocesser = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained_data_name
        )
        # set the prompts features
        self.prompt_encodings = None

        total_n_parameters = np.sum(
            [int(np.prod(p.shape)) for p in self.model.parameters()]
        )

        logging.info("Pre-trained Model parameters: %d", total_n_parameters)

    def zeroshot_classification_forward(
        self,
        images: torch.Tensor,
        text_prompts: Union[List[str], List[List[str]]] = None,
        prompt_encodings: torch.Tensor = None,
        is_attain_encodings: bool = False,
    ):
        """Forwarding the model for prediction."""

        assert text_prompts is not None or prompt_encodings is not None

        if self.prompt_encodings is None and prompt_encodings is None:
            self.generate_prompt_encodins(text_prompts)

        if prompt_encodings is not None:
            self.prompt_encodings = prompt_encodings

        with torch.no_grad():
            image_features = self.model.encode_image(images).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)

        probs = (100.0 * image_features @ self.prompt_encodings.T).softmax(dim=-1)

        outputs = {"probs": probs}

        if is_attain_encodings:
            outputs["text_embeds"] = self.prompt_encodings
            outputs["image_embeds"] = image_features

        return outputs

    def generate_prompt_encodins(self, text_prompts: Union[List[str], List[List[str]]]):
        """Getting the embeddings for textprompt.

        The way to obtain the embedding derives from:
        Prompt_Engineering_for_ImageNet.ipynb of https://github.com/openai/CLIP

        :param text_prompts: Two formats of prompts
            - A `List` in which each item is string of the prompt.
            - A `List[List]` in which each item is a `List` holding one group of
            prompts.

        :return prompts_encodings: A `Tensor` holding the embeddings of the input
            textual prompts
            with shape, [n_prompts, n_embeddings]
            n_prompts == len(text_prompts)
        """

        if not isinstance(text_prompts[0], list):
            text_prompts = [[prompt] for prompt in text_prompts]

        with torch.no_grad():
            prompts_encodings = []
            for prompts in text_prompts:
                text_tokens = tokenizer.tokenize(prompts)
                # get features with shape
                # [n, d] where n == len(prompts)
                text_features = self.model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                # get mean of the features
                # with shape, [d]
                prompt_features = text_features.mean(dim=0)
                prompt_features /= prompt_features.norm()
                # reshape to [1, d]
                prompt_features = prompt_features.reshape(1, -1)
                prompts_encodings.append(prompt_features)
            # assign to the prompt encodings
            # with shape, [n_prompts, n_embeddings]
            # where n_prompts == len(text_prompts)
        self.prompt_encodings = torch.cat(prompts_encodings, dim=0).to(
            Config().device()
        )
