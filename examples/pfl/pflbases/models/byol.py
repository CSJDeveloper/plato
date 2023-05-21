
"""
Implementation of the BYOL [1] method under Plato.

[1]. Jean-Bastien Grill, et.al, Bootstrap Your Own Latent A New Approach to Self-Supervised Learning.
https://arxiv.org/pdf/2006.07733.pdf.

Source code: https://github.com/lucidrains/byol-pytorch
The third-party code: https://github.com/sthalles/PyTorch-BYOL

"""

import copy

from torch import nn
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad


from plato.models.cnn_encoder import Model as encoder_registry
from plato.config import Config

class BYOL(nn.Module):
    def __init__(self, encoder=None, encoder_dim=None):
        super().__init__()

        # define the encoder
        encoder_name = Config().trainer.encoder_name
        encoder_params = (
            Config().params.encoder if hasattr(Config().params, "encoder") else {}
        )

        # define the encoder
        self.encoder = (
            encoder
            if encoder is not None
            else encoder_registry.get(model_name=encoder_name, **encoder_params)
        )

        self.encoding_dim = self.encoder.encoding_dim
        projection_hidden_dim = Config().trainer.projection_hidden_dim
        projection_out_dim = Config().trainer.projection_out_dim
        prediction_hidden_dim = Config().trainer.prediction_hidden_dim
        prediction_out_dim = Config().trainer.prediction_out_dim

        self.projection_head = BYOLProjectionHead(
            self.encoding_dim, projection_hidden_dim, projection_out_dim
        )
        self.prediction_head = BYOLPredictionHead(
            projection_out_dim, prediction_hidden_dim, prediction_out_dim
        )

        self.encoder_momentum = copy.deepcopy(self.encoder)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.encoder_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward_direct(self, samples):
        encoded_examples = self.encoder(samples).flatten(start_dim=1)
        projected_examples = self.projection_head(encoded_examples)
        output = self.prediction_head(projected_examples)
        return output

    def forward_momentum(self, samples):
        encoded_examples = self.encoder_momentum(samples).flatten(start_dim=1)
        projected_examples = self.projection_head_momentum(encoded_examples)
        projected_examples = projected_examples.detach()
        return projected_examples

    def forward(self, multiview_samples):
        samples1, samples2 = multiview_samples
        output1 = self.forward_direct(samples1)
        projected_samples1 = self.forward_momentum(samples1)
        output2 = self.forward_direct(samples2)
        projected_samples2 = self.forward_momentum(samples2)
        return (output1, projected_samples2), (output2, projected_samples1)