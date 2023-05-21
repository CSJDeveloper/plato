
"""
implementation of the MoCoV2 [2] method, which is the enhanced version of MoCoV1 [1].

[1]. Kaiming He, et.al., Momentum Contrast for Unsupervised Visual Representation Learning, 
CVPR 2020. https://arxiv.org/abs/1911.05722.

[2]. Xinlei Chen, et.al, Improved Baselines with Momentum Contrastive Learning, ArXiv, 2020.
https://arxiv.org/abs/2003.04297.

The official code: https://github.com/facebookresearch/moco

"""


import copy

from torch import nn
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule

from plato.models.cnn_encoder import Model as encoder_registry
from plato.config import Config


class MoCoV2(nn.Module):
    def __init__(self, encoder=None):
        super().__init__()

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

        # define heads
        projection_hidden_dim = Config().trainer.projection_hidden_dim
        projection_out_dim = Config().trainer.projection_out_dim
        self.projection_head = MoCoProjectionHead(
            self.encoding_dim, projection_hidden_dim, projection_out_dim
        )

        self.encoder_momentum = copy.deepcopy(self.encoder)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.encoder_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward_direct(self, samples):
        query = self.encoder(samples).flatten(start_dim=1)
        query = self.projection_head(query)
        return query

    def forward_momentum(self, samples):
        key = self.encoder_momentum(samples).flatten(start_dim=1)
        key = self.projection_head_momentum(key).detach()
        return key

    def forward(self, multiview_samples):
        query_samples, key_samples = multiview_samples
        query = self.forward_direct(query_samples)
        key = self.forward_momentum(key_samples)

        return query, key

