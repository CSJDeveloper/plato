"""
Implementation of the Dino [1] method under Plato.

[1]. Jean-Bastien Grill, et.al, Bootstrap Your Own Latent A New Approach to Self-Supervised Learning.
https://arxiv.org/pdf/2006.07733.pdf.

Source code: https://github.com/lucidrains/byol-pytorch
The third-party code: https://github.com/sthalles/PyTorch-BYOL
"""
import copy

from torch import nn
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad

from plato.models.cnn_encoder import Model as encoder_registry
from plato.config import Config


class DINO(nn.Module):
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
        projection_bottleneck_dim = Config().trainer.projection_bottleneck_dim
        projection_out_dim = Config().trainer.projection_out_dim

        self.projection_head = DINOProjectionHead(
            self.encoding_dim,
            projection_hidden_dim,
            projection_bottleneck_dim,
            projection_out_dim,
        )
        # Detach the weights from the computation graph
        self.projection_head.last_layer.weight_g.detach_()
        self.projection_head.last_layer.weight_g.requires_grad = False

        self.teacher_encoder = copy.deepcopy(self.encoder)
        self.teacher_head = DINOProjectionHead(
            self.encoding_dim,
            projection_hidden_dim,
            projection_bottleneck_dim,
            projection_out_dim,
        )
        self.teacher_head.last_layer.weight_g.detach_()
        self.teacher_head.last_layer.weight_g.requires_grad = False

        deactivate_requires_grad(self.teacher_encoder)
        deactivate_requires_grad(self.teacher_head)

    def forward_student(self, samples):
        encoded_examples = self.encoder(samples).flatten(start_dim=1)
        projected_examples = self.projection_head(encoded_examples)
        return projected_examples

    def forward_teacher(self, samples):
        encoded_examples = self.teacher_encoder(samples).flatten(start_dim=1)
        projected_examples = self.teacher_head(encoded_examples)
        return projected_examples

    def forward(self, multiview_samples):
        global_views = multiview_samples[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward_student(view) for view in global_views]
        return teacher_out, student_out
