import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

# A wrapper model for Classifier-free guidance **SAMPLING** only
# Code borrowed from MDM: https://github.com/GuyTevet/motion-diffusion-model

# https://arxiv.org/abs/2207.12598
class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model  # model is the actual model to run

        assert self.model.cond_mask_prob > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions'

        # pointers to inner model
        self.rot2xyz = self.model.rot2xyz
        self.num_layers = self.model.num_layers
        self.latent_dim = self.model.latent_dim
        self.translation = self.model.translation
        self.njoints = self.model.njoints
        self.nfeats = self.model.nfeats
        self.data_rep = self.model.data_rep
        self.is_ours = self.model.is_ours
        
    def get_embbed_text(self, text):
        return self.model.get_embbed_text(text)
    
    def set_is_ours(self, cond):
        self.model.set_is_ours(cond)
        
    def forward(self, x, timesteps, y=None):
        # y_uncond = deepcopy(y)
        y['uncond'] = True
        out = self.model(x, timesteps, y)
        out_uncond = self.model(x, timesteps, y)
        y['uncond'] = False
        return out_uncond + (y['scale'].view(-1, 1, 1, 1) * (out - out_uncond))

