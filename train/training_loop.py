# Code partially borrowed from MDM: https://github.com/GuyTevet/motion-diffusion-model

# This implements our method inspired by Imagic: https://imagic-editing.github.io/

import torch
from torch.optim import AdamW
import numpy as np
from diffusion.resample import create_named_schedule_sampler
from torch.autograd import Variable
from data_loaders.tensors import collate
from tqdm import tqdm
import torch.nn.functional as F
# JOINT IDX TABLE
# 00: Pelvis
# 01: L_Hip
# 02: R_Hip
# 03: Spine1
# 04: L_Knee
# 05: R_Knee
# 06: Spine2
# 07: L_Ankle
# 08: R_Ankle
# 09: Spine3
# 10: L_Foot
# 11: R_Foot
# 12: Neck
# 13: L_Collar
# 14: R_Collar
# 15: Head
# 16: L_Shoulder
# 17: R_Shoulder
# 18: L_Elbow
# 19: R_Elbow
# 20: L_Wrist
# 21: R_Wrist
class TrainLoopOurs:
    def __init__(self, num_steps, n_frames, batch_size, guidance_param, lr, edit_motion, edit_text, base_motion, base_text,
                 model, diffusion, train_embedding=True, pre_embedding=None, train_soft_token=False, pre_soft_tokens=None,
                 soft_token_len=4, prob_base_training=0.1, device='cuda', begin_frame=None, end_frame=None, main_frame=None,
                 pose_insertion_frame=None, frames_before_disregard=None, frames_after_disregard=None,  insertion_weight=None):

        self.model = model
        self.diffusion = diffusion
        self.num_steps = num_steps
        self.n_frames = n_frames
        self.batch_size = batch_size
        self.guidance_param = guidance_param
        self.lr = lr
        self.edit_text = edit_text
        self.base_text = base_text
        self.device = device
        self.soft_tokens = pre_soft_tokens
        self.train_embedding = train_embedding
        self.train_soft_token = train_soft_token
        self.prob_base_training = prob_base_training
        self.begin_frame = begin_frame
        self.end_frame = end_frame
        self.main_frame = main_frame
        self.motion = edit_motion.repeat(batch_size, 1, 1, 1)
        self.base_motion = base_motion.repeat(batch_size, 1, 1, 1)
        self.pose_insertion_frame = pose_insertion_frame
        self.frames_before_disregard = frames_before_disregard
        self.frames_after_disregard = frames_after_disregard
        self.insertion_weight = insertion_weight

        self.motion = self.motion.to(self.device)
        self.base_motion = self.base_motion.to(self.device)

        if train_embedding:
            if pre_embedding is not None:
                self.edit_embedding = Variable(pre_embedding, requires_grad=True)
            else:
                self.edit_embedding = self.model.get_embbed_text(self.edit_text)
                self.edit_embedding = Variable(self.edit_embedding, requires_grad=True)
            self.opt = AdamW([self.edit_embedding], lr=self.lr)
        elif train_soft_token:
            self.soft_tokens = Variable(torch.zeros(self.model.num_layers, soft_token_len, 1, self.model.latent_dim, device=self.device), requires_grad=True)
            self.soft_tokens = torch.nn.init.xavier_uniform_(self.soft_tokens, gain=1.0)
            self.opt = AdamW([self.soft_tokens], lr=self.lr)
            self.edit_embedding = pre_embedding.requires_grad_(False)
        else:
            self.edit_embedding = pre_embedding.requires_grad_(False)
            if self.guidance_param != 1:
                self.opt = AdamW(list(self.model.model.parameters()), lr=self.lr)
            else:
                self.opt = AdamW(list(self.model.parameters()), lr=self.lr)

        self.base_embedding = self.model.get_embbed_text(self.base_text).detach()
        self.base_embedding = self.base_embedding.to(self.device)
        self.edit_embedding = self.edit_embedding.to(self.device)
        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)

    def run_loop(self):
        self.model.set_is_ours(True)
        pbar = tqdm(range(self.num_steps), desc='Training')
        for step in pbar:
            # aug_motion_scale = np.random.uniform(low=0.9, high=1.1)
            # aug_motion = F.interpolate(self.motion.squeeze(2), scale_factor=aug_motion_scale) # Shape will be [bs, 263, timesteps]
            aug_motion = self.motion
            aug_n_frames = aug_motion.shape[-1]

            texts = ['']
            collate_args = [{'inp': torch.zeros(aug_n_frames), 'tokens': None, 'lengths': aug_n_frames}]
            collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
            _, cond = collate(collate_args)

            # Weights for video-extracted motions
            # z_indices = []
            # joint_exclude = [] # [0, 3, 6, 9]
            # joint_indices = [i for i in range(0, 21) if i not in joint_exclude]
            # for jidx in joint_indices:
            #     jidx_ = [67 + jidx*6 + i for i in range(0, 6)]
            #     z_indices += jidx_

            z_indices = [i for i in range(67, 193)]
            if self.begin_frame is not None and self.end_frame is not None:
                # LOCAL VIDEO
                cond['y']['weight'] = torch.ones(aug_motion.shape, device=self.device)
                cond['y']['weight'][:, z_indices, :, self.begin_frame-5:self.begin_frame+5] = 0
                cond['y']['weight'][:, z_indices, :, self.end_frame - 5:self.end_frame + 5] = 0
                if self.main_frame is not None:
                    cond['y']['weight'][:, z_indices, :, self.begin_frame+self.main_frame] = 10
            elif self.pose_insertion_frame is not None:
                # POSE GLOBAL
                cond['y']['weight'] = torch.zeros(aug_motion.shape, device=self.device)
                cond['y']['weight'][:, z_indices, :, :] = 1
                if self.pose_insertion_frame != -1:
                    # POSE LOCAL
                    cond['y']['weight'] = torch.ones(aug_motion.shape, device=self.device)
                    cond['y']['weight'][:, :, :, self.pose_insertion_frame - self.frames_before_disregard:self.pose_insertion_frame+self.frames_after_disregard] = 0
                    cond['y']['weight'][:, z_indices, :, self.pose_insertion_frame] = self.insertion_weight
            else:
                # GLOBAL VIDEO
                cond['y']['weight'] = torch.zeros(aug_motion.shape, device=self.device)
                cond['y']['weight'][:, z_indices, :, :] = 1

            cond['y']['emb'] = self.edit_embedding
            if self.guidance_param != 1:
                cond['y']['scale'] = torch.ones(self.batch_size, device=self.device) * self.guidance_param
            cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}
            if self.train_soft_token or self.soft_tokens is not None:
                cond['y'][f'soft_tokens'] = self.soft_tokens
            self.run_step(aug_motion, cond)

            if not self.train_embedding and np.random.rand() < self.prob_base_training:
                base_n_frames = self.base_motion.shape[-1]
                collate_args = [{'inp': torch.zeros(base_n_frames), 'tokens': None, 'lengths': base_n_frames}]
                collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, [''])]
                _, cond = collate(collate_args)
                cond['y']['weight'] = torch.ones(self.base_motion.shape, device=self.device)
                cond['y']['emb'] = self.base_embedding
                if self.guidance_param != 1:
                    cond['y']['scale'] = torch.ones(self.batch_size, device=self.device) * self.guidance_param
                cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in
                             cond['y'].items()}
                if self.train_soft_token or self.soft_tokens is not None:
                    cond['y'][f'soft_tokens'] = self.soft_tokens
                self.run_step(self.base_motion, cond)

        self.model.set_is_ours(False)
    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.opt.step()
    def forward_backward(self, batch, cond):
        self.opt.zero_grad()
        t, weights = self.schedule_sampler.sample(self.batch_size, self.device)
        losses = self.diffusion.training_losses(self.model, batch, t, model_kwargs=cond)
        weights = weights.view(-1, 1, 1, 1)
        loss = (losses["loss"] * weights).mean()
        loss.backward(retain_graph=True)