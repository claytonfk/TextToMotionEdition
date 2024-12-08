# Code partially borrowed from MDM: https://github.com/GuyTevet/motion-diffusion-model

import clip
import torch
import torch.nn as nn
import numpy as np

from model.rotation2xyz import Rotation2xyz

class Model(nn.Module):
    def __init__(self, njoints, nfeats, latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, 
                 dropout=0.1, activation='gelu', device='cuda', dataset='humanml',
                 data_rep='hml_vec', cond_mask_prob=0.1):
        super().__init__()
        
        
        self.njoints = njoints
        self.nfeats = nfeats
        self.device = device
        # Transformer hyper-parameters
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.dataset = dataset
        self.translation = True
        self.is_ours = False
        
        self.cond_mask_prob = cond_mask_prob
        self.data_rep = data_rep
        self.input_feats = self.njoints * self.nfeats
        
        # Load frozen clip model
        self.clip_model = self.clip_load()
        # Text embedding
        self.embed_text = nn.Linear(512, self.latent_dim)
        
        # Positional encoding and timestep embedder
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        # Transformer
        t_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                             nhead=self.num_heads,
                                             dim_feedforward=self.ff_size,
                                             dropout=self.dropout,
                                             activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(t_layer, num_layers=self.num_layers)
  
        # Pre-process and post-process
        self.input_process = InputProcess(self.input_feats, self.latent_dim)
        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints, self.nfeats)
        self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)
            
    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]
    
    def set_is_ours(self, cond):
        self.is_ours = True
        
    def clip_load(self, model_name="ViT-B/32"):
        # Loading frozen clip model
        clip_model, _ = clip.load(model_name, device=self.device, jit=False)  # Must set jit=False for training

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
            
        return clip_model
            

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond
        
    def get_embbed_text(self, text):
        text = clip.tokenize(text, truncate=True).to(self.device)
        text = self.clip_model.encode_text(text).float()
        emb = self.embed_text(text)
        
        return emb
    
    
    def forward(self, x, t, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames]
        timesteps: [batch_size] (int)
        y: list of strings corresponding to prompts
        """
        
        bs, njoints, nfeats, nframes = x.shape
        x = self.input_process(x)
        emb = self.embed_timestep(t)  # [1, batch_size, latent_dim]
        force_mask = y.get('uncond', False)

        # Process strings
        if not self.is_ours:
            text = clip.tokenize(y['text'], truncate=True).to(self.device)
            text = self.clip_model.encode_text(text).float()
            #text = self.mask_cond(text, force_mask=force_mask)
            text = self.embed_text(text)
        else:
            text = y['emb']

        emb += text
        # Transformer
        xseq = torch.cat((emb, x), axis=0)
        xseq = self.sequence_pos_encoder(xseq)
        for layer_idx, layer in enumerate(self.seqTransEncoder.layers):
            if 'soft_tokens' in y:
                st = y['soft_tokens'][layer_idx, ...].repeat(1, bs, 1)
                xseq = torch.cat((xseq, st), axis=0)
                # print(xseq.shape)
                # print(st.shape)

            xseq = layer(xseq, src_mask=None, src_key_padding_mask=None)
            if 'soft_tokens' in y:
                xseq = xseq[:-1*st.shape[0]]

        xseq = xseq[1:]
        output = self.output_process(xseq)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)



class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)
        x = self.poseEmbedding(x)
        
        return x

class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        output = self.poseFinal(output)  # [seqlen, bs, 150]
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output