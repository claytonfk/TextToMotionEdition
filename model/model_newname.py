# Code partially borrowed from MDM: https://github.com/GuyTevet/motion-diffusion-model

import clip
import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module):
    def __init__(self, njoints, nfeats, latent_dim=256, ff_size=1024, num_layers=8, 
                 num_heads=4, dropout=0.1, activation='gelu', device = 'cuda', 
                 data_rep = 'hml_vec', cond_mask_prob = 0.1):
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
        
        self.cond_mask_prob = cond_mask_prob
        
        self.input_feats = self.njoints * self.nfeats
        
        # Load frozen clip model
        self.clip_model = self.clip_load()
        
        # Text embedding
        self.embed_text = nn.Linear(512, self.latent_dim)
        
        # Positional encoding and timestep embedder
        self.pe = PositionalEncoding(self.latent_dim, self.dropout)
        self.te = TimestepEmbedder(self.latent_dim, self.pe)
        
        # Transformer
        t_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                             nhead=self.num_heads,
                                             dim_feedforward=self.ff_size,
                                             dropout=self.dropout,
                                             activation=self.activation)
 
        self.transformer = nn.TransformerEncoder(t_layer, num_layers=self.num_layers)
        
        # Pre-process and post-process
        self.input_process = InputProcess(self.input_feats, self.latent_dim)
            
    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]
        
    def clip_load(self, model_name = "ViT-B/32"):
        # Loading frozen clip model
        clip_model, _ = clip.load(model_name, device=self.device, jit=False)  # Must set jit=False for training
    
        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
            

    def mask_cond(self, cond):
        bs, d = cond.shape
        if self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond
        
        
    def forward(self, x, t, y = None):
        """
        x: [batch_size, njoints, nfeats, max_frames]
        timesteps: [batch_size] (int)
        y: list of strings corresponding to prompts
        

        
        """
        
        bs, njoints, nfeats, nframes = x.shape
        
        x = self.input_process(x)
        
        emb = self.te(t)  # [1, batch_size, latent_dim]

        # Process strings
        text = clip.tokenize(y['text'], truncate=True).to(self.device)
        text = self.clip_model.encode_text(text).float()
        text = self.mask_cond(text)
        text = self.embed_text(text)
        emb += text
        
        
        # Transformer
        
        xseq = torch.cat((emb, x), axis=0) 
        xseq = self.pe(xseq) 
        
        output = self.transformer(xseq)[1:]
        
        
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
    def __init__(self, latent_dim, pe):
        super().__init__()
        self.latent_dim = latent_dim
        self.pe = pe

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.pe.pe[timesteps]).permute(1, 0, 2)



class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.pose_emb = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)
        x = self.pose_emb(x)
        
        return x
