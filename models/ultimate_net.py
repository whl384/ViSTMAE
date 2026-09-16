import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(MODELS_DIR)
DINOV2_PATH = os.path.join(PROJECT_DIR, "dinov2-main")

if DINOV2_PATH not in sys.path:
    sys.path.insert(0, DINOV2_PATH)


class SpatialMAEForCrossChannel(nn.Module):
    def __init__(self, ts_channels=25, ts_length=224, patch_size=16, embed_dim=128, depth=4):
        super().__init__()
        self.ts_channels = ts_channels
        self.ts_length = ts_length
        self.patch_size = patch_size
        self.num_patches = ts_length // patch_size

        self.spatial_embed = nn.Linear(patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max(128, ts_channels), embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads=4, mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, patch_size)

        nn.init.normal_(self.pos_embed, std=.02)
        nn.init.normal_(self.mask_token, std=.02)

    def forward(self, x_time, mask_ratio=0.40, is_training=True):
        B, C, L = x_time.shape
        x_patches = x_time.reshape(B, C, self.num_patches, self.patch_size)
        x_spatial_input = x_patches.permute(0, 2, 1, 3).reshape(B * self.num_patches, C, self.patch_size)

        h = self.spatial_embed(x_spatial_input) + self.pos_embed[:, :C, :]
        if is_training and mask_ratio > 0:
            Total_Tokens = C
            len_keep = max(1, int(Total_Tokens * (1 - mask_ratio)))
            noise = torch.rand(h.shape[0], Total_Tokens, device=x_time.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            ids_keep = ids_shuffle[:, :len_keep]
            
            h_masked = torch.gather(h, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, h.shape[-1]))
            mask_tokens = self.mask_token.repeat(h_masked.shape[0], Total_Tokens - len_keep, 1)
            h_full = torch.cat([h_masked, mask_tokens], dim=1)
            h = torch.gather(h_full, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, h.shape[-1]))

        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)

        pred_patches_flat = self.decoder_pred(h)
        pred_channels = pred_patches_flat.reshape(B, self.num_patches, C, self.patch_size).permute(0, 2, 1, 3).reshape(B, C, L)
        loss = F.mse_loss(pred_channels, x_time) if is_training else torch.tensor(0.0, device=x_time.device)
        return pred_channels, loss, h.mean(dim=1)


class TimeMAEFor1DSequence(nn.Module):
    def __init__(self, ts_channels=25, ts_length=224, patch_len=8, embed_dim=128, depth=4):
        super().__init__()
        self.ts_channels = ts_channels
        self.ts_length = ts_length
        self.patch_len = patch_len
        self.num_patches = ts_length // patch_len

        self.patch_embed = nn.Linear(patch_len * ts_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads=4, mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, patch_len * ts_channels)

    def sequence_to_patches(self, x):
        B, C, L = x.shape
        x = x.reshape(B, C, self.num_patches, self.patch_len)
        return x.permute(0, 2, 1, 3).reshape(B, self.num_patches, self.patch_len * C)

    def patches_to_sequence(self, patches):
        B, P, D = patches.shape
        x = patches.reshape(B, P, self.ts_channels, self.patch_len)
        return x.permute(0, 2, 1, 3).reshape(B, self.ts_channels, self.ts_length)

    def forward(self, x_time, mask_ratio=0.75, is_training=True):
        patches = self.sequence_to_patches(x_time)
        x = self.patch_embed(patches) + self.pos_embed[:, 1:, :]

        ids_restore = None
        len_keep = 0
        if is_training and mask_ratio > 0:
            N, L, D = x.shape
            len_keep = max(1, int(L * (1 - mask_ratio)))
            
            with torch.no_grad():
                B_p, P_p, D_p = patches.shape
                patches_view = patches.reshape(B_p, P_p, self.ts_channels, self.patch_len)
                d1 = torch.abs(patches_view[:, :, :, 1:] - patches_view[:, :, :, :-1])
                d2 = torch.abs(d1[:, :, :, 1:] - d1[:, :, :, :-1])
                patch_suspicion = d1.sum(dim=[-2, -1]) + 0.5 * F.pad(d2, (0, 1)).sum(dim=[-2, -1])
                ids_shuffle = torch.argsort(patch_suspicion, dim=1)
                ids_restore = torch.argsort(ids_shuffle, dim=1)
                ids_keep = ids_shuffle[:, :len_keep]

            x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        pred_patches = self.decoder_pred(x[:, 1:, :])
        if is_training and mask_ratio > 0:
            mask_tokens = torch.zeros(pred_patches.shape[0], self.num_patches - len_keep, pred_patches.shape[2], device=x.device)
            x_ = torch.cat([pred_patches, mask_tokens], dim=1)
            pred_patches = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2]))

        x_time_purified = self.patches_to_sequence(pred_patches)
        loss = (pred_patches - patches) ** 2 if is_training else torch.tensor(0.0, device=x_time.device)
        return x_time_purified, loss.mean()


def load_FROZEN_dinov2_backbone(device):
    from dinov2.hub.backbones import dinov2_vits14
    dinov2_model = dinov2_vits14(pretrained=True)
    dinov2_model.to(device)
    dinov2_model.eval()
    for param in dinov2_model.parameters():
        param.requires_grad = False
    return dinov2_model


class UltimateDinoTFMAE(nn.Module):
    def __init__(self, ts_channels=25, ts_length=224, device="cuda"):
        super().__init__()
        self.ts_channels = ts_channels
        self.ts_length = ts_length

        self.temporal_mae = TimeMAEFor1DSequence(ts_channels, ts_length)
        self.spatial_mae = SpatialMAEForCrossChannel(ts_channels, ts_length)

        self.dinov2_extractor = load_FROZEN_dinov2_backbone(device)
        self.dino_to_ts = nn.Sequential(
            nn.Linear(384, 128),
            nn.GELU(),
            nn.Linear(128, ts_channels * ts_length)
        )

    def train(self, mode=True):
        super().train(mode)
        self.dinov2_extractor.eval()
        return self

    def project_to_dinov2_space(self, char_x):
        return F.interpolate(
            torch.tanh(char_x / 3.0).unsqueeze(1), 
            size=(518, 518), 
            mode='bilinear', 
            align_corners=False
        ).repeat(1, 3, 1, 1)

    def compute_symmetric_kld(self, p, f, temperature=1.0):
        p_clamped = torch.clamp(p.float() / temperature, -20.0, 20.0)
        f_clamped = torch.clamp(f.float() / temperature, -20.0, 20.0)
        
        p_prob = torch.softmax(p_clamped, dim=-1)
        f_prob = torch.softmax(f_clamped, dim=-1)

        kl_p_f = torch.sum(p_prob * (torch.log(p_prob + 1e-5) - torch.log(f_prob + 1e-5)), dim=-1)
        kl_f_p = torch.sum(f_prob * (torch.log(f_prob + 1e-5) - torch.log(p_prob + 1e-5)), dim=-1)
        
        return (kl_p_f + kl_f_p).mean()