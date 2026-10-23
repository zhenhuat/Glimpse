# Transformer components adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

import math
from functools import partial
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_

from timm.models.layers import DropPath


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DeformableBlock(nn.Module):
    def __init__(self, feature_dim_list, dim, num_heads, num_samples, qkv_bias=False, drop_path=0., mlp_ratio=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        self.num_samples = num_samples
        head_dim = dim // num_heads
        self.norm1 = norm_layer(dim)

        # Joint deformable sampling parameters.
        self.attention_weights = nn.Linear(dim, num_heads * num_samples)
        self.sampling_offsets = nn.Linear(dim, 2 * num_heads * num_samples)

        # Limb-aware sampling (Human3.6M 17-joint topology).
        self.limb_pairs = [
            (0, 1), (1, 2), (2, 3),           # right leg
            (0, 4), (4, 5), (5, 6),           # left leg
            (0, 7), (7, 8),                   # spine
            (8, 11), (11, 12), (12, 13),     # left arm
            (8, 14), (14, 15), (15, 16),     # right arm
        ]
        self.num_limb_points = 3 # number of points sampled per limb
        self.num_limbs = len(self.limb_pairs)
        self.M = self.num_limbs * self.num_limb_points

        if self.M > 0:
            # Use joint features to predict limb offsets/weights
            self.limb_offset_proj = nn.Linear(dim, 2 * num_heads * num_samples)
            self.limb_weight_proj = nn.Linear(dim, num_heads * num_samples)
            self.limb_fuse_gate = nn.Parameter(torch.ones(1, 1, 1, dim))

        self.embed_proj = nn.ModuleList([nn.Linear(dim_in, head_dim) for dim_in in feature_dim_list])
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ray_embedding = GeometricCorrection(num_joints=17, feature_dim=dim, num_scales=4)
        self.ray_fuse_gate = nn.Parameter(torch.ones(1, 1, 1, dim))

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = 0.01 * (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.num_heads, 1, 2).repeat(1, self.num_samples, 1)
        for i in range(self.num_samples):
            grid_init[:, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)

        if self.M > 0:
            constant_(self.limb_offset_proj.weight.data, 0.)
            with torch.no_grad():
                self.limb_offset_proj.bias = nn.Parameter(grid_init.view(-1).clone())
            constant_(self.limb_weight_proj.weight.data, 0.)
            constant_(self.limb_weight_proj.bias.data, 0.)

    def _get_limb_ref_points(self, ref_joints):
        # ref_joints: [B, P, 2]
        B, P, _ = ref_joints.shape
        device = ref_joints.device
        t = torch.linspace(0.2, 0.8, steps=self.num_limb_points, device=device)  # avoid endpoints
        limb_points = []
        for i1, i2 in self.limb_pairs:
            p1 = ref_joints[:, i1:i1+1]  # [B, 1, 2]
            p2 = ref_joints[:, i2:i2+1]
            pts = p1 + (p2 - p1) * t.view(1, -1, 1)  # [B, num_limb_points, 2]
            limb_points.append(pts)
        return torch.cat(limb_points, dim=1)  # [B, M, 2]

    def forward(self, x, ref, features_list):
        x_0, x = x[:, :1], x[:, 1:]  # x_0: [B,1,P,C], x: [B,L,P,C]
        b, l, p, c = x.shape

        residual = x
        x_norm = self.norm1(x + x_0)  # [B, L, P, C]

        # Joint sampling.
        weights_joint = self.attention_weights(x_norm).view(b, l, p, self.num_heads, self.num_samples)
        weights_joint = F.softmax(weights_joint, dim=-1).unsqueeze(-1)  # [B,L,P,h,s,1]
        offsets_joint = self.sampling_offsets(x_norm).reshape(b, l, p, self.num_heads * self.num_samples, 2).tanh()
        pos_joint = offsets_joint + ref.view(b, 1, p, 1, -1)  # [B,L,P,h*s,2]

        # Limb sampling.
        if self.M > 0:
            ref_limbs = self._get_limb_ref_points(ref)  # [B, M, 2]
            # For each limb point, use average of its two endpoint joint features
            limb_features = []
            for i1, i2 in self.limb_pairs:
                for _ in range(self.num_limb_points):
                    feat = (x_norm[:, :, i1] + x_norm[:, :, i2]) * 0.5  # [B, L, C]
                    limb_features.append(feat)
            limb_features = torch.stack(limb_features, dim=2)  # [B, L, M, C]

            offsets_limb = self.limb_offset_proj(limb_features).reshape(b, l, self.M, self.num_heads * self.num_samples, 2).tanh()
            weights_limb = self.limb_weight_proj(limb_features).view(b, l, self.M, self.num_heads, self.num_samples)
            weights_limb = F.softmax(weights_limb, dim=-1).unsqueeze(-1)  # [B,L,M,h,s,1]
            pos_limb = offsets_limb + ref_limbs.view(b, 1, self.M, 1, -1)  # [B,L,M,h*s,2]
        else:
            pos_limb = None
            weights_limb = None


        # === 3. Multi-scale feature sampling ===
        # Sample for joints
        features_sampled_joint = []
        for idx, features in enumerate(features_list):
            sampled = F.grid_sample(features, pos_joint[:, idx], padding_mode='border', align_corners=True)
            sampled = sampled.permute(0, 2, 3, 1).contiguous()  # [B, P, h*s, C_in]
            features_sampled_joint.append(self.embed_proj[idx](sampled))  # [B, P, h*s, head_dim]
        features_sampled_joint = torch.stack(features_sampled_joint, dim=1)  # [B, L, P, h*s, head_dim]

        # Aggregate joint features
        joint_feat = features_sampled_joint.view(b, l, p, self.num_heads, self.num_samples, -1)
        out_joint = (weights_joint * joint_feat).sum(dim=-2).view(b, l, p, -1)  # [B, L, P, C]

        # Sample and aggregate limb features (if any)
        if self.M > 0:
            features_sampled_limb = []
            for idx, features in enumerate(features_list):
                sampled = F.grid_sample(features, pos_limb[:, idx], padding_mode='border', align_corners=True)
                sampled = sampled.permute(0, 2, 3, 1).contiguous()  # [B, M, h*s, C_in]
                features_sampled_limb.append(self.embed_proj[idx](sampled))  # [B, M, h*s, head_dim]
            features_sampled_limb = torch.stack(features_sampled_limb, dim=1)  # [B, L, M, h*s, head_dim]

            limb_feat = features_sampled_limb.view(b, l, self.M, self.num_heads, self.num_samples, -1)
            out_limb = (weights_limb * limb_feat).sum(dim=-2).view(b, l, self.M, -1)  # [B, L, M, C]

            # Distribute limb features back to their endpoint joints
            limb_to_joint = torch.zeros(b, l, p, c, device=out_limb.device)
            idx = 0
            for (i1, i2) in self.limb_pairs:
                for _ in range(self.num_limb_points):
                    # equally distribute limb point feature to its two joints
                    limb_feat_point = out_limb[:, :, idx:idx+1]  # [B, L, 1, C]
                    limb_to_joint[:, :, i1] += limb_feat_point.squeeze(2)
                    limb_to_joint[:, :, i2] += limb_feat_point.squeeze(2)
                    idx += 1

            alpha = torch.sigmoid(self.limb_fuse_gate)  # ∈ (0,1)
            out_joint = (1 - alpha) * out_joint + alpha * limb_to_joint


        # === 4. Residual + MLP ===
        x = residual + self.drop_path(out_joint)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # Refine pose tokens with geometric correction.
        ray = self.ray_embedding(x).unsqueeze(1)  # [B, 1, P, C]
        beta = torch.sigmoid(self.ray_fuse_gate)
        x_0 = (1 - beta) * x_0 + beta * ray


        x = torch.cat([x_0, x], dim=1)
        return x


class GeometricCorrection(nn.Module):
    def __init__(self, num_joints=17, feature_dim=128, num_scales=4, init_weight_zero=True, sigma=1.0):
        """Initialize the learnable reference and multi-scale feature fusion."""
        super().__init__()
        self.num_joints = num_joints
        self.feature_dim = feature_dim
        self.num_scales = num_scales
        self.sigma = sigma


        self.anchor_point = nn.Parameter(torch.zeros(1, 1, 3))


        self.mlp = nn.Sequential(
            nn.Linear(3, feature_dim),
            nn.ReLU()
        )


        if init_weight_zero:
            self.weight = nn.Parameter(torch.zeros(1, num_joints, num_scales))
        else:
            self.weight = nn.Parameter(torch.randn(1, num_joints, num_scales))


        self.softmax = nn.Softmax(dim=-1)
        self.activation = nn.ReLU()


        self.ray_transform = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def interpolate_features(self, sampled_features, anchor_points, sigma=1.0):
        """Apply Gaussian weighting to features [B, S, N, C] using reference features [B, N, C]."""
        batch_size, num_scales, num_joints, feature_dim = sampled_features.shape


        interpolated_features = []
        for scale_idx in range(num_scales):

            sampled = sampled_features[:, scale_idx]  # (B, N, C)


            distances = torch.norm(sampled - anchor_points, dim=-1)  # (B, N)
            weights = torch.exp(-distances.pow(2) / (2 * sigma**2))
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)


            weighted_feature = weights.unsqueeze(-1) * sampled  # (B, N, C)
            interpolated_features.append(weighted_feature.unsqueeze(-1))

        return torch.cat(interpolated_features, dim=-1)  # (B, N, C, num_scales)

    def forward(self, sampled_features):
        """Aggregate multi-scale input features [B, S, N, C] into correction features [B, N, C]."""
        batch_size, num_scales, num_joints, feature_dim = sampled_features.shape


        anchor = self.anchor_point.expand(batch_size, num_joints, 3)  # (B, N, 3)
        ref_point = self.mlp(anchor)  # (B, N, C)


        interpolated_features = self.interpolate_features(sampled_features, ref_point, sigma=self.sigma)


        ray_representations = []
        for scale_idx in range(num_scales):

            combined_feature = torch.cat([ref_point, interpolated_features[..., scale_idx]], dim=-1)  # (B, N, 2C)
            ray_representation = self.ray_transform(combined_feature)
            ray_representation = F.normalize(ray_representation, dim=-1)
            ray_representations.append(ray_representation)


        ray_representations = torch.stack(ray_representations, dim=-1)  # (B, N, C, num_scales)


        weights = self.softmax(self.weight)  # (1, N, num_scales)
        weights = weights.expand(batch_size, -1, -1)

        aggregated_rays = torch.sum(weights.unsqueeze(2) * ray_representations, dim=-1)  # (B, N, C)

        return aggregated_rays

class PoseTransformer(nn.Module):
    def __init__(self, config=None, backbone='hrnet_32', num_joints=17, in_chans=2,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None):
        """Initialize pose embedding, geometry learning, and fusion stages."""
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        base_dim = config.base_dim
        embed_dim_ratio = config.embed_dim_ratio
        depth = config.levels
        out_dim = 3    # 3D coordinates per joint.
        self.levels = config.levels
        embed_dim = embed_dim_ratio * (self.levels+1)

        ### spatial patch embedding
        self.coord_embed = nn.Linear(in_chans, embed_dim_ratio)

        if backbone in ['hrnet_32', 'hrnet_48']:
            feature_dim_list = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]
        elif backbone == 'cpn':
            feature_dim_list = [base_dim] * 4

        self.feat_embed = nn.ModuleList([nn.Linear(dim_in, embed_dim_ratio) for dim_in in feature_dim_list])

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, 1+self.levels, num_joints, embed_dim_ratio))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.joint_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.res_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])


        self.context_blocks = nn.ModuleList([
            DeformableBlock(feature_dim_list=feature_dim_list, dim=embed_dim_ratio, num_heads=4, num_samples=4, qkv_bias=qkv_bias, drop_path=dpr[i])
            for i in range(depth)])

        self.head = nn.Sequential(nn.LayerNorm(embed_dim),nn.Linear(embed_dim, out_dim),)


    def forward(self, keypoints_2d, ref, features_list):
        b, p, c = keypoints_2d.shape
        # Input keypoints: [B, P, 2].

        x = self.coord_embed(keypoints_2d)

        features_ref_list = [
            F.grid_sample(features, ref.unsqueeze(-2), align_corners=True).squeeze(-1).permute(0, 2, 1).contiguous() \
            for features in features_list]

        features_ref_list = [embed(features_ref_list[idx]) \
                             for idx, embed in enumerate(self.feat_embed)]


        x = torch.stack([x,*features_ref_list], dim=1) # [b, 5, p, c]


        x += self.Spatial_pos_embed
        x = self.pos_drop(x)

        for blk in self.context_blocks:
            x = blk(x, ref, features_list)


        x = rearrange(x, 'b l p c -> (b p) l c')

        for blk in self.res_blocks:
            x = blk(x)
        x = rearrange(x, '(b p) l c -> b p (l c)', b=b)


        for blk in self.joint_blocks:
            x = blk(x)


        x = self.head(x).view(b, 1, p, -1)


        return x

