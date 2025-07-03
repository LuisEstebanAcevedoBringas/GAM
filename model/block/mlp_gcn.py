from functools import partial
from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_
from dropblock import DropBlock2D
import math


class AdaptiveAdjacency(nn.Module):
    def __init__(self, A_init, learn_eps=True):
        super().__init__()
        self.A = nn.Parameter(A_init.clone())
        self.learn_eps = learn_eps
        if learn_eps:
            self.eps = nn.Parameter(torch.zeros(1))

    def forward(self):
        A_softmax = torch.softmax(self.A, dim=-1)
        if self.learn_eps:
            return (1 - self.eps) * self.A + self.eps * A_softmax
        return A_softmax


######################################

class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1, use_se=False, se_reduction=4):
        super().__init__()
        self.use_se = use_se
        t = int(abs((math.log2(channels) + b) / gamma))
        k_size = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

        if self.use_se:
            self.se_fc = nn.Sequential(
                nn.Linear(channels, channels // se_reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channels // se_reduction, channels, bias=False),
                nn.Sigmoid()
            )

    def forward(self, x):  # [B, C, J]
        y = self.avg_pool(x)       # [B, C, 1]
        y = y.permute(0, 2, 1)     # [B, 1, C]
        y = self.conv(y)           # [B, 1, C]
        y = y.permute(0, 2, 1)     # [B, C, 1]
        eca_weight = self.sigmoid(y)

        out = x * eca_weight       # channel-wise attention

        if self.use_se:
            se_weight = self.se_fc(x.mean(-1))  # [B, C]
            se_weight = se_weight.unsqueeze(-1)  # [B, C, 1]
            out = out * se_weight

        return out           # [B, C, J] * [B, C, 1] — broadcasted


class Gcn(nn.Module):
    def __init__(self, in_channels, out_channels, adj, drop_prob=0.1, use_residual=True):
        super().__init__()
        self.kernel_size = adj.size(0)
        self.use_residual = use_residual and in_channels == out_channels

        self.adaptive_adj = AdaptiveAdjacency(adj)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels * self.kernel_size, kernel_size=1)
        self.dropblock = DropBlock2D(block_size=3, drop_prob=drop_prob)

        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x if self.use_residual else None

        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.dropblock(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        adj = self.adaptive_adj()  # Shape: [K, V, V]

        # Graph convolution
        x = torch.einsum('nkctv,kvw->nctw', x, adj)  # [N, C, T, V]

        x = self.norm(x)
        x = self.act(x)

        if res is not None:
            x = x + res

        return x.contiguous()


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                    act_layer=nn.GELU, drop=0., reduction=4,
                    use_eca=True, use_dropblock=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or (in_features // reduction)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.use_eca = use_eca
        self.use_dropblock = use_dropblock
        if use_dropblock:
            self.dropblock = DropBlock2D(block_size=3, drop_prob=drop)
        if use_eca:
            self.eca = ECABlock(out_features)

    def forward(self, x):  # x: [B, J, C]
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)

        if self.use_dropblock:
            x = self.dropblock(x.unsqueeze(-1).transpose(1, 2)).squeeze(-1).transpose(1, 2)

        if self.use_eca:
            x = self.eca(x.transpose(1, 2)).transpose(1, 2)  # Correct shape for ECABlock

        return x


class Mlp_ln(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                    act_layer=nn.GELU, drop=0., reduction=4,
                    use_eca=True, use_dropblock=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or (in_features // reduction)

        self.fc1 = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features)
        )
        self.act = act_layer()
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_features, out_features),
            nn.LayerNorm(out_features)
        )
        self.drop = nn.Dropout(drop)

        self.use_eca = use_eca
        self.use_dropblock = use_dropblock
        if use_dropblock:
            self.dropblock = DropBlock2D(block_size=3, drop_prob=drop)
        if use_eca:
            self.eca = ECABlock(out_features)

    def forward(self, x):  # x: [B, J, C]
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)

        if self.use_dropblock:
            x = self.dropblock(x.unsqueeze(-1).transpose(1, 2)).squeeze(-1).transpose(1, 2)

        if self.use_eca:
            x = self.eca(x.transpose(1, 2)).transpose(1, 2)  # Correct shape for ECABlock

        return x



class Block(nn.Module):
    def __init__(self, length, frames, dim, tokens_dim, channels_dim, adj, drop=0.,
                    drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, reduction=4):
        super().__init__()
        self.norm1 = norm_layer(length)
        self.norm2 = norm_layer(dim)

        self.gcn_1 = Gcn(dim, dim, adj, drop_prob=drop)
        self.gcn_2 = Gcn(dim, dim, adj, drop_prob=drop)

        self.alpha1 = nn.Parameter(torch.ones(1))  # residual scaling
        self.alpha2 = nn.Parameter(torch.ones(1))

        if frames == 1:
            self.mlp_1 = Mlp(in_features=length, hidden_features=tokens_dim, act_layer=act_layer, drop=drop, reduction=reduction)
        else:
            self.mlp_1 = Mlp_ln(in_features=length, hidden_features=tokens_dim, act_layer=act_layer, drop=drop, reduction=reduction)

        self.mlp_2 = Mlp(in_features=dim, hidden_features=channels_dim, act_layer=act_layer, drop=drop, reduction=reduction)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        ## Spatial Graph MLP
        x = rearrange(x, 'b j c -> b c j') 
        res = x
        x = self.norm1(x)
        x_gcn_1 = rearrange(x, 'b c j -> b c 1 j') 
        x_gcn_1 = self.gcn_1(x_gcn_1)
        x_gcn_1 = rearrange(x_gcn_1, 'b c 1 j -> b c j') 
        x = res + self.alpha1 * self.drop_path(self.mlp_1(x) + x_gcn_1)

        ## Channel Graph MLP
        x = rearrange(x, 'b c j -> b j c') 
        res = x
        x = self.norm2(x)
        x_gcn_2 = rearrange(x, 'b j c -> b c 1 j') 
        x_gcn_2 = self.gcn_2(x_gcn_2)
        x_gcn_2 = rearrange(x_gcn_2, 'b c 1 j -> b j c') 
        x = res + self.alpha2 * self.drop_path(self.mlp_2(x) + x_gcn_2)

        return x


class Mlp_gcn(nn.Module):
    def __init__(self, depth, embed_dim, channels_dim, tokens_dim, adj, drop_rate=0.10, length=17, frames=1):
        super().__init__()
        drop_path_rate = 0.2

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  

        self.blocks = nn.ModuleList([
            Block(
                length, frames, embed_dim, tokens_dim, channels_dim, adj, 
                drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x