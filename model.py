import torch
import torch.nn as nn


class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        c1 = out_channels // 4
        c2 = out_channels // 4
        c3 = out_channels - c1 - c2
        self.branch1x1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        self.branch3x3 = nn.Conv2d(in_channels, c2, kernel_size=3, padding=1)
        self.branch5x5 = nn.Conv2d(in_channels, c3, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y1 = self.branch1x1(x)
        y2 = self.branch3x3(x)
        y3 = self.branch5x5(x)
        out = torch.cat((y1, y2, y3), dim=1)
        return self.relu(self.bn(out))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        hidden = max(in_channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.avg_pool(x).view(b, c)
        w = self.mlp(w).view(b, c, 1, 1)
        return x * self.sigmoid(w)


class NpyBranch(nn.Module):
    def __init__(self, in_channels=30):
        super().__init__()
        self.multi_scale = MultiScaleConv(in_channels, out_channels=64)
        self.res1 = ResidualBlock(64, 128, stride=2)
        self.attn1 = ChannelAttention(128, reduction=8)
        self.res2 = ResidualBlock(128, 256, stride=2)
        self.attn2 = ChannelAttention(256, reduction=8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.multi_scale(x)
        x = self.res1(x)
        x = self.attn1(x)
        x = self.res2(x)
        x = self.attn2(x)
        return self.flatten(self.pool(x))


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=16, patch_size=4, in_channels=3, embed_dim=64):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        b = x.size(0)
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        return torch.cat((cls_tokens, x), dim=1) + self.pos_embed


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        normalized = self.norm1(x)
        x = x + self.attn(normalized, normalized, normalized)[0]
        return x + self.mlp(self.norm2(x))


class PngBranch(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size=16, patch_size=4, in_channels=3, embed_dim=embed_dim)
        self.encoder = nn.Sequential(*[
            TransformerEncoder(embed_dim, num_heads=num_heads, mlp_ratio=4.0, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.png_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.encoder(self.patch_embed(x))
        return self.png_head(x[:, 0])


class DualInputModel(nn.Module):
    def __init__(self, num_classes=4, npy_in_channels=30):
        super().__init__()
        self.npy_branch = NpyBranch(in_channels=npy_in_channels)
        self.png_branch = PngBranch()
        self.fusion = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, npy_input, png_input):
        npy_feat = self.npy_branch(npy_input)
        png_feat = self.png_branch(png_input)
        return self.fusion(torch.cat((npy_feat, png_feat), dim=1))
