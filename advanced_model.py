"""
Advanced architectures with multi-modal support, attention mechanisms, and language understanding.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class TextEncoder(nn.Module):
    """Encode text using a pre-trained transformer (DistilBERT)."""
    
    def __init__(self, model_name="distilbert-base-uncased", hidden_dim=256):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        for param in self.bert.parameters():
            param.requires_grad = False  # Freeze transformer
        
    def forward(self, text_list):
        """
        Args:
            text_list: List of strings or single string.
        Returns:
            Tensor of shape (batch_size, hidden_dim)
        """
        if isinstance(text_list, str):
            text_list = [text_list]
        
        tokens = self.tokenizer(text_list, return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(self.bert.device) for k, v in tokens.items()}
        
        with torch.no_grad():
            output = self.bert(**tokens)
        
        # Use [CLS] token (first token) as sequence representation
        cls_embedding = output.last_hidden_state[:, 0, :]
        return self.projection(cls_embedding)


class SelfAttention(nn.Module):
    """Self-attention block."""
    
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)
        
    def forward(self, x):
        b, n, d = x.shape
        qkv = self.to_qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, d)
        return self.to_out(out)


class VisionTransformerBlock(nn.Module):
    """Vision Transformer block with patch embedding."""
    
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192, num_heads=8, depth=12, num_classes=10):
        super().__init__()
        
        # Patch embedding
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(embed_dim),
                'attn': SelfAttention(embed_dim, num_heads),
                'norm2': nn.LayerNorm(embed_dim),
                'mlp': nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Linear(embed_dim * 4, embed_dim)
                )
            })
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # (b, embed_dim, h', w')
        x = x.flatten(2).transpose(1, 2)  # (b, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = x + block['attn'](block['norm1'](x))
            x = x + block['mlp'](block['norm2'](x))
        
        x = self.norm(x)
        x = x[:, 0]  # Use class token
        return self.head(x)


class MultiModalClassifier(nn.Module):
    """Combine image and text features for classification."""
    
    def __init__(self, image_encoder, text_encoder, num_classes=10, fusion_dim=512):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        
        # Image feature extraction (global average pooling from conv layers)
        self.image_proj = nn.Linear(256, fusion_dim)  # Adjust based on image encoder output
        
        # Text feature projection
        self.text_proj = nn.Linear(256, fusion_dim)  # DistilBERT projected to 256
        
        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, num_classes)
        )
        
    def forward(self, images, texts=None):
        # Image features
        img_features = self.image_encoder(images)  # (batch, 256)
        img_proj = self.image_proj(img_features)
        
        # Text features (default: assume class descriptions)
        if texts is None:
            # Use default descriptions if no text provided
            texts = ["This is an image"]
        
        text_features = self.text_encoder(texts)  # (batch, 256)
        text_proj = self.text_proj(text_features)
        
        # Fuse and classify
        fused = torch.cat([img_proj, text_proj], dim=1)
        return self.fusion(fused)


class ResidualAttentionBlock(nn.Module):
    """Residual block with channel attention."""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        identity = x
        
        # Conv path
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Channel attention
        ca = self.avg_pool(out).flatten(1)
        ca = self.fc(ca).view(out.size(0), -1, 1, 1)
        out = out * ca
        
        return out + identity


class EfficientNetBlock(nn.Module):
    """Mobile-efficient block with separable convolutions."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expansion_factor=6):
        super().__init__()
        hidden_channels = in_channels * expansion_factor
        
        # Expansion phase
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True)
        ) if expansion_factor != 1 else nn.Identity()
        
        # Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size, stride=stride, padding=kernel_size//2, groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True)
        )
        
        # Projection phase
        self.project = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.use_res_connect = stride == 1 and in_channels == out_channels
        
    def forward(self, x):
        identity = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.project(x)
        if self.use_res_connect:
            x += identity
        return x


class EfficientNet(nn.Module):
    """Lightweight efficient CNN."""
    
    def __init__(self, num_classes=10, width_multiplier=1.0, depth_multiplier=1.0):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, int(32 * width_multiplier), 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(32 * width_multiplier)),
            nn.SiLU(inplace=True)
        )
        
        # Build efficient blocks
        def make_layers(in_channels, out_channels, expansion_factor, kernel_size, stride, num_blocks):
            layers = []
            for i in range(int(num_blocks * depth_multiplier)):
                if i == 0:
                    layers.append(EfficientNetBlock(in_channels, out_channels, kernel_size, stride, expansion_factor))
                else:
                    layers.append(EfficientNetBlock(out_channels, out_channels, kernel_size, 1, expansion_factor))
            return nn.Sequential(*layers)
        
        c = int(32 * width_multiplier)
        self.blocks = nn.Sequential(
            make_layers(c, int(16 * width_multiplier), 1, 3, 1, 1),
            make_layers(int(16 * width_multiplier), int(24 * width_multiplier), 6, 3, 2, 2),
            make_layers(int(24 * width_multiplier), int(40 * width_multiplier), 6, 5, 2, 2),
            make_layers(int(40 * width_multiplier), int(80 * width_multiplier), 6, 3, 2, 3),
            make_layers(int(80 * width_multiplier), int(112 * width_multiplier), 6, 5, 1, 3),
            make_layers(int(112 * width_multiplier), int(192 * width_multiplier), 6, 5, 2, 4),
            make_layers(int(192 * width_multiplier), int(320 * width_multiplier), 6, 3, 1, 1),
        )
        
        # Classification head
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(int(320 * width_multiplier), int(1280 * width_multiplier)),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(int(1280 * width_multiplier), num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.avg_pool(x).flatten(1)
        return self.classifier(x)
