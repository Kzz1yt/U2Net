import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class FCSF(nn.Module):
    """Fast Cross-Scale Fusion Module"""
    def __init__(self, in_channels_list, out_channels):
        super(FCSF, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        
        # 1. Reparameterization - unify channels with 1x1 conv
        self.reparam_layers = nn.ModuleList()
        for in_channels in in_channels_list:
            self.reparam_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
        
        # 2. Linear Mapping - lightweight linear mapping
        self.linear_mapping = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        
        # 3. Weighted Integration - learn weights for different scales
        self.attention = nn.Parameter(torch.ones(len(in_channels_list)) / len(in_channels_list), requires_grad=True)
        
        # Final representation
        self.final_conv = ConvBnRelu(out_channels, out_channels)
    
    def forward(self, features):
        assert len(features) == len(self.in_channels_list)
        
        # 1. Reparameterization
        reparam_features = []
        
        # Find minimum spatial size across all features
        min_h = min(feat.size()[2] for feat in features)
        min_w = min(feat.size()[3] for feat in features)
        target_size = (min_h, min_w)
        
        # Align all features to target size
        aligned_features = []
        for feat in features:
            if feat.size()[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=True)
            aligned_features.append(feat)
        
        # Apply reparameterization
        for i, (feat, reparam_layer) in enumerate(zip(aligned_features, self.reparam_layers)):
            reparam_feat = reparam_layer(feat)
            reparam_features.append(reparam_feat)
        
        # 2. Linear Mapping
        mapped_features = []
        for feat in reparam_features:
            mapped_feat = self.linear_mapping(feat)
            mapped_features.append(mapped_feat)
        
        # 3. Weighted Integration and Representations
        # Get weights using softmax
        weights = F.softmax(self.attention, dim=0)
        
        # Verify all mapped features have same size
        for i, feat in enumerate(mapped_features):
            if feat.size()[2:] != target_size:
                mapped_features[i] = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=True)
        
        # Weighted fusion
        fused = torch.zeros_like(mapped_features[0])
        for weight, feat in zip(weights, mapped_features):
            # Ensure size match again
            if feat.size() != fused.size():
                feat = F.interpolate(feat, size=fused.size()[2:], mode='bilinear', align_corners=True)
            fused = fused + weight * feat
        
        # Final representation
        fused = self.final_conv(fused)
        
        return fused, weights

class BEGM(nn.Module):
    """Boundary-Enhanced Guidance Module"""
    def __init__(self, in_channels):
        super(BEGM, self).__init__()
        
        # 1. Boundary-aware feature learning
        # Use small kernels and deep supervision to learn boundary features
        self.boundary_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.boundary_bn = nn.BatchNorm2d(in_channels)
        
        # 2. Boundary attention generation
        self.boundary_attention = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        
        # 3. Feature refinement with boundary guidance
        self.refine_conv = ConvBnRelu(in_channels, in_channels)
    
    def forward(self, x):
        # 1. Extract boundary features
        boundary_feat = self.boundary_conv(x)
        boundary_feat = self.boundary_bn(boundary_feat)
        boundary_feat = F.relu(boundary_feat, inplace=True)
        
        # 2. Generate boundary attention map
        boundary_att = self.boundary_attention(boundary_feat)
        boundary_att = torch.sigmoid(boundary_att)
        
        # 3. Use boundary attention to guide feature learning
        guided_feat = x * (1 + boundary_att)  # Enhance features at boundary regions
        
        # 4. Feature refinement
        refined_feat = self.refine_conv(guided_feat)
        
        return refined_feat

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv1 = ConvBnRelu(in_channels, out_channels)
        self.conv2 = ConvBnRelu(out_channels, out_channels)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        out = self.pool(x)
        return x, x.clone()

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = ConvBnRelu(in_channels, out_channels)
    
    def forward(self, x1, x2):
        # Upsample x1 to x2 size
        x1 = self.up(x1)
        
        # Ensure x1 and x2 size match
        if x1.size()[2:] != x2.size()[2:]:
            x1 = F.interpolate(x1, size=x2.size()[2:], mode='bilinear', align_corners=True)
        
        # Concatenate features
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class FastLiteSSNet(nn.Module):
    def __init__(self, num_classes=10, width_mult=0.5):
        super(FastLiteSSNet, self).__init__()
        
        # Width multiplier for lightweight
        c = lambda x: int(x * width_mult)
        
        # Encoder
        self.enc1 = DownBlock(3, c(32))      # 3 → 32
        self.enc2 = DownBlock(c(32), c(64))   # 32 → 64
        self.enc3 = DownBlock(c(64), c(128))  # 64 → 128
        self.enc4 = DownBlock(c(128), c(256)) # 128 → 256
        
        # 1. Fast Cross-Scale Fusion (FCSF)
        self.fcsf = FCSF([c(32), c(64), c(128), c(256)], c(256))
        
        # 2. Boundary-Enhanced Guidance Module (BEGM)
        self.begm_fcsf = BEGM(c(256))  # Apply after FCSF output
        self.begm_dec1 = BEGM(c(256))  # Apply at each decoder stage
        self.begm_dec2 = BEGM(c(128))
        self.begm_dec3 = BEGM(c(64))
        self.begm_dec4 = BEGM(c(32))
        
        # Bottleneck
        self.bottleneck = ConvBnRelu(c(256), c(512))
        
        # Decoder
        self.dec1 = UpBlock(c(512) + c(256), c(256))
        self.dec2 = UpBlock(c(256) + c(128), c(128))
        self.dec3 = UpBlock(c(128) + c(64), c(64))
        self.dec4 = UpBlock(c(64) + c(32), c(32))
        
        # Final prediction
        self.final = nn.Conv2d(c(32), num_classes, kernel_size=1)
    
    def forward(self, x, return_features=False):
        # Encoder
        x1, skip1 = self.enc1(x)
        x2, skip2 = self.enc2(x1)
        x3, skip3 = self.enc3(x2)
        x4, skip4 = self.enc4(x3)
        
        # 1. Fast Cross-Scale Fusion
        fused_feat, fusion_weights = self.fcsf([skip1, skip2, skip3, skip4])
        
        # 2. Boundary-Enhanced Guidance on fused features
        fused_feat = self.begm_fcsf(fused_feat)
        
        # Bottleneck
        x = self.bottleneck(fused_feat)
        
        # Decoder with boundary guidance
        dec1_out = self.dec1(x, skip4)
        dec1_out = self.begm_dec1(dec1_out)  # Boundary guidance after each decoder block
        
        dec2_out = self.dec2(dec1_out, skip3)
        dec2_out = self.begm_dec2(dec2_out)
        
        dec3_out = self.dec3(dec2_out, skip2)
        dec3_out = self.begm_dec3(dec3_out)
        
        dec4_out = self.dec4(dec3_out, skip1)
        dec4_out = self.begm_dec4(dec4_out)
        
        # Final prediction
        out = self.final(dec4_out)
        
        if return_features:
            # Return features matching teacher model for distillation
            # Teacher returns 5 features: [feat1, feat2, feat3, feat4, feat5]
            # Student returns corresponding features: [skip1, skip2, skip3, skip4, fused_feat]
            return out, [skip1, skip2, skip3, skip4, fused_feat]
        else:
            return out
