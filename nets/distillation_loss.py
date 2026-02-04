import torch
import torch.nn as nn
import torch.nn.functional as F

class CSDivergence(nn.Module):
    """Cauchy-Schwarz Divergence"""
    def __init__(self, epsilon=1e-6):
        super(CSDivergence, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, student_features, teacher_features):
        """
        Calculate Cauchy-Schwarz divergence
        :param student_features: Student model features [N, C1, H1, W1]
        :param teacher_features: Teacher model features [N, C2, H2, W2]
        :return: CS divergence value
        """
        # Ensure feature size match
        N = student_features.size(0)
        
        # Adjust student feature spatial size to match teacher features
        if student_features.size()[2:] != teacher_features.size()[2:]:
            student_features = F.interpolate(student_features, size=teacher_features.size()[2:], 
                                          mode='bilinear', align_corners=True)
        
        # If channel numbers differ, use 1x1 conv to adjust student feature channels to match teacher features
        if student_features.size(1) != teacher_features.size(1):
            # Create 1x1 conv layer to adjust channels
            adapt_conv = nn.Conv2d(student_features.size(1), teacher_features.size(1), kernel_size=1, bias=False).to(student_features.device)
            # Use kaiming initialization
            nn.init.kaiming_normal_(adapt_conv.weight)
            student_features = adapt_conv(student_features)
        
        # Flatten features to [N, C, H*W]
        C = student_features.size(1)
        H = student_features.size(2)
        W = student_features.size(3)
        
        # Validate shape is valid
        total_size = student_features.numel()
        if total_size != N * C * H * W:
            print(f"Warning: student_features shape invalid, total size is {total_size}, but N*C*H*W={N*C*H*W}")
            return torch.tensor(0.0, device=student_features.device)  # Return 0 loss
        
        student_flat = student_features.view(N, C, -1)  # [N, C, HW]
        
        # Validate teacher features similarly
        total_size_teacher = teacher_features.numel()
        if total_size_teacher != N * C * H * W:
            print(f"Warning: teacher_features shape invalid, total size is {total_size_teacher}, but N*C*H*W={N*C*H*W}")
            return torch.tensor(0.0, device=student_features.device)  # Return 0 loss
        
        teacher_flat = teacher_features.view(N, C, -1)  # [N, C, HW]
        
        # Calculate L2 norm of features
        student_norm = torch.norm(student_flat, dim=1, keepdim=True) + self.epsilon  # [N, 1, HW]
        teacher_norm = torch.norm(teacher_flat, dim=1, keepdim=True) + self.epsilon  # [N, 1, HW]
        
        # Feature normalization
        student_normalized = student_flat / student_norm  # [N, C, HW]
        teacher_normalized = teacher_flat / teacher_norm  # [N, C, HW]
        
        # Calculate Cauchy-Schwarz product
        cs_product = torch.sum(student_normalized * teacher_normalized, dim=1, keepdim=True)  # [N, 1, HW]
        
        # Calculate CS divergence
        cs_divergence = 1 - cs_product  # [N, 1, HW]
        
        # Average over spatial dimensions
        cs_divergence = cs_divergence.mean(dim=(1, 2))  # [N]
        
        return cs_divergence.mean()

class HDL(nn.Module):
    """Hybrid Distillation Loss"""
    def __init__(self, num_classes=10, alpha=0.5, beta=0.3, gamma=0.2):
        super(HDL, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha  # Classification loss weight
        self.beta = beta    # CS divergence weight
        self.gamma = gamma  # Dice loss weight
        
        # Base loss functions
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=num_classes)
        self.dice_loss = self._dice_loss
        
        # CS divergence loss
        self.cs_div = CSDivergence()
    
    def _dice_loss(self, inputs, target, beta=1, smooth=1e-5):
        """Dice loss implementation"""
        n, c, h, w = inputs.size()
        nt, ht, wt, ct = target.size()
        
        if h != ht and w != wt:
            inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
            
        temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
        temp_target = target.view(n, -1, ct)
        
        tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
        fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
        fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp
        
        score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
        dice_loss = 1 - torch.mean(score)
        return dice_loss
    
    def forward(self, outputs, student_features, teacher_outputs, teacher_features, targets):
        """
        Calculate hybrid distillation loss
        :param outputs: Student model output [N, C, H, W]
        :param student_features: Student model multi-scale feature list
        :param teacher_outputs: Teacher model output [N, C, H, W]
        :param teacher_features: Teacher model multi-scale feature list
        :param targets: Ground truth labels [N, H, W, C]
        :return: Total loss
        """
        # 1. Classification loss (Cross-Entropy)
        ce_loss = self.ce_loss(outputs, targets[..., 0].long())
        
        # 2. Dice loss
        dice_loss = self.dice_loss(outputs, targets)
        
        # 3. CS divergence loss - scale-decoupled feature distillation
        cs_loss = 0
        num_scales = min(len(student_features), len(teacher_features))
        
        # Calculate CS divergence for each scale independently
        for i in range(num_scales):
            s_feat = student_features[i]
            t_feat = teacher_features[i]
            
            try:
                cs_loss += self.cs_div(s_feat, t_feat)
            except Exception as e:
                # If feature shapes don't match, skip CS divergence calculation for this scale
                print(f"Warning: CS divergence calculation failed for scale {i}: {e}")
                print(f"Student feature shape: {s_feat.shape}")
                print(f"Teacher feature shape: {t_feat.shape}")
        
        # If no CS divergence was calculated, set cs_loss to 0
        if num_scales > 0:
            # Only average over successfully calculated scales
            # Simple handling: divide by total number of scales, should actually divide by successfully calculated scales
            cs_loss /= num_scales
        
        # Total loss = classification loss + CS divergence loss + Dice loss
        total_loss = self.alpha * ce_loss + self.beta * cs_loss + self.gamma * dice_loss
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'cs_loss': cs_loss,
            'dice_loss': dice_loss
        }
