import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPLoss(nn.Module):
    def __init__(self, temperature: float, lambda_0: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.lambda_0 = lambda_0
        self.lambda_1 = 1-lambda_0

    def forward(self, out0: torch.Tensor, out1: torch.Tensor):
        # Handle ViT sequence output - take CLS token
        if out0.dim() == 3:
            out0 = out0[:, 0]  # Take CLS token
        if out1.dim() == 3:
            out1 = out1[:, 0]  # Take CLS token
            
        # Normalize embeddings
        out0 = F.normalize(out0, dim=1)
        out1 = F.normalize(out1, dim=1)
        
        # Calculate similarity matrix
        logits = torch.matmul(out0, out1.T) / self.temperature
        
        # Calculate bi-directional contrastive loss
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_0 = self.lambda_0 * F.cross_entropy(logits, labels)
        loss_1 = self.lambda_1 * F.cross_entropy(logits.T, labels)
        
        return loss_0 + loss_1, logits, None




# RESNET50
# import torch
# import torch.nn as nn
# import traceback
# import torch.nn.functional as F
# from typing import List, Tuple, Dict, Any

# class CLIPLoss(torch.nn.Module):
#     """Loss function for multimodal contrastive learning"""
#     def __init__(self, temperature: float, lambda_0: float = 0.5) -> None:
#         super().__init__()
#         self.temperature = temperature
#         self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
#         self.lambda_0 = lambda_0
#         self.lambda_1 = 1-lambda_0
#         print(f"\nInitializing CLIPLoss:")
#         print(f"Temperature: {temperature}")
#         print(f"Lambda_0: {lambda_0}")

#     def forward(self, out0: torch.Tensor, out1: torch.Tensor, indices: List[int] = None) -> Tuple[torch.Tensor, torch.Tensor, None]:
#         out0 = nn.functional.normalize(out0, dim=1)
#         out1 = nn.functional.normalize(out1, dim=1) 
        
#         logits = torch.matmul(out0, out1.T) / self.temperature
        
#         # 使用batch size作为标签
#         labels = torch.arange(out0.size(0), device=out0.device)
        
#         loss_0 = self.lambda_0 * self.cross_entropy(logits, labels)
#         loss_1 = self.lambda_1 * self.cross_entropy(logits.T, labels)
#         loss = loss_0 + loss_1
        
#         return loss, logits, None
        
