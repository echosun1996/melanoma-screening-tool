from typing import Tuple, List, Union
from typing import List, Tuple, Dict, Union, Optional
import traceback
import torch
from torch import nn
import torch.nn.functional as F

class ReconstructionLoss(torch.nn.Module):
    """Loss function for tabular data reconstruction"""
    def __init__(self, num_cat: int, cat_offsets: torch.Tensor, num_con: int) -> None:
        super().__init__()
        self.num_cat = num_cat
        self.num_con = num_con
        self.register_buffer('cat_offsets', cat_offsets)
        self.softmax = nn.Softmax(dim=-1)
        self.eps = 1e-6

        print(f"\nInitializing ReconstructionLoss:")
        print(f"Number of categorical features: {num_cat}")
        print(f"Number of continuous features: {num_con}")
        print(f"Category offsets: {cat_offsets}")

    
    # def forward(self, out: Tuple, y: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
    #     B, _, D = out[0].shape
    #     # (B*N1, D)
    #     out_cat = out[0].reshape(B*self.num_cat, D)
    #     # (B, N2)  
    #     out_con = out[1].squeeze(-1)
    #     target_cat = (y[:, :self.num_cat].long()+self.cat_offsets).reshape(B*self.num_cat)
    #     target_con = y[:, self.num_cat:]
    #     mask_cat = mask[:, :self.num_cat].reshape(B*self.num_cat)
    #     mask_con = mask[:, self.num_cat:]

    #     # cat loss
    #     prob_cat = self.softmax(out_cat)
    #     onehot_cat = torch.nn.functional.one_hot(target_cat, num_classes=D)
    #     loss_cat = -onehot_cat * torch.log(prob_cat+1e-8)
    #     loss_cat = loss_cat.sum(dim=1)
    #     loss_cat = (loss_cat*mask_cat).sum()/mask_cat.sum()   

    #     # con loss
    #     loss_con = (out_con-target_con)**2
    #     loss_con = (loss_con*mask_con).sum()/mask_con.sum()   

    #     loss = (loss_cat + loss_con)/2

    #     return loss, prob_cat, target_cat, mask_cat



    # OK
    def forward(self, multimodal_output: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], 
            target: torch.Tensor, 
            mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            multimodal_output: 模型输出,(cat_out, con_out)元组或单个张量
            target: 目标值
            mask: 随机生成的mask
        Returns:
            total_loss: 总损失
            cat_probs: 分类概率
            cat_targets: 分类目标
            cat_mask: 分类mask
        """
        try:
            # 生成随机mask
            if mask is None:
                B = target.shape[0]
                mask = torch.rand(B, target.shape[1], device=target.device) < 0.5

            # 处理输入
            if isinstance(multimodal_output, tuple):
                out_cat, out_con = multimodal_output[0], multimodal_output[1]
            else:
                if multimodal_output.dim() == 3:  
                    B = multimodal_output.shape[0]
                    out_cat = multimodal_output[:, :self.num_cat]
                    out_con = multimodal_output[:, self.num_cat:]
                else:  
                    B = multimodal_output.shape[0]
                    out_cat = multimodal_output[:, :self.num_cat].view(B, self.num_cat, -1)
                    out_con = multimodal_output[:, self.num_cat:].view(B, -1)
                
            # 提取目标
            cat_targets = target[:, :self.num_cat].long()
            con_targets = target[:, self.num_cat:self.num_cat + self.num_con]
            
            # 应用mask
            cat_mask = mask[:, :self.num_cat]
            con_mask = mask[:, self.num_cat:self.num_cat + self.num_con]

            # 计算分类损失
            loss_cat = 0
            cat_probs_list = []
            for i in range(self.num_cat):
                curr_logits = out_cat[:, i]
                curr_targets = torch.clamp(cat_targets[:, i] + self.cat_offsets[i], 0, out_cat.shape[-1]-1)
                curr_mask = cat_mask[:, i]
                curr_probs = self.softmax(curr_logits)
                cat_probs_list.append(curr_probs)
                curr_onehot = F.one_hot(curr_targets, num_classes=curr_logits.shape[-1])
                curr_loss = -(curr_onehot * torch.log(curr_probs + self.eps)).sum(dim=-1)
                curr_loss = (curr_loss * curr_mask).sum() / (curr_mask.sum() + self.eps)
                loss_cat += curr_loss

            loss_cat = loss_cat / self.num_cat
            cat_probs = torch.stack(cat_probs_list, dim=1)

            # 计算连续值损失
            if out_con.dim() == 3:
                out_con = out_con.mean(dim=-1)
                    
            if con_mask.any():
                scale = torch.max(torch.abs(con_targets[con_mask])).detach() + self.eps
                if scale > 1:
                    out_con = out_con / scale
                    con_targets = con_targets / scale
                
            out_con = torch.clamp(out_con, -100, 100)
            con_targets = torch.clamp(con_targets, -100, 100)
            
            squared_diff = (out_con - con_targets) ** 2
            loss_con = torch.where(con_mask, squared_diff, torch.zeros_like(squared_diff))
            mask_sum = con_mask.sum().float()
            loss_con = loss_con.sum() / (mask_sum + self.eps) if mask_sum > 0 else torch.tensor(0.0, device=out_con.device)

            # 总损失
            total_loss = (loss_cat + loss_con) / 2
            return total_loss, cat_probs, cat_targets, cat_mask

        except Exception as e:
            print(f"\nError in ReconstructionLoss forward:")
            print(f"Error: {str(e)}")
            raise
