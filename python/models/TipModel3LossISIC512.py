'''
* Licensed under the Apache License, Version 2.
* Modified for ISIC dataset with gradient accumulation support
* Based on TIP codebase
'''
from typing import List, Tuple, Dict, Any, Optional

import random
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from sklearn.linear_model import LogisticRegression
from lightly.models.modules import SimCLRProjectionHead
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pl_bolts.utils.self_supervised import torchvision_ssl_encoder
import math
import traceback

from .Tip_utils.Transformer import TabularTransformerEncoder, MultimodalTransformerEncoder, TabularPredictor
from .Tip_utils.VisionTransformer_imagenet import create_vit
from .EvaluatorpAUC import PartialAUC

from .utils.clip_loss import CLIPLoss
from .utils.reconstruct_loss import ReconstructionLoss


class TIP3LossISIC(pl.LightningModule):
    """ISIC dataset three-loss pre-training model with gradient accumulation"""
    
    def __init__(self, hparams) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        
        print("\nInitializing TIP3Loss model for ISIC dataset with gradient accumulation...")
        
        # 添加这两个属性的初始化
        self._epoch_start_printed = False
        self._epoch_header_printed = False
        
        # Initialize encoders
        self.initialize_imaging_encoder_and_projector()
        if self.hparams.pretrain.checkpoints.imaging:
            self.load_pretrained_imaging_weights()
            
        self.initialize_tabular_encoder_and_projector()
        self.initialize_multimodal_encoder_and_predictor()
        
        # ITM head
        self.itm_head = torch.nn.Linear(self.hparams.multimodal_embedding_dim, 2)
        
        pos_weight = self.calculate_pos_weight()
        print(f"\nPositive sample weight: {pos_weight:.2f}")

        # Loss functions
        self.criterion_val_itc = CLIPLoss(
            temperature=self.hparams.temperature,
            lambda_0=self.hparams.lambda_0
        )
        self.criterion_train_itc = self.criterion_val_itc
        
        self.criterion_tr = ReconstructionLoss(
            num_cat=2,
            cat_offsets=self.encoder_tabular.cat_offsets,
            num_con=30
        )
        
        self.criterion_itm = torch.nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, pos_weight]).to(self.device)
        )
        
        # Initialize metrics
        self.initialize_metrics()
        self.clip_val = 1.0
        
        # Gradient accumulation settings
        self.accum_iter = 4  # Accumulate gradients over 4 batches for effective batch size of 512
        self.automatic_optimization = False  # Turn off automatic optimization for manual gradient accumulation

        print("\nModel Configuration:")
        print(f"Batch size: {self.hparams.batch_size}")
        print(f"Accumulation steps: {self.accum_iter}")
        print(f"Effective batch size: {self.hparams.batch_size * self.accum_iter}")
        print(f"Temperature: {self.hparams.temperature}")
        print(f"Lambda_0: {self.hparams.lambda_0}")
        print(f"Learning rate: {self.hparams.lr}")
        print(f"Weight decay: {self.hparams.weight_decay}")
        if self.hparams.pretrain.enabled:
            print("Using pretrained model")

    def initialize_metrics(self) -> None:
        """初始化评估指标"""
        self.classifier_acc_train = torchmetrics.Accuracy(task='binary')
        self.classifier_acc_val = torchmetrics.Accuracy(task='binary')
        self.classifier_auc_train = torchmetrics.AUROC(task='binary')
        self.classifier_auc_val = torchmetrics.AUROC(task='binary')
        self.pauc_train = PartialAUC(min_tpr=0.80)
        self.pauc_val = PartialAUC(min_tpr=0.80)
        print("Metrics initialized")

    def load_pretrained_imaging_weights(self) -> None:
        """加载预训练的图像编码器权重"""
        print("\nLoading pretrained weights...")
        try:
            loaded_chkpt = torch.load(self.hparams.pretrain.checkpoints.imaging)
            state_dict = loaded_chkpt['state_dict']
            
            state_dict_encoder = {
                k[len('encoder_imaging.'):]: v 
                for k, v in state_dict.items()
                if k.startswith('encoder_imaging.')
            }
            
            missing, unexpected = self.encoder_imaging.load_state_dict(
                state_dict_encoder, 
                strict=False
            )
            print(f"Loaded from: {self.hparams.pretrain.checkpoints.imaging}")
            print(f"Missing keys: {missing}")
            print(f"Unexpected keys: {unexpected}")
            
            if self.hparams.pretrain.strategies.imaging == 'frozen':
                print("Freezing pretrained weights...")
                for param in self.encoder_imaging.parameters():
                    param.requires_grad = False
                    
        except Exception as e:
            print(f"Error loading pretrained weights: {str(e)}")
            raise

    def initialize_imaging_encoder_and_projector(self) -> None:
        """初始化图像编码器和投影头"""
        if self.hparams.model.startswith('vit'):
            self.encoder_imaging_type = 'vit'
            self.encoder_imaging = create_vit(self.hparams)
            self.pooled_dim = self.hparams.embedding_dim
        elif self.hparams.model.startswith('resnet'):
            self.encoder_imaging_type = 'resnet'
            self.encoder_imaging = torchvision_ssl_encoder(
                self.hparams.model, 
                return_all_feature_maps=True
            )
            self.pooled_dim = 2048 if self.hparams.model=='resnet50' else 512
            
        self.projector_imaging = SimCLRProjectionHead(
            input_dim=self.pooled_dim,
            hidden_dim=self.hparams.embedding_dim,
            output_dim=self.hparams.projection_dim
        )

    def initialize_tabular_encoder_and_projector(self) -> None:
        """初始化表格编码器和投影头"""
        self.encoder_tabular = TabularTransformerEncoder(
            args=self.hparams,
            cat_lengths_tabular=[2, 2],  # ISIC dataset has 2 categorical features
            con_lengths_tabular=[1]*30  # ISIC dataset has 30 numerical features
        )
        
        self.projector_tabular = SimCLRProjectionHead(
            input_dim=self.hparams.tabular_embedding_dim,
            hidden_dim=self.hparams.tabular_embedding_dim,
            output_dim=self.hparams.projection_dim
        )

    def initialize_multimodal_encoder_and_predictor(self) -> None:
        """初始化多模态编码器和预测器"""
        self.encoder_multimodal = MultimodalTransformerEncoder(
            args=self.hparams
        )

        self.predictor_tabular = TabularPredictor(
            args=self.hparams,
            cat_lengths_tabular=[2, 2],
            con_lengths_tabular=[1]*30,
            num_unique_cat=2
        )

    def calculate_pos_weight(self) -> float:
        """计算正样本权重"""
        try:
            labels_path = self.hparams.labels_train_path if not self.hparams.labels_train_path.startswith('/') else self.hparams.labels_train_path
            full_path = labels_path if not self.hparams.data_base or labels_path.startswith('/') else f"{self.hparams.data_base}/{labels_path}"
            
            import os
            if not os.path.exists(full_path):
                raise ValueError(f"Labels file not found at: {full_path}")
                    
            labels = torch.load(full_path)
            neg_count = (labels == 0).sum()
            pos_count = (labels == 1).sum()
            
            if pos_count == 0:
                raise ValueError("No positive samples found in training data")
                    
            pos_weight = float(neg_count) / float(pos_count)
            return pos_weight
                
        except Exception as e:
            print(f"Error calculating positive weight: {str(e)}")
            print("Using default weight of 100.0")
            return 100.0

    def forward_imaging(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播 - 图像部分"""
        if self.encoder_imaging_type == 'vit':
            y = self.encoder_imaging(x)[0]
        else:
            y = self.encoder_imaging(x)[-1]
            y = F.adaptive_avg_pool2d(y, (1, 1)).squeeze(-1).squeeze(-1)
            
        z = self.projector_imaging(y)
        return z, y

    def forward_tabular(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                      mask_special: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播 - 表格部分"""
        y = self.encoder_tabular(x, mask=mask, mask_special=mask_special)
        z = self.projector_tabular(y[:, 0, :])  # take CLS token
        return z, y

    def cal_image_tabular_matching_loss(self, image_embeddings: torch.Tensor,
                                      tabular_embeddings: torch.Tensor,
                                      logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算图像-表格匹配损失"""
        try:
            current_device = image_embeddings.device
            output_pos = self.forward_multimodal_feature(
                tabular_features=tabular_embeddings,
                image_features=image_embeddings
            )
            B = image_embeddings.shape[0]
            
            # 计算权重矩阵
            with torch.no_grad():
                weights_i2t = F.softmax(logits, dim=1)
                weights_t2i = F.softmax(logits.T, dim=1)
                
                # 将对角线置为0
                mask = torch.eye(B, device=current_device).bool()
                weights_i2t.masked_fill_(mask, 0)
                weights_t2i.masked_fill_(mask, 0)
                
                # 处理可能的数值问题
                weights_i2t = torch.nan_to_num(weights_i2t, 0)
                weights_t2i = torch.nan_to_num(weights_t2i, 0)
                
                # 确保每行和不为0
                row_sums_i2t = weights_i2t.sum(dim=1, keepdim=True)
                row_sums_t2i = weights_t2i.sum(dim=1, keepdim=True)
                
                # 避免除以0
                row_sums_i2t = torch.where(row_sums_i2t == 0, 
                                         torch.ones_like(row_sums_i2t),
                                         row_sums_i2t)
                row_sums_t2i = torch.where(row_sums_t2i == 0,
                                         torch.ones_like(row_sums_t2i),
                                         row_sums_t2i)
                
                # 归一化
                weights_i2t = weights_i2t / row_sums_i2t
                weights_t2i = weights_t2i / row_sums_t2i

            # 创建负样本
            tabular_embeddings_neg = []
            image_embeddings_neg = []
            
            for b in range(B):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                tabular_embeddings_neg.append(tabular_embeddings[neg_idx])
                
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                image_embeddings_neg.append(image_embeddings[neg_idx])
                    
            tabular_embeddings_neg = torch.stack(tabular_embeddings_neg)
            image_embeddings_neg = torch.stack(image_embeddings_neg)
            
            # 计算匹配分数
            output_neg = self.forward_multimodal_feature(
                tabular_features=torch.cat([tabular_embeddings, tabular_embeddings_neg]),
                image_features=torch.cat([image_embeddings_neg, image_embeddings])
            )
            
            z = self.itm_head(torch.cat([output_pos, output_neg], dim=0))
            itm_labels = torch.cat([
                torch.ones(B, device=current_device),
                torch.zeros(2*B, device=current_device)
            ], dim=0).long()
            
            loss_itm = self.criterion_itm(z, itm_labels)
            
            return loss_itm, z, itm_labels

        except Exception as e:
            print(f"\nError in cal_image_tabular_matching_loss:")
            print(f"Image embeddings shape: {image_embeddings.shape}")
            print(f"Tabular embeddings shape: {tabular_embeddings.shape}")
            print(f"Logits shape: {logits.shape}")
            raise

    def training_step(self, batch, batch_idx):
        """训练步骤,使用梯度累积实现更大的有效批量大小"""
        # Get optimizer
        opt = self.optimizers()
        
        # 计算当前步骤是否需要梯度更新
        is_accumulating = (batch_idx + 1) % self.accum_iter != 0
        
        # 1. 解包输入数据
        im_views, tab_views, _, _, original_tab = batch
        
        # 2. ITC Loss
        z0, image_embeddings = self.forward_imaging(im_views[1])
        z1, tabular_embeddings = self.forward_tabular(tab_views[0])
        loss_itc, logits, _ = self.criterion_train_itc(z0, z1)
        
        # 3. ITM Loss
        loss_itm, _, _ = self.cal_image_tabular_matching_loss(
            image_embeddings,
            tabular_embeddings,
            logits
        )
        
        # 4. TR Loss 
        mask = tab_views[2]
        mask_special = tab_views[3]
        _, tabular_embeddings = self.forward_tabular(
            tab_views[1],
            mask=mask,
            mask_special=mask_special
        )
        _, multimodal_embeddings = self.forward_multimodal(
            tabular_features=tabular_embeddings,
            image_features=image_embeddings
        )

        cat_outputs, con_outputs = self.predictor_tabular(multimodal_embeddings)
        loss_tr, _, _, _ = self.criterion_tr(
            (cat_outputs, con_outputs),
            original_tab,
            mask=mask
        )
        
        # 5. 总损失 - 缩放累积梯度
        loss = (loss_itc + loss_tr + loss_itm) / 3.0
        scaled_loss = loss / self.accum_iter  # 除以累积步数，缩放梯度
        
        # 手动反向传播
        self.manual_backward(scaled_loss)
        
        # 记录训练指标
        self.log('train/itc_loss', loss_itc.item(), prog_bar=False, on_step=True, on_epoch=True, batch_size=len(im_views[1]))
        self.log('train/itm_loss', loss_itm.item(), prog_bar=False, on_step=True, on_epoch=True, batch_size=len(im_views[1]))
        self.log('train/tr_loss', loss_tr.item(), prog_bar=False, on_step=True, on_epoch=True, batch_size=len(im_views[1]))
        self.log('train/loss', loss.item(), prog_bar=True, on_step=True, on_epoch=True, batch_size=len(im_views[1]))

        # 只有在累积完成时才更新优化器
        if not is_accumulating:
            # 梯度裁剪 - 使用PyTorch的函数代替Lightning的方法
            if self.clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_val)
                
            # 执行优化步骤
            opt.step()
            # 清零梯度
            opt.zero_grad()
                
        # 返回损失以便在进度条中显示
        return {'loss': loss}

    def on_train_batch_start(self, batch, batch_idx):
        """训练批次开始时的处理 - 实现学习率调度"""
        # 计算当前迭代的进度
        if hasattr(self.hparams, 'scheduler') and self.hparams.scheduler == 'cosine':
            train_loader_len = self.trainer.num_training_batches
            # 只在每个累积周期的开始调整学习率
            if batch_idx % self.accum_iter == 0:
                # 计算全局步数和总步数
                global_step = self.trainer.global_step // self.accum_iter  # 考虑累积步数
                total_steps = (train_loader_len // self.accum_iter) * self.trainer.max_epochs
                
                # 获取优化器
                optimizer = self.optimizers()
                
                # 计算并更新学习率
                if self.trainer.current_epoch < self.hparams.warmup_epochs:
                    # 在预热阶段线性增加学习率
                    warmup_steps = (train_loader_len // self.accum_iter) * self.hparams.warmup_epochs
                    warmup_progress = min(1.0, global_step / warmup_steps)
                    lr = self.hparams.lr * warmup_progress
                else:
                    # 预热后使用余弦衰减
                    warmup_steps = (train_loader_len // self.accum_iter) * self.hparams.warmup_epochs
                    cosine_steps = total_steps - warmup_steps
                    cosine_progress = min(1.0, (global_step - warmup_steps) / cosine_steps)
                    lr = self.hparams.lr * 0.5 * (1 + math.cos(math.pi * cosine_progress))
                
                # 更新所有参数组的学习率
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                
                # 记录学习率
                self.log('learning_rate', lr, on_step=True, on_epoch=False)

    def validation_step(self, batch, batch_idx):
        """验证步骤,仅包含预训练loss的验证"""
        try:
            # 1. 解包数据 
            im_views, tab_views, _, original_im, original_tab = batch
            original_im = original_im.to(self.device)
            original_tab = original_tab.to(self.device)
            
            # 2. ITC Loss
            z0, image_embeddings = self.forward_imaging(original_im)
            z1, tabular_embeddings = self.forward_tabular(original_tab)
            loss_itc, logits, _ = self.criterion_val_itc(z0, z1)
            
            # 3. ITM Loss
            loss_itm, _, _ = self.cal_image_tabular_matching_loss(
                image_embeddings, 
                tabular_embeddings, 
                logits
            )
            
            # 4. TR Loss
            mask = tab_views[2]
            mask_special = tab_views[3]
            _, tabular_embeddings = self.forward_tabular(
                tab_views[1],
                mask=mask,
                mask_special=mask_special
            )
            _, multimodal_embeddings = self.forward_multimodal(
                tabular_features=tabular_embeddings,
                image_features=image_embeddings
            )

            cat_outputs, con_outputs = self.predictor_tabular(multimodal_embeddings)
            loss_tr, _, _, _ = self.criterion_tr(
                (cat_outputs, con_outputs),
                original_tab,
                mask=mask
            )
            
            # 5. 总损失
            loss = (loss_itc + loss_tr + loss_itm) / 3.0
            
            # 记录验证指标
            self.log('val/itc_loss', loss_itc.item(), prog_bar=False, on_step=False, on_epoch=True, batch_size=len(original_im))
            self.log('val/itm_loss', loss_itm.item(), prog_bar=False, on_step=False, on_epoch=True, batch_size=len(original_im))
            self.log('val/tr_loss', loss_tr.item(), prog_bar=False, on_step=False, on_epoch=True, batch_size=len(original_im))
            self.log('val/loss', loss.item(), prog_bar=True, on_step=False, on_epoch=True, batch_size=len(original_im))

            return {
                'val_loss': loss, 
                'itc_loss': loss_itc,
                'itm_loss': loss_itm,
                'tr_loss': loss_tr
            }
        except Exception as e:
            print(f"\nError in validation step:")
            print(f"Batch index: {batch_idx}")
            print(f"Error: {str(e)}")
            traceback.print_exc()
            raise

    def validation_epoch_end(self, outputs) -> None:
        """预训练阶段的验证epoch结束处理"""
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        
        # 只记录预训练相关的loss
        metrics = {
            "val/loss": avg_loss,
            "val/itc_loss": torch.stack([x.get('itc_loss', torch.tensor(0.0)) for x in outputs]).mean(),
            "val/itm_loss": torch.stack([x.get('itm_loss', torch.tensor(0.0)) for x in outputs]).mean(),
            "val/tr_loss": torch.stack([x.get('tr_loss', torch.tensor(0.0)) for x in outputs]).mean(),
        }
        
        self.log_dict(metrics, on_epoch=True)
        
        # 打印验证指标
        print(f"\nValidation Metrics:")
        print(f"Total Loss: {metrics['val/loss']:.4f}")
        print(f"ITC Loss: {metrics['val/itc_loss']:.4f}")
        print(f"ITM Loss: {metrics['val/itm_loss']:.4f}")
        print(f"MTR Loss: {metrics['val/tr_loss']:.4f}")

    def forward_multimodal(self, tabular_features: torch.Tensor,
                      image_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """处理多模态特征并预测"""
        try:
            # 处理维度
            if len(tabular_features.shape) == 2:
                tabular_features = tabular_features.unsqueeze(1)
                
            if len(image_features.shape) == 2:
                image_features = image_features.unsqueeze(1)
                
            # 多模态编码
            y = self.encoder_multimodal(
                tabular_features=tabular_features,
                image_features=image_features
            )
            
            # 表格预测
            z = self.predictor_tabular(y)
            
            return z, y[:, 0, :]  # 返回预测和CLS token
            
        except Exception as e:
            print(f"\nError in forward_multimodal:")
            print(f"Tabular features shape: {tabular_features.shape}")
            print(f"Image features shape: {image_features.shape}")
            print(f"Error: {str(e)}")
            raise

    def forward_multimodal_feature(self, tabular_features: torch.Tensor, 
                                image_features: torch.Tensor) -> torch.Tensor:
        """获取多模态融合特征"""
        try:
            # 处理维度
            if len(tabular_features.shape) == 2:
                tabular_features = tabular_features.unsqueeze(1)
                
            if len(image_features.shape) == 2:
                image_features = image_features.unsqueeze(1)
                
            # 多模态编码
            y = self.encoder_multimodal(
                tabular_features=tabular_features,
                image_features=image_features
            )
            
            return y[:, 0, :]  # 返回CLS token
            
        except Exception as e:
            print(f"\nError in forward_multimodal_feature:")
            print(f"Tabular features shape: {tabular_features.shape}")
            print(f"Image features shape: {image_features.shape}")
            raise

    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        param_groups = [
            {
                'params': self.encoder_imaging.parameters(),
                'lr': self.hparams.lr_imaging,
                'weight_decay': self.hparams.weight_decay
            },
            {
                'params': self.projector_imaging.parameters()
            },
            {
                'params': self.encoder_tabular.parameters(),
                'lr': self.hparams.lr_tabular
            },
            {
                'params': self.projector_tabular.parameters()
            },
            {
                'params': self.encoder_multimodal.parameters()
            },
            {
                'params': self.predictor_tabular.parameters()
            },
            {
                'params': self.itm_head.parameters()
            }
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.95)  # 使用与MAE相同的beta值
        )

        return optimizer

    def on_train_start(self) -> None:
        """训练开始时的初始化"""
        self._epoch_start_printed = False
        self._epoch_header_printed = False

    def on_train_epoch_start(self) -> None:
        """每个epoch开始时的操作"""
        if not self._epoch_start_printed:
            print(f"\nEpoch {self.current_epoch}")
            if hasattr(self.hparams, 'pretrain') and self.hparams.pretrain.enabled:
                print("Using pretrained model")
            self._epoch_start_printed = True
            self._epoch_header_printed = True

    def on_train_epoch_end(self) -> None:
        """训练epoch结束时处理"""
        # 只记录预训练相关的loss
        metrics = {
            "train/loss": self.trainer.callback_metrics.get("train/loss", torch.tensor(0.0)),  
            "train/itc_loss": self.trainer.callback_metrics.get("train/itc_loss", torch.tensor(0.0)),
            "train/itm_loss": self.trainer.callback_metrics.get("train/itm_loss", torch.tensor(0.0)),
            "train/tr_loss": self.trainer.callback_metrics.get("train/tr_loss", torch.tensor(0.0))
        }
        
        self.log_dict(metrics)
        
        # 打印训练指标
        print(f"\nTraining Metrics:")
        print(f"Total Loss: {metrics['train/loss']:.4f}")
        print(f"ITC Loss: {metrics['train/itc_loss']:.4f}")
        print(f"ITM Loss: {metrics['train/itm_loss']:.4f}")
        print(f"MTR Loss: {metrics['train/tr_loss']:.4f}")
        self._epoch_start_printed = False

    def on_validation_epoch_start(self) -> None:
        """验证开始时重置epoch header标志"""
        if self._epoch_header_printed:
            self._epoch_header_printed = False

    def on_validation_epoch_end(self) -> None:
        """验证epoch结束时处理"""
        # 只记录预训练相关的loss
        metrics = {
            "val/loss": self.trainer.callback_metrics.get("val/loss", torch.tensor(0.0)),
            "val/itc_loss": self.trainer.callback_metrics.get("val/itc_loss", torch.tensor(0.0)),
            "val/itm_loss": self.trainer.callback_metrics.get("val/itm_loss", torch.tensor(0.0)),
            "val/tr_loss": self.trainer.callback_metrics.get("val/tr_loss", torch.tensor(0.0))
        }
        
        self.log_dict(metrics)
        
        # 打印验证指标
        print(f"\nValidation Metrics:")
        print(f"Total Loss: {metrics['val/loss']:.4f}")
        print(f"ITC Loss: {metrics['val/itc_loss']:.4f}")
        print(f"ITM Loss: {metrics['val/itm_loss']:.4f}")
        print(f"MTR Loss: {metrics['val/tr_loss']:.4f}")