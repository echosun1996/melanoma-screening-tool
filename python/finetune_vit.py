
import os
import sys
import time
import math
import warnings
from typing import Dict, List, Tuple
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
# from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar  # or RichProgressBar
import torchmetrics
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, confusion_matrix
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

# 导入修改后的TIP模型
from models.TipModel3LossISIC512 import TIP3LossISIC

# 修改后的TBPContrastiveDataset类
class TBPContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self,
        data_path_imaging: str,
        data_path_tabular: str, 
        field_lengths_tabular: str,
        labels_path: str,
        img_size: int = 224,
        augmentation_rate: float = 0.5,
        corruption_rate: float = 0.15,
        replace_random_rate: float = 0.3,
        replace_special_rate: float = 0.3,
        one_hot_tabular: bool = False,
        missing_tabular: bool = False,
        missing_strategy: str = 'value',
        missing_rate: float = 0.0, 
        mask_path: str = None) -> None:
        """初始化TBP对比学习数据集"""
        super().__init__()
        
        # 定义特征列表 (移除了 'dnn_lesion_confidence' 和 'nevi_confidence')
        self.numerical_features = [
            'A', 'Aext', 'B', 'Bext', 'C', 'Cext', 'H', 'Hext', 'L', 'Lext', 
            'areaMM2', 'area_perim_ratio', 'color_std_mean', 'deltaA', 'deltaB', 
            'deltaL', 'deltaLB', 'deltaLBnorm', 'eccentricity', 
            'majorAxisMM', 'minorAxisMM', 'norm_border', 'norm_color', 
            'perimeterMM', 'radial_color_std_max', 'stdL', 'stdLExt', 'symm_2axis', 
            'symm_2axis_angle', 'age'
        ]
        
        self.categorical_features = ['location_simple', 'Sex']
        
        # 基本参数设置
        self.img_size = img_size
        self.augmentation_rate = augmentation_rate
        self.corruption_rate = corruption_rate
        self.replace_random_rate = replace_random_rate
        self.replace_special_rate = replace_special_rate
        self.one_hot_tabular = one_hot_tabular
        self.missing_tabular = missing_tabular
        self.num_cat = len(self.categorical_features)
        self.num_con = len(self.numerical_features)
        
        # 标准化均值和标准差 (ImageNet预训练值)
        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]
        
        try:
            # 加载数据
            self.image_paths = torch.load(data_path_imaging)
            df = pd.read_csv(data_path_tabular, low_memory=False)
            self.field_lengths = torch.load(field_lengths_tabular)
            labels_np = torch.load(labels_path)
            self.labels = torch.from_numpy(labels_np).long() if isinstance(labels_np, np.ndarray) else labels_np.long()
            
            # 处理分类特征
            for cat_feat in self.categorical_features:
                if cat_feat in df.columns:
                    df[cat_feat] = df[cat_feat].fillna(df[cat_feat].mode().iloc[0])
                    df[cat_feat] = pd.Categorical(df[cat_feat].astype(str)).codes
                else:
                    print(f"Warning: Categorical feature '{cat_feat}' not found in dataframe")
            
            # 处理连续特征
            available_num_features = [feat for feat in self.numerical_features if feat in df.columns]
            if len(available_num_features) < len(self.numerical_features):
                missing_features = set(self.numerical_features) - set(available_num_features)
                print(f"Warning: The following numerical features are missing: {missing_features}")
                self.numerical_features = available_num_features
            
            df[self.numerical_features] = df[self.numerical_features].fillna(
                df[self.numerical_features].mean()
            )
            
            # 转换为numpy数组
            self.numerical_data = df[self.numerical_features].values.astype(np.float32)
            self.categorical_data = df[self.categorical_features].values.astype(np.int64)
            
            # 组合特征
            self.tabular_data = np.concatenate([
                self.categorical_data,
                self.numerical_data
            ], axis=1)
            
            # 处理缺失值掩码
            if missing_tabular and mask_path:
                if os.path.exists(mask_path):
                    self.missing_mask = np.load(mask_path)
                    print(f"Loaded missing mask from {mask_path}, shape: {self.missing_mask.shape}")
                else:
                    print(f"Warning: Missing mask file {mask_path} not found")
                    self.missing_mask = None
            else:
                self.missing_mask = None
                
            # 生成特征分布
            self.generate_marginal_distributions()
                
            # 设置图像增强
            self.default_transform = A.Compose([
                A.Resize(height=self.img_size, width=self.img_size),
                A.Normalize(
                    mean=self.normalize_mean,
                    std=self.normalize_std
                ),
                ToTensorV2()
            ])
            
            self.transform = A.Compose([
                A.RandomResizedCrop(
                    height=self.img_size,
                    width=self.img_size,
                    scale=(0.2, 1.0),
                    ratio=(0.75, 1.3333333333333333)
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.OneOf([
                    A.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                        hue=0.1,
                        p=0.8
                    ),
                    A.ToGray(p=0.2)
                ], p=0.8),
                A.GaussianBlur(sigma_limit=[.1, 2.], p=0.5),
                A.Normalize(
                    mean=self.normalize_mean,
                    std=self.normalize_std
                ),
                ToTensorV2()
            ])
            
            print(f"Dataset initialized successfully:")
            print(f"Total samples: {len(self)}")
            print(f"Numerical features: {len(self.numerical_features)}")
            print(f"Categorical features: {len(self.categorical_features)}")
            print(f"Image size: {self.img_size}x{self.img_size}")
            print(f"Tabular data shape: {self.tabular_data.shape}")
            
        except Exception as e:
            print(f"Error initializing dataset: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def __len__(self) -> int:
        return len(self.image_paths)
    
    def generate_imaging_views(self, index: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """生成两个图像视图和原始图像"""
        try:
            # 加载图像
            img_path = self.image_paths[index]
            
            # 检查文件是否存在
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")
                
            # 尝试加载NumPy文件
            try:
                img = np.load(img_path)
            except:
                # 如果是PNG文件，使用PIL加载
                from PIL import Image
                img = np.array(Image.open(img_path))
            
            # 生成视图
            imaging_views = []
            
            # 第一个视图一定使用增强
            view1 = self.transform(image=img)['image']
            imaging_views.append(view1)
            
            # 第二个视图根据概率使用增强
            if random.random() < self.augmentation_rate:
                view2 = self.transform(image=img)['image']
            else:
                view2 = self.default_transform(image=img)['image']
            imaging_views.append(view2)
            
            # 未增强的原始图像
            unaugmented = self.default_transform(image=img)['image']
            
            return imaging_views, unaugmented
            
        except Exception as e:
            print(f"Error in generate_imaging_views: {str(e)}")
            print(f"Image path: {self.image_paths[index]}")
            raise

    def generate_marginal_distributions(self) -> None:
        """生成特征的经验边缘分布"""
        try:
            # 处理分类特征
            self.marginal_distributions_cat = []
            for i in range(self.num_cat):
                unique_vals = np.unique(self.categorical_data[:, i])
                valid_vals = unique_vals[~np.isnan(unique_vals)]
                self.marginal_distributions_cat.append(valid_vals)
                
            # 处理连续特征
            self.marginal_distributions_num = []
            for i in range(self.num_con):
                vals = self.numerical_data[:, i]
                valid_vals = vals[~np.isnan(vals)]
                self.marginal_distributions_num.append(valid_vals)
                
        except Exception as e:
            print(f"Error generating marginal distributions: {str(e)}")
            raise

    def corrupt_tabular(self, subject: np.ndarray) -> np.ndarray:
        """损坏表格特征用于对比学习"""
        try:
            subject = subject.copy()
            
            if self.corruption_rate == 0:
                return subject
                
            num_features = len(subject)
            num_corrupt = int(num_features * self.corruption_rate)
            corrupt_indices = random.sample(range(num_features), num_corrupt)
            
            for idx in corrupt_indices:
                if self.missing_tabular and self.missing_mask is not None:
                    valid_mask = ~self.missing_mask[:, idx]
                else:
                    valid_mask = slice(None)
                
                if idx < self.num_cat:
                    valid_values = self.marginal_distributions_cat[idx][valid_mask]
                    if len(valid_values) > 0:
                        subject[idx] = np.random.choice(valid_values)
                else:
                    num_idx = idx - self.num_cat
                    valid_values = self.marginal_distributions_num[num_idx][valid_mask]
                    if len(valid_values) > 0:
                        subject[idx] = np.random.choice(valid_values)
            
            return subject
            
        except Exception as e:
            print(f"Error in corrupt_tabular: {str(e)}")
            raise

    def mask_tabular(self, subject: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """创建表格特征掩码和重建视图"""
        try:
            subject = subject.copy()
            
            if self.replace_random_rate + self.replace_special_rate == 0:
                mask = np.zeros(len(subject), dtype=bool)
                mask_random = np.zeros(len(subject), dtype=bool)
                mask_special = np.zeros(len(subject), dtype=bool)
                return subject, mask, mask_special, mask_random

            # 计算掩码数量
            num_features = len(subject)
            total_mask = int(num_features * (self.replace_random_rate + self.replace_special_rate))
            
            if self.replace_random_rate == 0:
                num_random = 0
                num_special = total_mask
            elif self.replace_special_rate == 0:
                num_random = total_mask
                num_special = 0
            else:
                num_random = int(total_mask * self.replace_random_rate / 
                            (self.replace_random_rate + self.replace_special_rate))
                num_special = total_mask - num_random
            
            # 创建掩码
            mask = np.zeros(num_features, dtype=bool)
            mask_random = np.zeros(num_features, dtype=bool)
            mask_special = np.zeros(num_features, dtype=bool)
            
            # 应用随机替换掩码
            if num_random > 0:
                random_indices = random.sample(range(num_features), num_random)
                for idx in random_indices:
                    mask[idx] = True
                    mask_random[idx] = True
                    
                    if self.missing_tabular and self.missing_mask is not None:
                        valid_mask = ~self.missing_mask[:, idx]
                    else:
                        valid_mask = slice(None)
                        
                    if idx < self.num_cat:
                        valid_values = self.marginal_distributions_cat[idx][valid_mask]
                        if len(valid_values) > 0:
                            subject[idx] = np.random.choice(valid_values)
                    else:
                        num_idx = idx - self.num_cat
                        valid_values = self.marginal_distributions_num[num_idx][valid_mask]
                        if len(valid_values) > 0:
                            subject[idx] = np.random.choice(valid_values)
            
            # 应用特殊token掩码
            if num_special > 0:
                remaining_indices = [i for i in range(num_features) if not mask[i]]
                special_indices = random.sample(remaining_indices, num_special)
                for idx in special_indices:
                    mask[idx] = True
                    mask_special[idx] = True
            
            return subject, mask, mask_special, mask_random
            
        except Exception as e:
            print(f"Error in mask_tabular: {str(e)}")
            raise

    def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取单个数据样本"""
        try:
            # 生成图像视图
            imaging_views, unaugmented_image = self.generate_imaging_views(index)
            
            # 处理表格特征
            if self.corruption_rate > 0:
                tabular_views = [torch.tensor(self.corrupt_tabular(self.tabular_data[index]), dtype=torch.float)]
            else:
                tabular_views = [torch.tensor(self.tabular_data[index], dtype=torch.float)]
                
            # 生成掩码视图
            masked_view, mask, mask_special, mask_random = self.mask_tabular(self.tabular_data[index])
            tabular_views.append(torch.from_numpy(masked_view).float())
            tabular_views.append(torch.from_numpy(mask))
            tabular_views.append(torch.from_numpy(mask_special))
            
            # 获取标签和原始视图
            label = self.labels[index]
            unaugmented_tabular = torch.tensor(self.tabular_data[index], dtype=torch.float)
            
            return imaging_views, tabular_views, label, unaugmented_image, unaugmented_tabular
            
        except Exception as e:
            print(f"Error in __getitem__: {str(e)}")
            print(f"Index: {index}")
            raise

    def get_input_size(self) -> int:
        """返回输入特征维度"""
        if self.one_hot_tabular:
            return int(sum(self.field_lengths))
        return len(self.numerical_features) + len(self.categorical_features)

# 评估指标 - 部分AUC
class PartialAUC(torchmetrics.Metric):
    """计算部分AUC (pAUC)，带有最小TPR阈值"""
    def __init__(self, min_tpr: float = 0.8):
        super().__init__()
        self.min_tpr = min_tpr
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.preds.append(preds.detach())
        self.target.append(target.detach())
        
    def compute(self) -> torch.Tensor:
        try:
            preds = torch.cat(self.preds)
            target = torch.cat(self.target)
            
            if len(preds) == 0 or len(target) == 0:
                return torch.tensor(0.0, device=self.device)
            
            v_gt = target.cpu().numpy()
            v_pred = preds.cpu().numpy()
            
            max_fpr = 1.0 - self.min_tpr
            
            # 计算ROC曲线
            fpr, tpr, _ = roc_curve(v_gt, v_pred)
            
            if len(fpr) < 2 or len(tpr) < 2:
                return torch.tensor(0.0, device=self.device)
                
            # 线性插值
            stop = np.searchsorted(fpr, max_fpr, "right")
            if stop >= len(fpr):
                stop = len(fpr) - 1
            if stop == 0:
                return torch.tensor(0.0, device=self.device)
                
            x_interp = [fpr[stop - 1], fpr[stop]]
            y_interp = [tpr[stop - 1], tpr[stop]]
            tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
            fpr = np.append(fpr[:stop], max_fpr)
            
            partial_auc = auc(fpr, tpr)
            return torch.tensor(partial_auc, device=self.device)
            
        except Exception as e:
            print(f"Error in pAUC computation: {str(e)}")
            traceback.print_exc()
            return torch.tensor(0.0, device=self.device)

# 自定义进度条
class CustomProgressBar(TQDMProgressBar):
    """显示验证指标的自定义进度条"""
    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        
        if trainer.training:
            metrics = trainer.callback_metrics
            
            # 训练指标
            if "train_loss" in metrics:
                items["loss"] = f"{metrics['train_loss']:.3f}"
            if "train_pauc" in metrics:
                items["pauc"] = f"{metrics['train_pauc']:.4f}"
            if "train_auroc" in metrics:
                items["auroc"] = f"{metrics['train_auroc']:.3f}"
                
            # 验证指标
            if "val_loss" in metrics:
                items["val_loss"] = f"{metrics['val_loss']:.3f}"
            if "val_pauc" in metrics:
                items["val_pauc"] = f"{metrics['val_pauc']:.4f}"
            if "val_auroc" in metrics:
                items["val_auroc"] = f"{metrics['val_auroc']:.3f}"
                
        return items

# 焦点损失 (Focal Loss) 用于处理类别不平衡
class FocalLoss(nn.Module):
    """用于处理不平衡分类的焦点损失"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets.float(), reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

# Mixup增强层
class MixupLayer:
    """Mixup数据增强"""
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        
    def __call__(self, x, y):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

# TIP微调模型
class TIPFineTuneModel(pl.LightningModule):
    """用于TBP筛查的TIP模型微调"""
    def __init__(self, pretrained_model: TIP3LossISIC, config: Dict):
        super().__init__()
        self.save_hyperparameters(config)
        
        # 从预训练模型获取编码器
        self.encoder_imaging = pretrained_model.encoder_imaging
        self.encoder_tabular = pretrained_model.encoder_tabular
        self.encoder_multimodal = pretrained_model.encoder_multimodal
        
        # 预处理层
        self.pre_norm = nn.LayerNorm(768) 
        self.post_norm = nn.LayerNorm(768)
        
        # 分类头
        self.classifier = nn.Linear(768, 1)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.classifier.bias.data.fill_(0.0)
        
        # 辅助分类器
        self.aux_classifier_img = nn.Linear(768, 1)
        self.aux_classifier_tab = nn.Linear(768, 1)
        
        # 损失函数
        self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        self.mixup = MixupLayer(alpha=0.2)
        
        # 指标
        metrics = ['train', 'val']
        for prefix in metrics:
            setattr(self, f'{prefix}_auroc', torchmetrics.AUROC(task='binary'))
            setattr(self, f'{prefix}_pauc', PartialAUC(min_tpr=0.8))
            setattr(self, f'{prefix}_precision', torchmetrics.Precision(task='binary'))
            setattr(self, f'{prefix}_recall', torchmetrics.Recall(task='binary'))
            
        self._print_trainable_params()
        
    def _print_trainable_params(self):
        """打印可训练参数统计"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"\n可训练参数: {trainable:,} / {total:,}")
            
    def forward(self, x_img: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        try:
            # 图像特征
            img_features = self.encoder_imaging(x_img)
            if isinstance(img_features, list):
                img_features = img_features[-1]
                
            # 获取CLS token并规范化
            img_features = self.pre_norm(img_features)
            img_features = img_features[:, 0]  # [B, 768]
            
            # 表格特征
            tab_features = self.encoder_tabular(x_tab)  # [B, N+1, 768]
            tab_features = self.pre_norm(tab_features)
                
            # 多模态融合
            multimodal_features = self.encoder_multimodal(
                tabular_features=tab_features,
                image_features=img_features.unsqueeze(1)  # [B, 1, 768]
            )
            
            # 后处理
            multimodal_features = self.post_norm(multimodal_features)
            cls_token = multimodal_features[:, 0]  # [B, 768]
            
            # 预测
            logits = self.classifier(cls_token)  # [B, 1]
            
            return logits.squeeze(-1)  # [B]
                
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            traceback.print_exc()
            raise
            
    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        """训练步骤，使用mixup和辅助损失"""
        try:
            # 解包批次数据
            x_img, x_tab, y = batch

            # 以50%的概率应用mixup增强
            if random.random() < 0.5:
                # Mixup图像并获取混合标签
                x_img_mixed, y_a, y_b, lam = self.mixup(x_img, y)
                x_img = x_img_mixed
            else:
                # 不使用mixup，使用原始标签
                y_a = y_b = y
                lam = 1.0

            # 提取图像和表格特征
            img_features = self.encoder_imaging(x_img)
            if isinstance(img_features, list):
                img_features = img_features[-1]
                img_cls = img_features[:, 0]  # 获取CLS token
            else:
                B, C, H, W = img_features.shape
                img_features = img_features.view(B, C, -1).permute(0, 2, 1)
                img_cls = img_features.mean(dim=1)  # 全局池化
                
            # 获取表格特征
            tab_features = self.encoder_tabular(x_tab)  # [B, N+1, 768]
            tab_cls = tab_features[:, 0]  # 获取CLS token

            # 获取主要多模态预测
            multimodal_features = self.encoder_multimodal(
                tabular_features=tab_features,
                image_features=img_features.unsqueeze(1) if len(img_features.shape) == 2 else img_features
            )
            multimodal_cls = multimodal_features[:, 0]  # 获取CLS token
            logits = self.classifier(multimodal_cls)
            logits = logits.squeeze(-1)

            # 使用mixup计算主要损失
            main_loss = lam * self.criterion(logits, y_a) + (1 - lam) * self.criterion(logits, y_b)

            # 计算图像和表格分支的辅助损失
            aux_logits_img = self.aux_classifier_img(img_cls).squeeze(-1)
            aux_logits_tab = self.aux_classifier_tab(tab_cls).squeeze(-1)
            
            aux_loss = lam * (
                self.criterion(aux_logits_img, y_a) + 
                self.criterion(aux_logits_tab, y_a)
            ) + (1 - lam) * (
                self.criterion(aux_logits_img, y_b) + 
                self.criterion(aux_logits_tab, y_b)
            )

            # 合并损失 (主损失 + 0.4 * 辅助损失)
            loss = main_loss + 0.4 * aux_loss

            # 计算并记录指标
            with torch.no_grad():
                probs = torch.sigmoid(logits.detach())
                
                # 更新指标
                self.train_auroc(probs, y)
                self.train_pauc(probs, y)
                self.train_precision(probs, y)
                self.train_recall(probs, y)

                # 打印第一个批次信息用于调试
                if batch_idx == 0 and not hasattr(self, '_printed_batch_info'):
                    print(f"\n批次统计:")
                    print(f"图像形状: {x_img.shape}")
                    print(f"表格形状: {x_tab.shape}")
                    print(f"标签形状: {y.shape}")
                    print(f"正样本: {y.sum()}/{len(y)}")
                    self._printed_batch_info = True

            # 记录所有指标
            self.log_dict({
                'train_loss': loss,
                'train_loss_main': main_loss,
                'train_loss_aux': aux_loss,
                'train_auroc': self.train_auroc,
                'train_pauc': self.train_pauc,
                'train_precision': self.train_precision,
                'train_recall': self.train_recall
            }, prog_bar=True)

            return loss

        except Exception as e:
            print(f"\n训练步骤错误 (batch {batch_idx}):")
            print(f"输入形状:")
            print(f"  图像: {x_img.shape if 'x_img' in locals() else '未创建'}")
            print(f"  表格: {x_tab.shape if 'x_tab' in locals() else '未创建'}")
            print(f"  标签: {y.shape if 'y' in locals() else '未创建'}")
            print(f"错误: {str(e)}")
            traceback.print_exc()
            raise
            
    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> Dict:
        """验证步骤"""
        try:
            x_img, x_tab, y = batch
            
            # 前向传播
            logits = self(x_img, x_tab)
            loss = self.criterion(logits, y.float())
            
            # 计算概率
            probs = torch.sigmoid(logits)
            
            # 更新指标
            self.val_auroc(probs, y)
            self.val_pauc(probs, y)
            self.val_precision(probs, y)
            self.val_recall(probs, y)
            
            # 记录指标
            self.log_dict({
                'val_loss': loss,
                'val_auroc': self.val_auroc,
                'val_pauc': self.val_pauc,
                'val_precision': self.val_precision,
                'val_recall': self.val_recall
            }, prog_bar=True)
            
            return {
                'val_loss': loss,
                'predictions': probs.detach(),
                'targets': y.detach()
            }
            
        except Exception as e:
            print(f"验证步骤错误: {str(e)}")
            traceback.print_exc()
            return None
            
    def configure_optimizers(self):
        """配置具有分层学习率的优化器"""
        # 不同组件的不同学习率
        layer_groups = [
            {
                'params': self.encoder_imaging.parameters(),
                'lr': self.hparams.lr * 0.1  # 预训练视觉编码器使用较低的学习率
            },
            {
                'params': self.encoder_tabular.parameters(),
                'lr': self.hparams.lr
            },
            {
                'params': self.encoder_multimodal.parameters(),
                'lr': self.hparams.lr
            },
            {
                'params': list(self.classifier.parameters()) + 
                         list(self.aux_classifier_img.parameters()) + 
                         list(self.aux_classifier_tab.parameters()),
                'lr': self.hparams.lr * 10  # 分类头使用较高的学习率
            }
        ]
        
        # AdamW优化器
        optimizer = torch.optim.AdamW(
            layer_groups,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        # 学习率调度器，带预热
        num_training_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = min(2000, num_training_steps // 10)
        
        scheduler = {
            'scheduler': get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            ),
            'interval': 'step',
            'frequency': 1
        }
            
        return [optimizer], [scheduler]

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1
):
    """创建带预热和余弦衰减的学习率调度"""
    def lr_lambda(current_step):
        # 预热阶段
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # 余弦衰减阶段
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def robust_collate_fn(batch: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """健壮的数据收集函数"""
    try:
        valid_samples = []
        valid_labels = []
        
        for i, sample in enumerate(batch):
            try:
                # 解包样本
                image_views, tabular_views, label, orig_img, orig_tab = sample
                
                # 验证检查
                if orig_img is None or orig_tab is None:
                    print(f"警告: 样本 {i} 有None值")
                    continue
                    
                if not (torch.is_tensor(orig_img) and torch.is_tensor(orig_tab)):
                    print(f"警告: 样本 {i} 有非张量值")
                    continue
                    
                if torch.isnan(orig_img).any() or torch.isnan(orig_tab).any():
                    print(f"警告: 样本 {i} 有NaN值")
                    continue
                    
                # 转换为float32
                orig_img = orig_img.float()
                orig_tab = orig_tab.float()
                
                valid_samples.append((orig_img, orig_tab))
                valid_labels.append(label)
                
            except Exception as e:
                print(f"处理样本 {i} 错误: {str(e)}")
                continue
                
        if not valid_samples:
            raise ValueError("批次中没有有效样本")
            
        # 堆叠张量
        images, tabulars = zip(*valid_samples)
        images = torch.stack(images)
        tabulars = torch.stack(tabulars)
        labels = torch.tensor(valid_labels, dtype=torch.float32)
        
        return images, tabulars, labels
        
    except Exception as e:
        print(f"收集函数错误: {str(e)}")
        traceback.print_exc()
        raise

class Config:
    """TIP ViT在TBP Screening上微调的配置类"""
    def __init__(self):
        # 基本设置
        self.seed = 42
        self.gpu_id = 1  # 使用cuda:2

        # 路径
        self.pretrained_path = '/mnt/hdd/sdc/ysheng/TIP/results/isic/0328_1327/best_model_epoch=233.ckpt'
        self.data_base = '/mnt/hdd/sdc/ysheng/TBP_Screening/hop_mym_processed_data'
        self.save_dir = '/mnt/hdd/sdc/ysheng/TBP_Screening/tip_finetune_results'
        
        # 数据路径
        self.data_train_imaging = 'tbp_train_numpy_paths.pt'
        self.data_val_imaging = 'tbp_val_numpy_paths.pt'
        self.data_train_tabular = 'tbp_features_train.csv'
        self.data_val_tabular = 'tbp_features_val.csv'
        self.field_lengths_tabular = 'tabular_lengths.pt'
        self.labels_train_path = 'tbp_labels_train.pt'
        self.labels_val_path = 'tbp_labels_val.pt'
        
        # 模型设置
        self.multimodal_embedding_dim = 768
        self.img_size = 224  # TBP图像大小为224x224
        
        # 训练设置
        self.lr = 1e-4
        self.weight_decay = 0.001
        self.batch_size = 128
        self.max_epochs = 100
        self.num_workers = 4

def main():
    """主训练函数"""
    warnings.filterwarnings('ignore')
    
    # 创建配置
    config = Config()
    
    try:
        start_time = time.time()
        print(f"\n开始TIP ViT-BASE在TBP Screening上的微调...")
        
        # 设置GPU
        if not torch.cuda.is_available():
            raise RuntimeError("需要CUDA")
            
        device = torch.device(f"cuda:{config.gpu_id}")
        print(f"使用GPU {config.gpu_id}: {torch.cuda.get_device_name(config.gpu_id)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(config.gpu_id).total_memory/1024**3:.1f}GB")
            
        # 设置随机种子
        pl.seed_everything(config.seed, workers=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # 创建保存目录
        os.makedirs(config.save_dir, exist_ok=True)
        


        # 创建具有修改后特征的数据集（移除了dnn_lesion_confidence和nevi_confidence）
        print("\n创建修改后的TBP数据集...")
        train_dataset = TBPContrastiveDataset(
            data_path_imaging=os.path.join(config.data_base, config.data_train_imaging),
            data_path_tabular=os.path.join(config.data_base, config.data_train_tabular),
            field_lengths_tabular=os.path.join(config.data_base, config.field_lengths_tabular),
            labels_path=os.path.join(config.data_base, config.labels_train_path),
            img_size=config.img_size,
            augmentation_rate=0.8,
            corruption_rate=0.0,
            one_hot_tabular=False,
            missing_tabular=False
        )
        
        val_dataset = TBPContrastiveDataset(
            data_path_imaging=os.path.join(config.data_base, config.data_val_imaging),
            data_path_tabular=os.path.join(config.data_base, config.data_val_tabular),
            field_lengths_tabular=os.path.join(config.data_base, config.field_lengths_tabular),
            labels_path=os.path.join(config.data_base, config.labels_val_path),
            img_size=config.img_size,
            augmentation_rate=0.0,  # 验证不使用增强
            corruption_rate=0.0,
            one_hot_tabular=False,
            missing_tabular=False
        )

        print("\n数据集大小:")
        print(f"训练集: {len(train_dataset)} 样本")
        print(f"验证集: {len(val_dataset)} 样本")

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
            collate_fn=robust_collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size * 2,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
            collate_fn=robust_collate_fn
        )

        print("\n数据加载器批次数:")
        print(f"训练: {len(train_loader)} 批次")
        print(f"验证: {len(val_loader)} 批次")

        # 初始化模型
        print(f"\n加载预训练模型: {config.pretrained_path}")
        pretrained_model = TIP3LossISIC.load_from_checkpoint(
            config.pretrained_path,
            strict=False
        )

        model = TIPFineTuneModel(
            pretrained_model=pretrained_model,
            config={
                'lr': config.lr,
                'weight_decay': config.weight_decay,
                'multimodal_embedding_dim': config.multimodal_embedding_dim
            }
        )


        # 设置回调函数
        callbacks = [
            ModelCheckpoint(
                monitor='val_pauc',
                dirpath=os.path.join(config.save_dir, 'checkpoints'),
                filename='best-model',
                save_top_k=1,
                mode='max',
                save_last=True,
                verbose=True
            ),
            EarlyStopping(
                monitor='val_pauc',
                patience=10,
                min_delta=0.0002,
                mode='max',
                verbose=True
            ),
            CustomProgressBar()
        ]

        # 配置训练器
        trainer = Trainer(
            max_epochs=config.max_epochs,
            callbacks=callbacks,
            accelerator='gpu',
            devices=[config.gpu_id],
            precision=32,
            gradient_clip_val=1.0,
            deterministic=True,
            check_val_every_n_epoch=1,
            num_sanity_val_steps=2,
            enable_progress_bar=True,
            enable_checkpointing=True,
            default_root_dir=os.path.join(config.save_dir, 'logs')
        )

        # 训练模型
        print("\n开始训练...")
        trainer.fit(model, train_loader, val_loader)

        # 加载最佳模型
        best_model_path = trainer.checkpoint_callback.best_model_path
        print(f"\n最佳模型保存在: {best_model_path}")
        best_model = TIPFineTuneModel.load_from_checkpoint(
            best_model_path,
            pretrained_model=pretrained_model,
            config={
                'lr': config.lr,
                'weight_decay': config.weight_decay,
                'multimodal_embedding_dim': config.multimodal_embedding_dim
            }
        )
        best_model.eval()
        best_model.to(device)

        # 生成验证预测
        print("\n生成验证预测...")
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="验证中"):
                x_img, x_tab, y = batch
                x_img = x_img.to(device)
                x_tab = x_tab.to(device)
                
                logits = best_model(x_img, x_tab)
                probs = torch.sigmoid(logits)
                
                val_predictions.extend(probs.cpu().numpy())
                val_targets.extend(y.cpu().numpy())

        # 保存验证预测
        val_results = pd.DataFrame({
            'prediction': val_predictions,
            'true_label': val_targets
        })
        
        val_results_path = os.path.join(config.save_dir, 'val_predictions.csv')
        val_results.to_csv(val_results_path, index=False)
        print(f"验证预测已保存到 {val_results_path}")
        
        # 计算验证指标
        fpr, tpr, thresholds = roc_curve(val_targets, val_predictions)
        auroc = auc(fpr, tpr)
        
        # 计算部分AUC (pAUC)
        min_tpr = 0.8
        max_fpr = 1.0 - min_tpr
        
        stop = np.searchsorted(fpr, max_fpr, "right")
        partial_tpr = np.append(tpr[:stop], np.interp(max_fpr, [fpr[stop-1], fpr[stop]], [tpr[stop-1], tpr[stop]]))
        partial_fpr = np.append(fpr[:stop], max_fpr)
        pauc = auc(partial_fpr, partial_tpr)
        
        # 计算最优阈值
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        # 计算最优阈值下的精确率和召回率
        binary_preds = (np.array(val_predictions) >= optimal_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(val_targets, binary_preds).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # 打印验证指标
        print("\n验证指标:")
        print(f"AUROC: {auroc:.4f}")
        print(f"pAUC: {pauc:.4f}")
        print(f"最优阈值: {optimal_threshold:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"混淆矩阵:")
        print(f"  真负例: {tn}")
        print(f"  假正例: {fp}")
        print(f"  假负例: {fn}")
        print(f"  真正例: {tp}")
        

        # 打印运行时间
        end_time = time.time()
        hours, remainder = divmod(end_time - start_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f'\n总运行时间: {int(hours)}h {int(minutes)}m {int(seconds)}s')
        
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        traceback.print_exc()
        
        # 记录错误
        error_path = os.path.join(config.save_dir, 'error_log.txt')
        with open(error_path, 'w') as f:
            traceback.print_exc(file=f)
        print(f"错误日志已保存到 {error_path}")
        
        sys.exit(1)
        
    finally:
        # 清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()

if __name__ == '__main__':
    main()