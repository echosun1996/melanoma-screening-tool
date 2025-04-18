from typing import Tuple
import os
import sys

# 添加项目根目录到Python路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

import torch
import torchmetrics
import pytorch_lightning as pl
import numpy as np
from sklearn.metrics import roc_auc_score

try:
    from models.TabularModel import TabularModel
    from models.ImagingModel import ImagingModel
    from models.MultimodalModel import MultimodalModel
    from models.Tip_utils.Tip_downstream import TIPBackbone
    from models.Tip_utils.Tip_downstream_ensemble import TIPBackboneEnsemble
    from models.DAFT import DAFT
    from models.MultimodalModelMUL import MultimodalModelMUL
    from models.MultimodalModelTransformer import MultimodalModelTransformer
except ImportError as e:
    print(f"Warning: Some modules could not be imported. This is expected during testing. Error: {e}")

class PartialAUC(torchmetrics.Metric):
    """自定义 Partial AUC 指标"""
    def __init__(self, min_tpr=0.80):
        super().__init__()
        self.min_tpr = min_tpr
        self.max_fpr = abs(1 - min_tpr)
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("target", default=[], dist_reduce_fx=None)
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """更新状态
        Args:
            preds: 预测概率 [N] 或 [N, C]
            target: 目标标签 [N]
        """
        if preds.dim() > 1 and preds.shape[1] > 1:
            preds = preds[:, 1]  # 取正类的概率
        preds = preds.detach()
        target = target.detach()
        
        self.preds.append(preds)
        self.target.append(target)
    
    def compute(self):
        """计算 partial AUC"""
        if not self.preds or not self.target:
            return torch.tensor(0.0, device=self.preds[0].device if self.preds else 'cpu')
            
        try:
            # 将列表转换为张量
            all_preds = torch.cat(self.preds).detach().float()  # 确保是浮点数
            all_target = torch.cat(self.target).detach().long()  # 确保是整数
            
            # 处理和检查 NaN
            if torch.isnan(all_preds).any() or torch.isnan(all_target).any():
                print(f"Warning: NaN detected in pAUC computation")
                return torch.tensor(0.0, device=all_preds.device)
            
            # 转换为numpy并计算
            v_gt = all_target.cpu().numpy()
            v_pred = all_preds.cpu().numpy()
            
            # 标准化预测值到 [0,1] 范围
            v_pred = (v_pred - v_pred.min()) / (v_pred.max() - v_pred.min() + 1e-8)
            
            # 计算 partial AUC
            partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=self.max_fpr)
            partial_auc = (0.5 * self.max_fpr**2 + 
                        (self.max_fpr - 0.5 * self.max_fpr**2) / (1.0 - 0.5) * 
                        (partial_auc_scaled - 0.5))
                        
            return torch.tensor(partial_auc, device=all_preds.device)
            
        except Exception as e:
            print(f"Error computing pAUC: {str(e)}")
            if 'all_preds' in locals() and 'all_target' in locals():
                print(f"Predictions stats: min={all_preds.min():.4f}, max={all_preds.max():.4f}, "
                    f"has_nan={torch.isnan(all_preds).any()}")
                print(f"Target unique values: {torch.unique(all_target)}")
            return torch.tensor(0.0, device=self.preds[0].device)

    def reset(self):
        super().reset()
        self.preds.clear()
        self.target.clear()

class Evaluator(pl.LightningModule):
    """用于评估的模型类,特别针对ISIC2024数据集的极度不平衡问题"""
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        # 初始化模型
        if self.hparams.eval_datatype == 'imaging':
            self.model = ImagingModel(self.hparams)
        elif self.hparams.eval_datatype == 'multimodal':
            assert self.hparams.strategy == 'tip'
            if self.hparams.finetune_ensemble == True:
                self.model = TIPBackboneEnsemble(self.hparams)
            else:
                self.model = TIPBackbone(self.hparams)
        elif self.hparams.eval_datatype == 'tabular':
            self.model = TabularModel(self.hparams)
        elif self.hparams.eval_datatype == 'imaging_and_tabular':
            if self.hparams.algorithm_name == 'DAFT':
                self.model = DAFT(self.hparams)
            elif self.hparams.algorithm_name in set(['CONCAT','MAX']):
                if self.hparams.strategy == 'tip':
                    self.model = MultimodalModelTransformer(self.hparams)
                else:
                    self.model = MultimodalModel(self.hparams)
            elif self.hparams.algorithm_name == 'MUL':
                self.model = MultimodalModelMUL(self.hparams)

        # 设置评估指标
        task = 'binary' if self.hparams.num_classes == 2 else 'multiclass'
        
        # 准确率指标
        self.acc_train = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        self.acc_val = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        self.acc_test = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)

        # 标准AUC指标
        self.auc_train = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
        self.auc_val = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
        self.auc_test = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)

        # Partial AUC指标(关注高召回区域)
        self.pauc_train = PartialAUC(min_tpr=0.80)
        self.pauc_val = PartialAUC(min_tpr=0.80)
        self.pauc_test = PartialAUC(min_tpr=0.80)
        
        # 精确率、召回率和F1分数
        self.precision_train = torchmetrics.Precision(task=task, num_classes=self.hparams.num_classes)
        self.precision_val = torchmetrics.Precision(task=task, num_classes=self.hparams.num_classes)
        self.precision_test = torchmetrics.Precision(task=task, num_classes=self.hparams.num_classes)
        
        self.recall_train = torchmetrics.Recall(task=task, num_classes=self.hparams.num_classes)
        self.recall_val = torchmetrics.Recall(task=task, num_classes=self.hparams.num_classes)
        self.recall_test = torchmetrics.Recall(task=task, num_classes=self.hparams.num_classes)
        
        self.f1_train = torchmetrics.F1Score(task=task, num_classes=self.hparams.num_classes)
        self.f1_val = torchmetrics.F1Score(task=task, num_classes=self.hparams.num_classes)
        self.f1_test = torchmetrics.F1Score(task=task, num_classes=self.hparams.num_classes)
        
        # 混淆矩阵
        self.confusion_matrix_train = torchmetrics.ConfusionMatrix(task=task, num_classes=self.hparams.num_classes)
        self.confusion_matrix_val = torchmetrics.ConfusionMatrix(task=task, num_classes=self.hparams.num_classes)
        self.confusion_matrix_test = torchmetrics.ConfusionMatrix(task=task, num_classes=self.hparams.num_classes)

        # 损失函数(添加类别权重)
        pos_weight = torch.tensor([100.0]) if task == 'binary' else None  # 根据数据集不平衡程度调整
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight) if task == 'binary' else torch.nn.CrossEntropyLoss()

        # 记录最佳验证分数
        self.best_val_score = 0
        
        print(self.model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """生成预测"""
        y_hat = self.model(x)
        if len(y_hat.shape)==1:
            y_hat = torch.unsqueeze(y_hat, 0)
        return y_hat

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> None:
        """测试步骤"""
        x, y = batch 
        y_hat = self.forward(x)
        
        # 计算预测概率
        y_hat = torch.softmax(y_hat.detach(), dim=1)
        if self.hparams.num_classes==2:
            y_hat = y_hat[:,1]

        # 更新测试指标
        self.acc_test(y_hat, y)
        self.auc_test(y_hat, y)
        self.pauc_test(y_hat, y)
        self.precision_test(y_hat, y)
        self.recall_test(y_hat, y)
        self.f1_test(y_hat, y)
        self.confusion_matrix_test(y_hat, y)

    def test_epoch_end(self, _) -> None:
        """测试周期结束"""
        # 计算并记录所有测试指标
        metrics = {
            'test.acc': self.acc_test.compute(),
            'test.auc': self.auc_test.compute(),
            'test.pauc': self.pauc_test.compute(),
            'test.precision': self.precision_test.compute(),
            'test.recall': self.recall_test.compute(),
            'test.f1': self.f1_test.compute()
        }
        
        # 记录混淆矩阵
        conf_matrix = self.confusion_matrix_test.compute()
        
        # 记录所有指标
        for name, value in metrics.items():
            self.log(name, value)
            
        # 打印详细评估结果
        print("\nTest Results:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> torch.Tensor:
        """训练步骤"""
        x, y = batch
        
        # 前向传播
        y_hat = self.forward(x)
        
        # 计算损失
        loss = self.criterion(y_hat, y)
        
        # 计算预测概率
        y_hat = torch.softmax(y_hat.detach(), dim=1)
        if self.hparams.num_classes==2:
            y_hat = y_hat[:,1]
            
        # 更新训练指标
        self.acc_train(y_hat, y)
        self.auc_train(y_hat, y)
        self.pauc_train(y_hat, y)
        self.precision_train(y_hat, y)
        self.recall_train(y_hat, y)
        self.f1_train(y_hat, y)
        
        # 记录损失
        self.log('eval.train.loss', loss, on_epoch=True, on_step=False)
        
        return loss

    def training_epoch_end(self, _) -> None:
        """训练周期结束"""
        # 记录训练指标
        metrics = {
            'eval.train.acc': self.acc_train,
            'eval.train.auc': self.auc_train,
            'eval.train.pauc': self.pauc_train,
            'eval.train.precision': self.precision_train,
            'eval.train.recall': self.recall_train,
            'eval.train.f1': self.f1_train
        }
        
        for name, metric in metrics.items():
            self.log(name, metric, on_epoch=True, on_step=False)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> torch.Tensor:
        """验证步骤"""
        x, y = batch

        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)

        y_hat = torch.softmax(y_hat.detach(), dim=1)
        if self.hparams.num_classes==2:
            y_hat = y_hat[:,1]

        # 更新验证指标
        self.acc_val(y_hat, y)
        self.auc_val(y_hat, y)
        self.pauc_val(y_hat, y)
        self.precision_val(y_hat, y)
        self.recall_val(y_hat, y)
        self.f1_val(y_hat, y)
        
        self.log('eval.val.loss', loss, on_epoch=True, on_step=False)
    
    def validation_epoch_end(self, _) -> None:
        """验证周期结束"""
        if self.trainer.sanity_checking:
            return  

        # 计算所有验证指标
        metrics = {
            'eval.val.acc': self.acc_val.compute(),
            'eval.val.auc': self.auc_val.compute(),
            'eval.val.pauc': self.pauc_val.compute(),
            'eval.val.precision': self.precision_val.compute(),
            'eval.val.recall': self.recall_val.compute(),
            'eval.val.f1': self.f1_val.compute()
        }
        
        # 记录所有指标
        for name, value in metrics.items():
            self.log(name, value, on_epoch=True, on_step=False)

        # 基于pAUC更新最佳分数
        epoch_pauc_val = metrics['eval.val.pauc']
        self.best_val_score = max(self.best_val_score, epoch_pauc_val)

        # 重置验证指标
        self.acc_val.reset()
        self.auc_val.reset()
        self.pauc_val.reset()
        self.precision_val.reset()
        self.recall_val.reset()
        self.f1_val.reset()

    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.hparams.lr_eval, 
            weight_decay=self.hparams.weight_decay_eval
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',  # 使用pAUC作为指标
            patience=int(10/self.hparams.check_val_every_n_epoch),
            min_lr=self.hparams.lr*0.0001,
            factor=0.5,  # 每次调整降低50%
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'eval.val.pauc',  # 使用pAUC作为学习率调整的指标
                'interval': 'epoch',
                'frequency': 1
            }
        }