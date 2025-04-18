from python.models.TipModel3LossISIC512 import TIP3LossISIC
from finetune_vit import TIPFineTuneModel
# Melanoma Analysis Backend
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Start....")
# Load the pretrained model structure
pretrained_model = TIP3LossISIC.load_from_checkpoint(
    'checkpoint/best_model_epoch.ckpt',
    strict=False,
)

# Load finetuned model
model = TIPFineTuneModel.load_from_checkpoint(
    'checkpoint/best-model-v1.ckpt',
    pretrained_model=pretrained_model,
    # config={
    #     'lr': 1e-4,
    #     'weight_decay': 0.001,
    #     'multimodal_embedding_dim': 768
    # },
)

# Set to evaluation mode and move to device
model.eval()
model = model.to(DEVICE)

print("Load")