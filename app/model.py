import torchvision
import torch.nn as nn

def create_vitb16_model(
  num_classes: int,
):
  vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
  vit_model = torchvision.models.vit_b_16(weights=vit_weights)
  vit_transform = vit_weights.transforms()

  for param in vit_model.parameters():
    param.requires_grad = False

  vit_model.heads = nn.Sequential(
    nn.Linear(in_features=768, out_features=num_classes, bias=True),
  )

  return vit_model, vit_transform
