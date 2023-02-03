import torch
from torch import nn
from torchvision import transforms
from timeit import default_timer as timer
from typing import List

def predict_single_image(
  image,
  model: nn.Module,
  transform: transforms.Compose,
  class_names: List[str],
  device: str,
):
  start_time = timer()
  image = transform(image).unsqueeze(0).to(device)
  model.eval()
  logits = model(image)
  with torch.inference_mode():
    probs = torch.softmax(logits, dim=1)

  classes_and_probs = {class_names[i]: float(probs[0][i]) for i in range(len(class_names))}
  inference_time = round(timer() - start_time, 5)

  print(f'classes_and_probs: {classes_and_probs}')
  print(f'inference_time: {inference_time}')

  return classes_and_probs, inference_time
