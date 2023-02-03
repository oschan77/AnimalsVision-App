import torch
from pathlib import Path

def save_model(
  model: torch.nn.Module,
  target_path: str,
  model_name: str,
):
  assert model_name.endswith('pth') or model_name.endswith('.pt'), "[Invalid model name]: model_name should end with '.pth' or '.pt'."

  target_path = Path(target_path)
  target_path.mkdir(parents=True, exist_ok=True)

  torch.save(
    obj = model.state_dict(),
    f = target_path / model_name,
  )
