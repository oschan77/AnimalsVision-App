import os
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
  train_dir: str,
  test_dir: str,
  transform: transforms.Compose,
  batch_size: int,
  num_workers: int = NUM_WORKERS,
):
  train_data = ImageFolder(
    train_dir,
    transform=transform,
  )

  test_data = ImageFolder(
    test_dir,
    transform=transform,
  )

  train_dataloader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
  )

  test_dataloader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
  )

  class_names = train_data.classes

  return train_dataloader, test_dataloader, class_names
