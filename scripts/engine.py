import torch
from tqdm.auto import tqdm

def train_step(
  model: torch.nn.Module,
  dataloader: torch.utils.data.DataLoader,
  criterion: torch.nn.Module,
  optimizer: torch.optim.Optimizer,
  device: str,
):
  train_loss = 0

  model.train()

  for batch, (X, y) in enumerate(dataloader):
    X = X.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    out = model(X)
    loss = criterion(out, y)
    train_loss += loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_loss = train_loss / len(dataloader)
  
  return train_loss


def test_step(
  model: torch.nn.Module,
  dataloader: torch.utils.data.DataLoader,
  criterion: torch.nn.Module,
  device: str,
):
  test_loss = 0

  model.eval()

  with torch.inference_mode():
    for batch, (X, y) in enumerate(dataloader):
      X = X.to(device, non_blocking=True)
      y = y.to(device, non_blocking=True)

      out = model(X)
      loss = criterion(out, y)
      test_loss += loss.item()

      test_loss = test_loss / len(dataloader)
  
  return test_loss


def train(
  epochs: int,
  model: torch.nn.Module,
  train_dataloader: torch.utils.data.DataLoader,
  test_dataloader: torch.utils.data.DataLoader,
  criterion: torch.nn.Module,
  optimizer: torch.optim.Optimizer,
  device: str,
):
  model = model.to(device)

  for ep in tqdm(range(epochs)):
    train_loss = train_step(
      model=model,
      dataloader=train_dataloader,
      criterion=criterion,
      optimizer=optimizer,
      device=device,
    )

    test_loss = test_step(
      model=model,
      dataloader=test_dataloader,
      criterion=criterion,
      device=device,
    )

    print(f'Epoch: {ep} | Train loss: {train_loss} | Test loss: {test_loss}')
