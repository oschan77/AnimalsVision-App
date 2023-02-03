import torch
import get_data, create_vitb16_model, data_setup, engine, utils

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_PATH = 'data'
DATA_URL = 'https://github.com/oschan77/AnimalsVision-App/raw/main/data/elephant_chicken_sheep_data.zip'
NUM_CLASSES = 3
TRAIN_DIR = 'data/elephant_chicken_sheep_data/train'
TEST_DIR = 'data/elephant_chicken_sheep_data/test'
BATCH_SIZE = 32
LEARNING_RATE = 5e-3
EPOCHS = 10
TARGET_PATH = 'saved_models'
MODEL_NAME = 'vitb16_v1.pth'

get_data.download_and_extract_data(
  data_path=DATA_PATH,
  data_url=DATA_URL,
)

vit_model, vit_transform = create_vitb16_model.finetune_vitb16(
  num_classes=NUM_CLASSES,
)

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
  train_dir=TRAIN_DIR,
  test_dir=TEST_DIR,
  transform=vit_transform,
  batch_size=BATCH_SIZE,
)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vit_model.parameters(), lr=LEARNING_RATE)

engine.train(
  epochs=EPOCHS,
  model=vit_model,
  train_dataloader=train_dataloader,
  test_dataloader=test_dataloader,
  criterion=criterion,
  optimizer=optimizer,
  device=DEVICE,
)

utils.save_model(
  model=vit_model,
  target_path=TARGET_PATH,
  model_name=MODEL_NAME,
)
