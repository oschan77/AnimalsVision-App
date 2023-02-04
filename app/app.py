import model
import os
import torch
import gradio as gr
from timeit import default_timer as timer
from torch import nn
from torchvision import transforms

class_names = ['chicken', 'elephant' ,'sheep']

DEVICE = 'cpu'

vit_model, vit_transform = model.create_vitb16_model(
  num_classes=len(class_names)
)

vit_model.load_state_dict(
  torch.load(
    f='vitb16_v1.pth',
    map_location=torch.device(DEVICE),
  )
)

def predict_single_image(image):
  start_time = timer()
  image = vit_transform(image).unsqueeze(0).to(DEVICE)
  vit_model.eval()
  logits = vit_model(image)
  with torch.inference_mode():
    probs = torch.softmax(logits, dim=1)

  classes_and_probs = {class_names[i]: float(probs[0][i]) for i in range(len(class_names))}
  inference_time = round(timer() - start_time, 5)

  return classes_and_probs, inference_time

title = 'AnimalsVision \U0001F413\U0001F418\U0001F411'
description = 'A ViT computer vision model to classify images of animals as chicken, elephant or sheep.'
article = 'GitHub Repo: https://github.com/oschan77/AnimalsVision-App'

examples = [['examples/' + example] for example in os.listdir('examples/')]

app = gr.Interface(
  fn=predict_single_image,
  inputs=gr.Image(type='pil'),
  outputs=[
    gr.Label(num_top_classes=len(class_names), label='Predictions'),
    gr.Number(label='Prediction time (sec)'),
  ],
  examples=examples,
  title=title,
  description=description,
  article=article,
)

app.launch()
