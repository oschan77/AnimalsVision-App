import os
import requests
import zipfile
from pathlib import Path

def download_and_extract_data(
  data_path: str,
  data_url: str,
):
  # Create directory to store the data
  data_path = Path(data_path)
  if not data_path.is_dir():
    data_path.mkdir(parents=True, exist_ok=True)

  # Download data from github
  response = requests.get(data_url)
  file_name = data_url.split('/')[-1]
  with open(data_path / file_name, 'wb') as file:
    file.write(response.content)

  # Unzip file
  with zipfile.ZipFile(data_path / file_name, 'r') as zip_file:
    zip_file.extractall(data_path)

  # Remove zip file
  os.remove(data_path / file_name)
