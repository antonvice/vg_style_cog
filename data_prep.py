import pandas as pd
from PIL import Image
import requests
from io import BytesIO

class DLoader:
    def __init__(self, csv_file, batch_size: int = 1):
        self.df = pd.read_csv(csv_file)
        self.batch_size = batch_size
        self.urls = []
    def __iter__(self):
        for i in range(0, len(self.df), self.batch_size):
            yield self.df[i : i + self.batch_size]
    
    def __len__(self):
        return len(self.df) // self.batch_size
    
    def _filter(self):
        self.df = self.df[self.df["Artist"] == 'Vincent van Gogh']
        # drop all columns besides ImageURL
        self.urls = self.df["ImageURL"]
        self.urls = self.urls.reset_index(drop=True)

        
    def _get_imgs(self, img_folder):
        idx=0
        headers = {'User-Agent': 'CoolBot/0.1 (https://callconnect.ai/coolbot/; anton@callconnect.ai)'}

        # download images from links in column
        for url in self.urls:
            try:
                response = requests.get(url, headers=headers)
                img = Image.open(BytesIO(response.content))
                img.save(f'{img_folder}/{url.split("/")[-1]}')
                print(f'Progress: {idx+1}/{len(self.df)}')
            except Exception as e:
                print(f'Error downloading {url}') 
                print(e)
            idx+=1
    def _print_cols(self):
        print(self.urls)
            
# Load the Images
dataset = DLoader('vgdb_2016.csv')
dataset._filter()
image_paths = dataset._get_imgs('imgs')

#zip folder
import shutil

from pathlib import Path

folder = Path('imgs')
shutil.make_archive('vangogh_images', 'zip', folder)
print("Images zipped to vangogh_images.zip")