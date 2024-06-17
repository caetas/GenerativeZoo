import os
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
import cv2

def open_image(bytes_string):
    image = Image.open(BytesIO(bytes_string))
    image = np.array(image)
    return image

# open parquet file
df = pd.read_parquet('./../../../../data/raw/pokemons.parquet')

# create the directories
if not os.path.exists('./../../../../data/processed'):
    os.makedirs('./../../../../data/processed')
if not os.path.exists(os.path.join('./../../../../data/processed', 'pokemons')):
    os.makedirs(os.path.join('./../../../../data/processed', 'pokemons'))
if not os.path.exists(os.path.join('./../../../../data/processed', 'pokemons', 'images')):
    os.makedirs(os.path.join('./../../../../data/processed', 'pokemons', 'images'))

save_dir = os.path.join('./../../../../data/processed', 'pokemons')

# for each image, open it
for i in range(len(df)):
    image = open_image(df['image'][i]['bytes'])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    prompt = df['text'][i]
    # save the images
    # img names are 0000.png, 0001.png, etc.
    img_name = str(i).zfill(4) + '.png'
    cv2.imwrite(os.path.join(save_dir, 'images', img_name), image)
    # save img_name, sketch_name and prompt in a metadata.jsonl file
    with open(os.path.join(save_dir, 'train.jsonl'), 'a') as f:
        f.write('{"text": "' + prompt + '", "image": "./../../../../data/processed/pokemons/images/' + img_name + '"}\n')