from config import data_raw_dir
import matplotlib.pyplot as plt
import cv2
import h5py
import os
import pandas as pd


# open the file data_raw_dir/TextileDefects32/train32.h5
f = h5py.File(os.path.join(data_raw_dir,'TextileDefects32/test32.h5'), 'r')
# open csv file
df = pd.read_csv(os.path.join(data_raw_dir,'TextileDefects32/test32.csv'))


# print the keys of the dataframe
#df['index'] = df['index'] - 48000
# get df_good which contains only the good images
df_good = df[df['indication_type'] == 'good']
df_bad = df[df['indication_type'] != 'good']

# create a directory for the good images
os.mkdir(os.path.join(data_raw_dir,'TextileDefects32/test'))
os.mkdir(os.path.join(data_raw_dir,'TextileDefects32/test/good'))

# loop over the good images
for i in df_good['index']:
    # save the image in the directory
    cv2.imwrite(os.path.join(data_raw_dir,'TextileDefects32/test/good/{}.png'.format(i)), (f['images'][i]*255).astype('uint8'))

# create a directory for the bad images
os.mkdir(os.path.join(data_raw_dir,'TextileDefects32/test/bad'))

# loop over the bad images
for i in df_bad['index']:
    # save the image in the directory
    cv2.imwrite(os.path.join(data_raw_dir,'TextileDefects32/test/bad/{}.png'.format(i)), (f['images'][i]*255).astype('uint8'))
