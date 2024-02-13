from config import data_raw_dir
import os
import cv2
import shutil

images = os.listdir(os.path.join(data_raw_dir,'headct', 'head_ct'))

bad_imgs = [os.path.join(data_raw_dir, 'headct', 'head_ct', images[idx]) for idx in range(0,100)]
good_imgs = [os.path.join(data_raw_dir, 'headct', 'head_ct', images[idx]) for idx in range(100,200)]

for idx in range(len(bad_imgs)):
    #new destination should replace 'head_ct' by os.path.join(train,bad)
    if idx<90:
        shutil.move(bad_imgs[idx], bad_imgs[idx].replace('head_ct', os.path.join('train','bad')))
    else:
        shutil.move(bad_imgs[idx], bad_imgs[idx].replace('head_ct', os.path.join('test','bad')))

for idx in range(len(good_imgs)):
    #new destination should replace 'head_ct' by os.path.join(train,bad)
    if idx<90:
        shutil.move(good_imgs[idx], good_imgs[idx].replace('head_ct', os.path.join('train','good')))
    else:
        shutil.move(good_imgs[idx], good_imgs[idx].replace('head_ct', os.path.join('test','good')))