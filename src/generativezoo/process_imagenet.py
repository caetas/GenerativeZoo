from config import data_raw_dir
import os
from tqdm import tqdm
import tarfile
import cv2
import shutil

print("Processing ImageNet")

print("Processing Train Set")
train_dir = os.path.join(data_raw_dir, 'imagenet', 'train')
classes = os.listdir(train_dir)
# remove the tar files
classes = [c for c in classes if not c.endswith('.tar')]
path = []

for c in tqdm(classes, desc="Classes"):
    # if there exists a tar file and a folder with the same name, delete the tar file
    if os.path.exists(os.path.join(train_dir, f"{c}.tar")):
        # if the folder exists, delete the tar file
        os.remove(os.path.join(train_dir, f"{c}.tar"))
    counter = 0
    images = os.listdir(os.path.join(train_dir, c))
    for i in images:
        # open the image
        image = cv2.imread(os.path.join(train_dir, c, i), 1)
        # if shape is 128x128, continue
        if image.shape[0] == 128 and image.shape[1] == 128:
            counter += 1
            if counter == 100:
                break
            continue
        # resize smallest dimension to 128
        if image.shape[0] < image.shape[1]:
            image = cv2.resize(image, (int(image.shape[1] * 128 / image.shape[0]), 128))
            # get the center crop
            image = image[:, (image.shape[1] - 128) // 2:(image.shape[1] + 128) // 2]
        else:
            image = cv2.resize(image, (128, int(image.shape[0] * 128 / image.shape[1])))
            # get the center crop
            image = image[(image.shape[0] - 128) // 2:(image.shape[0] + 128) // 2, :]
        cv2.imwrite(os.path.join(train_dir, c, i), image)
        counter = 0
    
    #make the folder a tar file
    with tarfile.open(os.path.join(train_dir,f"{c}.tar"), 'w') as tar:
        tar.add(os.path.join(train_dir, c))

    #remove the folder
    shutil.rmtree(os.path.join(train_dir, c))

print("Processing Validation Set")
# if val.tar exists, skip
if os.path.exists(os.path.join(data_raw_dir, 'imagenet', 'val.tar')):
    print("Validation set already processed")
else:

    val_dir = os.path.join(data_raw_dir, 'imagenet', 'val')
    images = os.listdir(val_dir)
    counter = 0
    for i in tqdm(images, desc="Images"):
        # open the image
        image = cv2.imread(os.path.join(val_dir, i), 1)
        # if shape is 128x128, continue
        if image.shape[0] == 128 and image.shape[1] == 128:
            counter += 1
            if counter == 100:
                break
            continue
        # resize smallest dimension to 128
        if image.shape[0] < image.shape[1]:
            image = cv2.resize(image, (int(image.shape[1] * 128 / image.shape[0]), 128))
            # get the center crop
            image = image[:, (image.shape[1] - 128) // 2:(image.shape[1] + 128) // 2]
        else:
            image = cv2.resize(image, (128, int(image.shape[0] * 128 / image.shape[1])))
            # get the center crop
            image = image[(image.shape[0] - 128) // 2:(image.shape[0] + 128) // 2, :]
        cv2.imwrite(os.path.join(val_dir, i), image)
        counter = 0
    
    #make the folder a tar file
    with tarfile.open(os.path.join(data_raw_dir, 'imagenet', 'val.tar'), 'w') as tar:
        tar.add(val_dir)

    #remove the folder
    shutil.rmtree(val_dir)


print("Processing Test Set")
# if test.tar exists, skip
if os.path.exists(os.path.join(data_raw_dir, 'imagenet', 'test.tar')):
    print("Test set already processed")
else:
    test_dir = os.path.join(data_raw_dir, 'imagenet', 'test')
    images = os.listdir(test_dir)
    counter = 0
    for i in tqdm(images, desc="Images"):
        # open the image
        image = cv2.imread(os.path.join(test_dir, i), 1)
        # if shape is 128x128, continue
        if image.shape[0] == 128 and image.shape[1] == 128:
            counter += 1
            if counter == 100:
                break
            continue
        # resize smallest dimension to 128
        if image.shape[0] < image.shape[1]:
            image = cv2.resize(image, (int(image.shape[1] * 128 / image.shape[0]), 128))
            # get the center crop
            image = image[:, (image.shape[1] - 128) // 2:(image.shape[1] + 128) // 2]
        else:
            image = cv2.resize(image, (128, int(image.shape[0] * 128 / image.shape[1])))
            # get the center crop
            image = image[(image.shape[0] - 128) // 2:(image.shape[0] + 128) // 2, :]
        cv2.imwrite(os.path.join(test_dir, i), image)
        counter = 0
    
    #make the folder a tar file
    with tarfile.open(os.path.join(data_raw_dir, 'imagenet', 'test.tar'), 'w') as tar:
        tar.add(test_dir)

    #remove the folder
    shutil.rmtree(test_dir)
