#%%
import os
import tensorflow as tf
import tensorflow.keras as K
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
#%%
voc_dir = '/home/jeon/Desktop/jung/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'

def read_voc_images(voc_dir, is_train=True):
    """Read all VOC feature and label images."""
    txt_fname_train = os.path.join(voc_dir, 'ImageSets', 'Segmentation', 'train.txt' if is_train else 'val.txt')
    with open(txt_fname_train, 'r') as f:
        images_train = f.read().split()

    # train data read
    features, labels, labels_c = [], [], []
    for i, fname in enumerate(images_train):
        features_img = cv2.imread(os.path.join(voc_dir, 'JPEGImages',f'{fname}.jpg'))
        # features_img = (features_img.astype('float32') - 127.5) / 127.5
        features.append(cv2.cvtColor(features_img, cv2.COLOR_BGR2RGB)) # BRG to RGB
        
        labels_img = cv2.imread(os.path.join(voc_dir, 'SegmentationClass', f'{fname}.png'))
        labels.append(cv2.cvtColor(labels_img, cv2.COLOR_BGR2RGB)) # BRG to RGB
        
        labels_c_img = Image.open(os.path.join(voc_dir, 'SegmentationClass',  f'{fname}.png'))
        labels_c_img = np.array(labels_c_img)
        # idx_255 = (labels_c_img == 255)
        # labels_c_img[idx_255] = -1
        labels_c.append(labels_c_img)
    
    return features, labels, labels_c

train_features_, train_labels_, train_labels_c_ = read_voc_images(voc_dir, True)

#%%
for i in range(len(train_features_)):

    file_dir = "/home/jeon/Desktop/jung/VOC_image/" + str(i) + ".npy"

    tmp = {"img" : train_features_[i],
           "label" : train_labels_c_[i]
        #    "true" : train_labels_[i]
    }

    file = np.array([tmp])
    np.save(file_dir, file, allow_pickle=True)


# %%
a = np.load(file_dir, allow_pickle=True)
a[0]["img"]
plt.imshow(a[0]["img"])
plt.imshow(a[0]["label"])
