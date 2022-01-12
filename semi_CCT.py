#%%  22/01/10
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as K
import cv2
from matplotlib.image import pil_to_array
from tensorflow.keras import layers, Input, Model
from PIL import Image
from tensorflow.keras.applications import ResNet50

import matplotlib.pyplot as plt
import math, time
import random
import os

from tqdm import tqdm
import glob

from scipy.io import loadmat
from tensorflow import keras
import argparse

#%%
#############################
### Parameters dictionary
#############################

PARAMS = {
    "iterations" : 10000,
    "batch_size" : 8,
    "total_items" : 1464,
    "learning_rate" : 0.00001,
    "MEAN" : [0.485, 0.456, 0.406],
    "STD" : [0.229, 0.224, 0.225],
    "image_padding" : (np.array([0.485, 0.456, 0.406])*255.).tolist(), 
    # "image_padding" : (np.array([0.485, 0.456, 0.406])).tolist(), 
    # "crop_size" : [256, 256],
    "crop_size" : [320, 320],
    "base_size" : [400, 400], 
    "ignore_index" : 255 # void
}

#%% Read PASCAL VOC data
# training / validation / test : 1464 / 1449 / 1456
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
        features.append(cv2.cvtColor(features_img, cv2.COLOR_BGR2RGB)) # BRG to RGB
        
        labels_img = cv2.imread(os.path.join(voc_dir, 'SegmentationClass', f'{fname}.png'))
        labels.append(cv2.cvtColor(labels_img, cv2.COLOR_BGR2RGB)) # BRG to RGB
        
        labels_c_img = Image.open(os.path.join(voc_dir, 'SegmentationClass',  f'{fname}.png'))
        labels_c_img = np.array(labels_c_img)
        labels_c.append(labels_c_img)
        
    return features, labels, labels_c

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

cmap = color_map(N = 256)
labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'void']
nclasses = 21
cmap.shape
cmap = np.vstack([cmap[:21], cmap[-1]])

label_to_color = dict(enumerate(cmap[:-1]))
label_to_color[PARAMS["ignore_index"]] = np.array(cmap[-1])
label_to_color
    
def view(model, num=5, rand = True):
    
    idx = np.random.choice(total_items, num, replace=False).tolist()
    if isinstance(rand, list):
        idx = rand
    
    x_batch = [np.load(file_dir + "/" + file_lst[j] , allow_pickle=True)[0]["img"] for j in idx]
    y_batch = [np.load(file_dir + "/" + file_lst[j] , allow_pickle=True)[0]["label"] for j in idx]

    for k in range(num):
        x_batch[k], y_batch[k] = augmentation(x_batch[k], y_batch[k])

    img_list = x_batch
    
    x_batch = tf.cast(x_batch, dtype = tf.float32)
    x_batch = (x_batch - 127.5) / 127.5

    model_ = model
    out = model_(x_batch)
    # np.round(tf.nn.softmax(out),3)
    result = np.argmax(tf.nn.softmax(out), axis = 3)
    y_batch = np.array(y_batch)

    _, h, w = result.shape
    pred_list = []
    label_list = []
    for m in range(result.shape[0]):
        
        pred_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        label_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        
        for gray, rgb in label_to_color.items():
            pred_rgb[result[m] == gray, :] = rgb
            label_rgb[y_batch[m] == gray, :] = rgb
        
        pred_list.append(pred_rgb)
        label_list.append(label_rgb)
    
    result_img = np.hstack([pred_list[0], label_list[0], img_list[0]])
    
    for i in range(1,num):
        result_img = np.vstack([result_img, np.hstack([pred_list[i], label_list[i], img_list[i]])])
    
    plt.figure(figsize = (25,25))
    plt.imshow(result_img)
    plt.show()



#%% 
#############################
### data augmentation
#############################
def resize(image, label):
    scale = random.random() * 1.5 + 0.5 # Scaling between [0.5, 2]
    h, w = int(PARAMS["base_size"][0] * scale), int(PARAMS["base_size"][1] * scale)
    image = np.asarray(Image.fromarray(np.uint8(image)).resize((w, h), Image.BICUBIC))
    # image = np.asarray(Image.fromarray(np.float32(image)).resize((w, h), Image.BICUBIC))
    label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
    # tf.image.~~~ 활용
    return image, label

def crop(image, label, crop_h, crop_w):
    h, w, _ = image.shape
    pad_h = max(crop_h - h, 0)
    pad_w = max(crop_w - w, 0)

    pad_kwargs = {
                "top": 0,
                "bottom": pad_h,
                "left": 0,
                "right": pad_w,
                "borderType": cv2.BORDER_CONSTANT,}

    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, value = PARAMS["image_padding"], **pad_kwargs)
        label = cv2.copyMakeBorder(label, value = PARAMS["ignore_index"], **pad_kwargs)

    h, w, _ = image.shape
    start_h = random.randint(0, h - crop_h)
    start_w = random.randint(0, w - crop_w)
    end_h = start_h + crop_h
    end_w = start_w + crop_w
    image = image[start_h:end_h, start_w:end_w]
    label = label[start_h:end_h, start_w:end_w]
    
    return image, label

def flip(image, label):
    # Random H flip
    if random.random() > 0.5:
        image = np.fliplr(image).copy()
        label = np.fliplr(label).copy()
    return image, label

def augmentation(image, label, _resize = True, _crop = True, _flip = True):

    if _resize:
        image, label = resize(image, label)
    
    if _crop:
        image, label = crop(image, label, crop_h = PARAMS["crop_size"][0], crop_w = PARAMS["crop_size"][1])

    if _flip:
        image, label = flip(image, label)
    
    return image, label

#%%
#############################
### Model definition
#############################

def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output
# %%
def DeeplabV3Plus(image_size, num_classes):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    resnet50 = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    return keras.Model(inputs=model_input, outputs=model_output)


#%%
#############################
### Model definition
#############################

model = DeeplabV3Plus(image_size=320, num_classes=21)
# model.load_weights("/home/jeon/Desktop/jung/model_weights/weights_CCT")
optimizer = K.optimizers.Adam(PARAMS['learning_rate'])

# model.summary()
model.output

#%%
#############################
### Model training
#############################
file_dir = "/home/jeon/Desktop/jung/VOC_image"
file_lst = os.listdir(file_dir)

batch_size = PARAMS["batch_size"]
iterations = PARAMS["iterations"]
total_items = PARAMS["total_items"]

#%%
model.load_weights("/home/jeon/Desktop/jung/model_weights/weights_CCT2")
model.weights

#%%
start_time = time.time()
for i in range(PARAMS["iterations"]): # iteration

    # data load
    idx = np.random.choice(total_items, batch_size, replace=False).tolist()
    x_batch = [np.load(file_dir + "/" + file_lst[j] , allow_pickle=True)[0]["img"] for j in idx]
    y_batch = [np.load(file_dir + "/" + file_lst[j] , allow_pickle=True)[0]["label"] for j in idx]
    
    # data augmentation
    for k in range(batch_size):
        x_batch[k], y_batch[k] = augmentation(x_batch[k], y_batch[k])
    
    x_batch = tf.cast(x_batch, dtype = tf.float32)
    x_batch = (x_batch - 127.5) / 127.5
    y_batch =  tf.one_hot(y_batch, depth = 21)
    
    with tf.GradientTape() as tape:
        output = model(x_batch)
        prob = tf.nn.softmax(output)
        
        loss = - tf.reduce_mean(tf.reduce_mean(tf.reduce_sum(y_batch * tf.math.log(tf.clip_by_value(prob, 1e-10, 1.0)), axis=-1), axis=[1, 2]))

    grad = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))
    if i % 1000 == 0 and i > 1 : model.save_weights("/home/jeon/Desktop/jung/model_weights/weights_CCT2")
    
    print(i, loss)
end_time = time.time()
print(end_time - start_time, "s")

#%%
#############################
### result
#############################


#%%
view(model = model, num = 8, rand = True)


