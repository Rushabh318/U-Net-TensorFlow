import os
from PIL import Image
import numpy as np
import cv2

def LoadData (path1, path2):

    image_dataset = os.listdir(path1)
    mask_dataset = os.listdir(path2)

    orig_img = []
    mask_img = []

    for file in mask_dataset:
    
        mask_img.append(file)
        name = file.replace('.png','.jpg')
        orig_img.append(name)
        
    orig_img.sort()
    mask_img.sort()
    
    return orig_img, mask_img

def PreprocessData(img, mask, target_shape_img, target_shape_mask, path1, path2):

    m = len(img)                     # number of images
    i_h,i_w,i_c = target_shape_img   # pull height, width, and channels of image
    m_h,m_w,m_c = target_shape_mask  # pull height, width, and channels of mask
    
    X = np.zeros((m,i_h,i_w,i_c), dtype=np.dtype('float64'))
    y = np.zeros((m,m_h,m_w,m_c), dtype=np.int32)
    
    for file in img:
        index = img.index(file)
        path = os.path.join(path1, file)
        single_img = cv2.imread(path)
        single_img = cv2.resize(single_img, (i_h, i_w))
        single_img = np.reshape(single_img,(i_h,i_w,i_c))
        single_img = single_img/256.
        X[index] = single_img
        
        single_mask_ind = mask[index]
        path = os.path.join(path2, single_mask_ind)
        single_mask = Image.open(path).convert("L")
        single_mask = single_mask.resize((m_h, m_w))
        single_mask = np.reshape(single_mask,(m_h,m_w,m_c))
        single_mask = single_mask.astype('float32')
        single_mask[single_mask >= 200] = 255.0  
        single_mask[single_mask < 200] = 0
        single_mask[single_mask == 255.0] = 1.0
        y[index] = single_mask
    return X, y