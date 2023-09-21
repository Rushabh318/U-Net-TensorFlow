import os
from PIL import Image
import numpy as np
import cv2

def LoadData (path1, path2):

    image_dataset = os.listdir(path1)
    #write all filenames in directory in list mask_dataset
    mask_dataset = os.listdir(path2)

    orig_img = []
    mask_img = []

    for file in image_dataset:
        if '_multi_annot' in file:
            mask_img.append(file)
            name = file.replace('_multi_annot','')
            orig_img.append(name)
        
    orig_img.sort()
    mask_img.sort()
    
    return orig_img, mask_img

def PreprocessData(img, mask, target_shape_img, target_shape_mask, path1, path2):

    m = len(img)                     # number of images
    i_h,i_w,i_c = target_shape_img   # pull height, width, and channels of image
    m_h,m_w,m_c = target_shape_mask  # pull height, width, and channels of mask
    
    X = np.zeros((m,i_h,i_w,i_c), dtype=np.dtype('uint8'))
    y = np.zeros((m,m_h,m_w,m_c), dtype=np.dtype('uint8'))
    
    for file in img:
        index = img.index(file)
        path = os.path.join(path1, file)
        single_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).copy()
        single_img = cv2.resize(single_img, (i_h, i_w))
        single_img = np.reshape(single_img,(i_h,i_w,i_c))
        #IMREAD_GRAYSCALE already defaults to 8 bit. It is not necessary to divide by 8 bit again
        #single_img = single_img/256.
        X[index] = single_img        
        
        pixel_vals = [[ 36, 231, 253],
                       [120, 183,  53],
                       [141, 103,  48],
                       [ 84,   1,  68]]
        
        single_mask_ind = mask[index]
        path = os.path.join(path2, single_mask_ind)
        single_mask = cv2.imread(path)
        single_mask = cv2.resize(single_mask, [m_h,m_w], interpolation=cv2.INTER_NEAREST)
        single_mask[single_mask == pixel_vals[0]] = 0.0
        single_mask[single_mask == pixel_vals[1]] = 1.0
        single_mask[single_mask == pixel_vals[2]] = 2.0
        single_mask[single_mask == pixel_vals[3]] = 3.0
        single_mask = single_mask[:,:,0]
        single_mask = np.reshape(single_mask,(m_h,m_w,m_c))

        y[index] = single_mask
        
    return X, y