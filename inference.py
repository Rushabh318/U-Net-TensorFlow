from importlib.resources import path
import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import glob
from PIL import Image
import imageio

def Inference(input_path, model_path, target_shape_img=[960, 960, 3], is_video=False):

    i_h,i_w,i_c = target_shape_img
    unet = tf.keras.models.load_model(model_path)

    if is_video==False:
        single_img = cv2.imread(path)
        single_img = cv2.resize(single_img, (640, 1080))
        single_img = np.reshape(single_img,(i_h,i_w,i_c))
        single_img = single_img[np.newaxis, :, :, :]
        single_img = single_img/256
        pred = unet.predict(single_img)
    
    else:
        cap = cv2.VideoCapture(input_path) # a video feed from the camera directly can also be fed here.
        if (cap.isOpened()== False):
            print("Error opening video stream or file")
        while(cap.isOpened()):
            ret, single_img = cap.read()
            if ret==True:
                single_img = cv2.imread(path)
                single_img = cv2.resize(single_img, (i_h, i_w))
                single_img = np.reshape(single_img,(i_h,i_w,i_c))
                single_img = cv2.resize(single_img, (640, 1080))
                single_img = np.reshape(single_img,(i_h,i_w,i_c))
                single_img = single_img/256
                pred = unet.predict(single_img)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()

    pred[pred > 0.0] = 1.0
    pred[pred < 0.0] = 0.0
    pred = pred.squeeze()

    return pred

def main():
    input_path = '/home/messnix/bagfiles/pam_images_annotated/images_original/'
    model_path = 'model_path'
    target_shape_img = [960, 960, 3]
    pred = Inference(input_path, model_path, target_shape_img, False)
    imageio.imwrite('images/mask_{}_predicted.png', pred)

if __name__=="__main__":
    main()
