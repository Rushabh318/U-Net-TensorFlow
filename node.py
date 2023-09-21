import rospy
from sensor_msgs.msg import Image
import tensorflow as tf
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
import time

class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
  def __init__(self,
               y_true=None,
               y_pred=None,
               num_classes=None,
               name=None,
               dtype=None,
               **kwargs):
    super(UpdatedMeanIoU, self).__init__(num_classes = num_classes,name=name, dtype=dtype, **kwargs)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_pred = tf.math.argmax(y_pred, axis=-1)
    return super().update_state(y_true, y_pred, sample_weight)
  

class Segmentation:

    def __init__(self, model_path) -> None:
        self.unet = tf.keras.models.load_model(model_path, custom_objects={"UpdatedMeanIoU": UpdatedMeanIoU})
        self.subscriber = rospy.Subscriber("/nerian_stereo/left_image", Image, callback=self.image_processing, queue_size=1)
        self.publisher = rospy.Publisher('/amt/Image/pam/segmented_image', Image, queue_size=1)
        self.bridge = CvBridge()
        
    def image_processing(self, message):
        start_time = time.time()
        target_shape_img=[480, 480, 1]
        i_h,i_w,i_c = target_shape_img 
        image_msg = message
        cv_image = self.bridge.imgmsg_to_cv2(image_msg)

        # This block of code displays all pixels that are max value in cv_image
        # cv_image_test = np.zeros((768,1024))
        # cv_image_test[cv_image < 4090] = cv_image[cv_image < 4090]
        # cv_image_test = np.reshape(cv_image_test,(768,1024,1))
        # cv2.imshow('cv_image_test', cv_image_test)

        # cv image is 1080x1920 single channel uint16 image, vaue range 0 - 65535:

        if image_msg.encoding == 'mono16':
            cv_image = (cv_image/16).astype('uint8')
        #cv2.imshow('cv_image', cv_image)
        # single_image is 640x1080 single channel uint16 image, vaue range 0 - 65535:
        single_img = cv2.resize(cv_image, (i_h, i_w))        

        single_img = np.reshape(single_img,(i_h,i_w,i_c))
        #cv2.imshow("RGB_0", single_img)
 
        single_img = single_img[np.newaxis, :, :, :]

        #predict() should be replaced by direct call of mdoel according to https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict
        pred = self.unet(single_img).numpy()

        classes_dict = {
        (36, 231, 253):0,  # iron
        (120, 183,  53):1, # arm
        (141, 103,  48):2, # slack
        (84,   1,  68):3 # background
        }
        
        prediction = tf.argmax(pred, axis=-1)
        seg_map = np.zeros([pred.shape[1], pred.shape[2], 3], dtype=np.uint8)
        pred = prediction[0]
        for color, class_id in classes_dict.items():
            seg_map[pred==class_id] = color

        pred = seg_map

        ###### Resizing
        #nn_image = cv2.resize(mask, (1024, 768), interpolation=cv2.INTER_NEAREST)
        pred = cv2.resize(pred, (1024, 768), interpolation=cv2.INTER_LINEAR)
        pred = np.uint8(pred)
        gray = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
        pred = self.bridge.cv2_to_imgmsg(gray, "mono8")
        pred.header = image_msg.header
        #cubic_image = cv2.resize(mask, (1024, 768), interpolation=cv2.INTER_CUBIC)
   
        cv2.waitKey(1)
        self.publisher.publish(pred)
        print("Procedure time: {:.4f} seconds. Maximum rate: {:.4f} Hz".format((time.time() - start_time), 1/(time.time() - start_time)))

def main():
    model_path = "/home/rushabh/segmentation_models/multiclass_model_25-05_480x480x1/"
    
    rospy.init_node("segmentation_node", anonymous=True)
    seg = Segmentation(model_path)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main()

