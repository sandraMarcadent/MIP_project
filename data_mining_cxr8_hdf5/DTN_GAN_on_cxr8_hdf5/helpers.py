# READ DATA IMPORTS

import h5py
import os
import time

# CODE HELPERS
import tensorflow as tf
import numpy as np

# DATA VISU
import matplotlib.pyplot as plt

config = tf.ConfigProto()
proto = config.gpu_options.allow_growth=True


'''PREPROCESSING HELPERS'''

################################################################################################

def resize_all(data, img_height, img_width):
    
    with tf.Graph().as_default():
                init = (tf.global_variables_initializer(), tf.local_variables_initializer())  
                with tf.Session(config=config) as sess:
                        resized_data = sess.run(tf.image.resize_images(data,[img_height, img_width]))
    
    return resized_data

###############################################################################################
    
def convert_to_rgb(data):
    
    with tf.Graph().as_default():
                init = (tf.global_variables_initializer(), tf.local_variables_initializer())  
                with tf.Session(config=config) as sess:
                        rgb_data = sess.run(tf.image.grayscale_to_rgb(data))
                        
    return rgb_data

################################################################################################
    
def rescale_gray_levels(data, max_scale, offset):
       
    with tf.Graph().as_default():
                init = (tf.global_variables_initializer(), tf.local_variables_initializer())  
                with tf.Session(config=config) as sess:               
                    normalized_data = sess.run(tf.subtract(tf.div(data,max_scale), offset))
        
    return normalized_data

################################################################################################

def pre_processing_manager(batch_i, img_height, img_width, resize=True, rescale_grays=True, convert_to_rgb=True):
    
    if resize:
                        batch_i = resize_all(batch_i, img_height, img_width)
                    
    if convert_to_rgb:
                        batch_i = convert_to_rgb(batch_i)
                                            
    if rescale_grays:
                        max_scale = np.max(batch_i)
                        offset = np.min(batch_i)                        
                        batch_i = rescale_gray_levels(batch_i, max_scale, offset)
                        
    return batch_i 



''' DISPLAYING IMAGES'''

################################################################################################

def show_normed_gray_image(x_data, idx=0):
    
    with tf.Graph().as_default():
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())  

        with tf.Session(config=config) as sess:
            sess.run(init)
            img = sess.run(tf.image.grayscale_to_rgb(x_data[idx]*255)).astype(np.uint8)

        plt.imshow(img)
        plt.show()

################################################################################################

def output_to_img(sess, ndarray, nchannels):
    
            if nchannels == 1: 
                img = sess.run(tf.image.grayscale_to_rgb(ndarray*255)).astype(np.uint8)

            else:
                img = (ndarray*255).astype(np.uint8)

            return img
                    
