#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import time
import os
import sys
import pandas as pd
gpu_id = -1
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)

import threading
import argparse
import random
import math
import datetime
from PIL import Image
import PIL

import Networks as md
# Look Carefully
import utils_fnf as utils

# In[ ]:
(WIDTH, HEIGHT) = (500, 500)

parser = argparse.ArgumentParser(description='Training Medical Network')
parser.add_argument('--num_blocks', type=int, default=2)
parser.add_argument('--num_class', type=int, default=2)
parser.add_argument('--num_params', type=int, default=64)

parser.add_argument('--img_path', type=str, default=None)
parser.add_argument('--checkpoint', type=str, default='./trained_model/')
args = parser.parse_args()



# In[2]:


def Evaluate(sess, single_img_path):
	
	batch_img = utils.inference_si(single_img_path, WIDTH, HEIGHT)
	
		
	test_feed_dict = {
						tensor_img: batch_img,
						training_flag: False
					}
	
	re_logit = sess.run([real_logit], feed_dict=test_feed_dict)
	
	
	normal_per = re_logit[0][0][0] * 100
	abnormal_per = re_logit[0][0][1] * 100
	
	print(" ###    Medical Diagnosis Prediction    ### ")

	print(" ###    Normal --- %.2f     //    Abnormal --- %.2f     ### \n"%(normal_per, abnormal_per))
	


# In[ ]:


tensor_img = tf.placeholder('float32', [None, 450, 450, 3], name='tensor_input')

training_lr = tf.placeholder(tf.float32, name='learning_rate')
training_flag = tf.placeholder(tf.bool)


tensor_logit, _ = md.ResNet18_Only_Resize_Attention(tensor_img, args.num_blocks, args.num_class, args.num_params, training_flag, False)


real_logit = tf.nn.softmax(tensor_logit)

global_step = tf.Variable(0, name='global_step', trainable=False)

g_vars = [v for v in tf.global_variables()]

saver_medical = tf.train.Saver(tf.global_variables(), max_to_keep = 5)


# In[ ]:


with tf.Session() as sess:
	
	sess.run(tf.global_variables_initializer())
	ckpt_medical = tf.train.get_checkpoint_state(args.checkpoint)
	if ckpt_medical and tf.train.checkpoint_exists(ckpt_medical.model_checkpoint_path):
		saver_medical.restore(sess, ckpt_medical.model_checkpoint_path)
		print(' ---- Restored ---- ', saver_medical)
	else:
		print(' ---- No Checkpoint ---- ', sys.exit())
	
	
	Evaluate(sess, args.img_path)
	
	
	print('Finish##')
