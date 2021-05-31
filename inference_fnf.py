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
init_learning_rate = 1e-4#spa=1-4
data_type = utils.Train_Data.split('/')[-2].split('_')[0]
Network = 'ResNet18_Resize_Only_Spatial' #Network
checkpoint_path = './Custom_Directory'

parser = argparse.ArgumentParser(description='Training Medical Network')
parser.add_argument('--num_blocks', type=int, default=2)
parser.add_argument('--num_class', type=int, default=2)
parser.add_argument('--num_params', type=int, default=64)
args = parser.parse_args()

SPLIT = False
split_numb = 4

BATCH_SIZE = 8
TEST_BATCH = 1

TOTAL_EPOCHS = 1000



# In[2]:


def Evaluate(sess, path_list, value_list, label_list, TEST_ITER, EPOCH_LR, SET_TYPE):
	test_acc = 0.0
	test_loss = 0.0
	
	df = pd.DataFrame(columns=['Test Sample', 'Label', 'Normal', 'Abnormal'])

	for it in range(TEST_ITER):
		batch_img, batch_label, img_name = utils.get_batch(TEST_BATCH, path_list, value_list, label_list, _, it, True)
		
		test_feed_dict = {
							tensor_img: batch_img,
							tensor_label: batch_label,
							training_lr: EPOCH_LR,
							training_flag: False
						}
		
		test_batch_loss, test_batch_acc, re_label, re_logit = sess.run([cost_class, accuracy, tensor_label, real_logit], feed_dict=test_feed_dict)
		
		label_idx = img_name.split('/')[6]
		re_name = img_name.split('/')[-1].split('.')[0]

		normal_per = re_logit[0][0]
		abnormal_per = re_logit[0][1]

		df = df.append({'Test Sample' : re_name, 'Label' : label_idx, 'Normal' : normal_per*100, 'Abnormal' : abnormal_per*100}, ignore_index=True)

		test_acc += test_batch_acc
		test_loss += test_batch_loss

	df.to_excel('./Train_Hanyang_' + str(SET_TYPE) + '.xlsx', sheet_name='EASYDIFFICULT', float_format="%.2f")
	test_loss /= TEST_ITER
	test_acc /= TEST_ITER

	return test_loss, test_acc


# In[ ]:


tensor_img = tf.placeholder('float32', [None, 450, 450, 3], name='tensor_input')
tensor_label = tf.placeholder('float32', [None, args.num_class], name='tensor_label')

training_lr = tf.placeholder(tf.float32, name='learning_rate')
training_flag = tf.placeholder(tf.bool)


tensor_logit, _ = md.ResNet18_Only_Resize_Attention(tensor_img, args.num_blocks, args.num_class, args.num_params, training_flag, False)


real_logit = tf.nn.softmax(tensor_logit)

cost_class = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tensor_label, logits=tensor_logit))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

global_step = tf.Variable(0, name='global_step', trainable=False)

with tf.control_dependencies(update_ops):
	train_op = tf.train.AdamOptimizer(learning_rate=training_lr).minimize(cost_class, global_step=global_step)
	#train_op = tf.train.GradientDescentOptimizer(learning_rate=init_learning_rate).minimize(cost_class, global_step=global_step)
	
correct_prediction = tf.equal(tf.argmax(real_logit, 1), tf.argmax(tensor_label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

g_vars = [v for v in tf.global_variables()]
print('Here Are All Variables', update_ops)

saver_medical = tf.train.Saver(tf.global_variables(), max_to_keep = 5)
saver_medical_path = checkpoint_path

print("DIR EXISTS?:", os.path.isdir(saver_medical_path))
if(os.path.isdir(saver_medical_path) == False):
	os.mkdir(saver_medical_path)
print(saver_medical_path)


# In[ ]:


with tf.Session() as sess:
	
	sess.run(tf.global_variables_initializer())
	ckpt_medical = tf.train.get_checkpoint_state(saver_medical_path)
	if ckpt_medical and tf.train.checkpoint_exists(ckpt_medical.model_checkpoint_path):
		saver_medical.restore(sess, ckpt_medical.model_checkpoint_path)
		print(' ---- Restored ---- ', saver_medical)
	else:
		print(' ---- No Checkpoint ---- ', sys.exit())
	
	
	train_path_list, valid_path_list, test_path_list, external_path_list, train_value_list, valid_value_list, test_value_list, external_value_list, train_label_list, valid_label_list, test_label_list, external_label_list, normal_numb = utils.load_img(WIDTH, HEIGHT, split_numb, SPLIT)
	print('Normal Number', normal_numb)
	valid_iter = len(valid_path_list)
	test_iter = len(test_path_list)
	external_iter = len(external_path_list)
	print(len(valid_path_list), len(valid_value_list), len(valid_label_list))
	print(len(test_path_list), len(test_value_list), len(test_label_list))
	print(len(external_path_list), len(external_value_list), len(external_label_list))
	
	epoch_lr = init_learning_rate
	valid_loss, valid_acc = Evaluate(sess, valid_path_list, valid_value_list, valid_label_list, valid_iter, epoch_lr, 'VALID_SET')
	print('Here is valid, Loss & Acc', valid_loss, valid_acc)
	test_loss, test_acc = Evaluate(sess, test_path_list, test_value_list, test_label_list, test_iter, epoch_lr, 'TEST_SET')
	print('Here is test, Loss & Acc', test_loss, test_acc)
	external_loss, external_acc = Evaluate(sess, external_path_list, external_value_list, external_label_list, external_iter, epoch_lr, 'EXTERNAL_SET')
	print('Here is external, Loss & Acc', external_loss, external_acc)

	print('Finish##')
