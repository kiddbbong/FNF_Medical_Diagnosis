#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import time
import os
import sys
gpu_id = 0
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

parser = argparse.ArgumentParser(description='Training Medical Network')
parser.add_argument('--num_blocks', type=int, default=2)
parser.add_argument('--num_class', type=int, default=2)
parser.add_argument('--num_params', type=int, default=64)
args = parser.parse_args()

SPLIT = False
split_numb = 4

LOG_FILE = './hanyang_log.txt'
BEST_FILE = './hanyang_best_log.txt'

BATCH_SIZE = 8
TEST_BATCH = 1

TOTAL_EPOCHS = 1000


# In[2]:
def get_vars(name):
	return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)


#########################################################################################
def thread_func(sess, train_path_list, train_value_list, train_label_list, normal_numb):
	while True:
		batch_img, batch_label, img_name = utils.get_batch(BATCH_SIZE, train_path_list, train_value_list, train_label_list, normal_numb, 1, False)
		feed = {
			tensor_img : batch_img,
			tensor_label : batch_label
		}
		sess.run(enqueue, feed)

def thread_start(sess, train_path_list, train_value_list, train_label_list, normal_numb):
	n_threads = 4
	threads = []
	for _ in range(n_threads):
		t = threading.Thread(target=thread_func, args=(sess, train_path_list, train_value_list, train_label_list, normal_numb), daemon=True)
		t.start()
		threads.append(t)
	return threads

###########################################################################################

def Evaluate(sess, path_list, value_list, label_list, TEST_ITER, EPOCH_LR):
	valid_acc = 0.0
	valid_loss = 0.0

	for it in range(TEST_ITER):
		batch_img, batch_label, img_name = utils.get_batch(TEST_BATCH, path_list, value_list, label_list, _, it, True)
		
		valid_feed_dict = {
							tensor_img: batch_img,
							tensor_label: batch_label,
							training_lr: EPOCH_LR,
							training_flag: False
						}
		
		valid_batch_loss, valid_batch_acc, re_label, re_logit = sess.run([cost_class, accuracy, tensor_label, real_logit], feed_dict=valid_feed_dict)

		valid_acc += valid_batch_acc
		valid_loss += valid_batch_loss

	valid_loss /= TEST_ITER
	valid_acc /= TEST_ITER

	return valid_loss, valid_acc

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
	
correct_prediction = tf.equal(tf.argmax(real_logit, 1), tf.argmax(tensor_label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

g_vars = [v for v in tf.global_variables()]
print('Here Are All Variables', update_ops)

all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])


saver_loss = tf.train.Saver(tf.global_variables(), max_to_keep = 5)
saver_loss_path = './' + str(data_type) + '_' + str(Network) + '_Split_' + str(split_numb) + '_loss_model/'

saver_acc = tf.train.Saver(tf.global_variables(), max_to_keep = 5)
saver_acc_path = './' + str(data_type) + '_' + str(Network) + '_Split_' + str(split_numb) + '_acc_model/'

saver_latest = tf.train.Saver(tf.global_variables(), max_to_keep = 5)
saver_latest_path = './' + str(data_type) + '_' + str(Network) + '_Split_' + str(split_numb) + '_latest_model/'


print("DIR EXISTS LOSS?:", os.path.isdir(saver_loss_path))
if(os.path.isdir(saver_loss_path) == False):
	os.mkdir(saver_loss_path)
print(saver_loss_path)

print("DIR EXISTS ACC?:", os.path.isdir(saver_acc_path))
if(os.path.isdir(saver_acc_path) == False):
	os.mkdir(saver_acc_path)
print(saver_acc_path)

print("DIR EXISTS LATEST?:", os.path.isdir(saver_latest_path))
if(os.path.isdir(saver_latest_path) == False):
	os.mkdir(saver_latest_path)
print(saver_latest_path)

# In[ ]:
######################################QUEUE
Q_SIZE = 64
enq_list = [tensor_img, tensor_label]
queue = tf.FIFOQueue(Q_SIZE, dtypes=[tf.float32]*len(enq_list))
enqueue = queue.enqueue(enq_list)
print(queue)
dequeue = queue.dequeue()
###########################################


f_log = open(LOG_FILE, mode='at')
b_log = open(BEST_FILE, mode='at')

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
	
	sess.run(tf.global_variables_initializer())
	print('Number of Parameters', sess.run(all_trainable_vars))
	
	ckpt_medical = tf.train.get_checkpoint_state(saver_latest_path)
	if ckpt_medical and tf.train.checkpoint_exists(ckpt_medical.model_checkpoint_path):
		saver_latest.restore(sess, ckpt_medical.model_checkpoint_path)
		print('Restored!!!!!')
		transistor = 1
	else:
		print('Inited!!!!!')
		transistor = 0
	
	## Load Image Lists ##
	train_path_list, valid_path_list, test_path_list, external_path_list, train_value_list, valid_value_list, test_value_list, external_value_list, train_label_list, valid_label_list, test_label_list, external_label_list, normal_numb = utils.load_img(WIDTH, HEIGHT, split_numb, SPLIT)
	print('Normal Number', normal_numb)
	
	train_iter = int(len(train_path_list)/BATCH_SIZE)
	valid_iter = len(valid_path_list)
	test_iter = len(test_path_list)
	external_iter = len(external_path_list)
	
	##############learning rate###################
	epoch_lr = init_learning_rate
	
	if transistor == 1:
		valid_loss, valid_acc = Evaluate(sess, valid_path_list, valid_value_list, valid_label_list, valid_iter, epoch_lr)
		test_loss, test_acc = Evaluate(sess, test_path_list, test_value_list, test_label_list, test_iter, epoch_lr)
		external_loss, external_acc = Evaluate(sess, external_path_list, external_value_list, external_label_list, external_iter, epoch_lr)
		
		tmp_valid = valid_acc
		tmp_test = test_acc
		tmp_external = external_acc
		tmp_loss = (valid_loss + test_loss + external_loss)/3
		tmp_acc = (valid_acc + test_acc + external_acc)/3
		print('Trained Acc:%.5f, Loss:%.5f'%(valid_acc, valid_loss))
	else:
		tmp_valid = 0.0
		tmp_test = 0.0
		tmp_external = 0.0
		tmp_loss = 10000.0
		tmp_acc = 0.0
		
		
	#### Thread Start ####
	thread_start(sess, train_path_list, train_value_list, train_label_list, normal_numb)
	
	for epoch in range(1, TOTAL_EPOCHS + 1):
		train_acc = 0.0
		train_loss = 0.0
		
		if epoch%500 == 0:
			epoch_lr = epoch_lr/2
		
		for train in range(train_iter):
			batch_img, batch_label = sess.run(dequeue)
			#batch_img, batch_label, _ = utils.get_batch(BATCH_SIZE, train_path_list, train_value_list, train_label_list, normal_numb, train, False)
			
			
			train_feed_dict = {
							tensor_img: batch_img,
							tensor_label: batch_label,
							training_lr: epoch_lr,
							training_flag: True
						}

			_, train_step, batch_loss, batch_acc = sess.run([train_op, global_step, cost_class, accuracy], feed_dict=train_feed_dict)
			
			train_acc += batch_acc
			train_loss += batch_loss
			
		train_acc /= train_iter
		train_loss /= train_iter
		
		valid_loss, valid_acc = Evaluate(sess, valid_path_list, valid_value_list, valid_label_list, valid_iter, epoch_lr)
		test_loss, test_acc = Evaluate(sess, test_path_list, test_value_list, test_label_list, test_iter, epoch_lr)
		external_loss, external_acc = Evaluate(sess, external_path_list, external_value_list, external_label_list, external_iter, epoch_lr)
		
		triple_acc = (valid_acc + test_acc + external_acc)/3
		triple_loss = (valid_loss + test_loss + external_loss)/3
		
		line = "GLOBAL_STEP --- %d, Learning_Rate:%f\nTRAIN --- %d, loss:%.6f, accuracy:%.6f\nVALID --- %d, loss:%.6f, accuracy:%.6f\nTEST --- %d, loss:%.6f, accuracy:%.6f\nEXTERNAL VALID --- %d, loss:%.6f, accuracy:%.6f\n"%(train_step, epoch_lr, epoch, train_loss, train_acc, epoch, valid_loss, valid_acc, epoch, test_loss, test_acc, epoch, external_loss, external_acc)
		
		print(datetime.datetime.now(), line, file=f_log, flush=True)
		
		saver_latest.save(sess=sess, save_path= saver_latest_path + 'medical.ckpt', global_step=train_step)
		
		if triple_loss < tmp_loss:
			print(datetime.datetime.now(), line, file=b_log, flush=True)
			print('saved loss\n', file=b_log, flush=True)
			saver_loss.save(sess=sess, save_path= saver_loss_path + 'medical.ckpt', global_step=train_step)
			tmp_loss = triple_loss
			
		elif triple_acc > tmp_acc:
			print(datetime.datetime.now(), line, file=b_log, flush=True)
			print('saved acc\n', file=b_log, flush=True)
			saver_acc.save(sess=sess, save_path= saver_acc_path + 'medical.ckpt', global_step=train_step)
			tmp_acc = triple_acc
			
	print('##### FINISH #####')
	