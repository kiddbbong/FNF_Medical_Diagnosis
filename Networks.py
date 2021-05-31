#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from model_utils import *


# In[ ]:


def ResNet18_Only_Resize_Attention(feature, num_block, num_class, num_params, is_training=True, reuse=False):
	feature = conv('conv0', feature, num_params, 2, 7)
	feature = Batch_Normalization(feature, training=is_training, scope='bn0')
	feature = relu(feature)
	feature = max_pool('max0', feature, (3, 3), 2)
		
	for n in range(num_block):
		fir_stride = 1
		stage_numb = 1
		name = 'stage_first%d'%n
		with tf.variable_scope(name, reuse = reuse):
			feature = Only_Resize_Attention_residual_block(name, feature, num_params, fir_stride, stage_numb, is_training, reuse = reuse)
	
	for n in range(num_block):
		if n == 0:
			fir_stride = 2
		else:
			fir_stride = 1
		stage_numb = 2
		name = 'stage_second%d'%n
		with tf.variable_scope(name, reuse = reuse):
			feature = Only_Resize_Attention_residual_block(name, feature, num_params*2, fir_stride, stage_numb, is_training, reuse = reuse)
			
	for n in range(num_block):
		if n == 0:
			fir_stride = 2
		else:
			fir_stride = 1
		stage_numb = 3
		name = 'stage_third%d'%n
		with tf.variable_scope(name, reuse = reuse):
			feature = Only_Resize_Attention_residual_block(name, feature, num_params*4, fir_stride, stage_numb, is_training, reuse = reuse)
	
	for n in range(num_block):
		if n == 0:
			fir_stride = 2
		else:
			fir_stride = 1
		stage_numb = 4
		name = 'stage_fourth%d'%n
		with tf.variable_scope(name, reuse = reuse):
			feature = Only_Resize_Attention_residual_block(name, feature, num_params*8, fir_stride, stage_numb, is_training, reuse = reuse)
			
	tmp_feature = feature
	
	avg_feature = tf.reduce_mean(feature, reduction_indices=[1, 2], name='avg')
	
	feature_out = tf.layers.dense(inputs=avg_feature, units=num_class, name='linear')
	
	return feature_out, tmp_feature