#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf

# In[ ]:
def relu(feature):
	return tf.nn.relu(feature)

def conv(name, feature, ch_out, stride, kernel_size=3):
	with tf.name_scope(name):
		ch_in = feature.get_shape().as_list()[3]
		w = tf.get_variable(name, [kernel_size, kernel_size, ch_in, ch_out], initializer = tf.contrib.layers.xavier_initializer())
		feature = tf.nn.conv2d(feature, w, strides = [1, stride, stride, 1], padding='SAME')
		
		return feature
	
def avr_pool(x, size, strides):
	return tf.layers.average_pooling2d(inputs=x, pool_size=size, strides=strides, padding='valid')
	
def Average_pooling(x, pool_size=[2,2], stride=2, padding='SAME'):
	return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def global_pool(x, stride=1):
	return global_avg_pool(x, name='Global_Avg_Pooling')

def Batch_Normalization(x, training, scope):
	with tf.name_scope(scope):
		return tf.layers.batch_normalization(x, training=training)

def max_pool(name, feature, kernel_size, stride):
	with tf.name_scope(name):
		feature = tf.layers.max_pooling2d(inputs = feature, pool_size = kernel_size, strides = stride, name = name)
		
		return feature

################################ Attentions ###################################

def spatial_attention(input_feature, name):
	kernel_size = 7
	kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
	with tf.variable_scope(name):
		avg_pool = tf.reduce_mean(input_feature, axis=[3], keepdims=True)
		assert avg_pool.get_shape()[-1] == 1
		max_pool = tf.reduce_max(input_feature, axis=[3], keepdims=True)
		assert max_pool.get_shape()[-1] == 1
		concat = tf.concat([avg_pool,max_pool], 3)
		assert concat.get_shape()[-1] == 2
		concat = tf.layers.conv2d(concat, filters=1, kernel_size=[kernel_size,kernel_size], strides=[1,1],padding="same", activation=None, kernel_initializer=kernel_initializer, use_bias=False, name='conv')
	
		assert concat.get_shape()[-1] == 1
		concat = tf.sigmoid(concat, 'sigmoid')
	
	return input_feature * concat
	
def channel_attention(input_feature, name, ratio=16):
	kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
	bias_initializer = tf.constant_initializer(value=0.0)
	with tf.variable_scope(name):
		channel = input_feature.get_shape()[-1]
		avg_pool = tf.reduce_mean(input_feature, axis=[1,2], keepdims=True)
		
		assert avg_pool.get_shape()[1:] == (1,1,channel)
		avg_pool = tf.layers.dense(inputs=avg_pool, units=channel//ratio, activation=tf.nn.relu, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='mlp_0', reuse=None)
		assert avg_pool.get_shape()[1:] == (1,1,channel//ratio)
		avg_pool = tf.layers.dense(inputs=avg_pool, units=channel, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='mlp_1', reuse=None)
		assert avg_pool.get_shape()[1:] == (1,1,channel)
		max_pool = tf.reduce_max(input_feature, axis=[1,2], keepdims=True)
		assert max_pool.get_shape()[1:] == (1,1,channel)
		max_pool = tf.layers.dense(inputs=max_pool, units=channel//ratio, activation=tf.nn.relu, name='mlp_0', reuse=True)
		assert max_pool.get_shape()[1:] == (1,1,channel//ratio)
		max_pool = tf.layers.dense(inputs=max_pool, units=channel, name='mlp_1', reuse=True)
		assert max_pool.get_shape()[1:] == (1,1,channel)
		
		scale = tf.sigmoid(avg_pool + max_pool, 'sigmoid')
		
	return input_feature * scale

def cbam_block(input_feature, name, ratio=16):
	with tf.variable_scope(name):
		attention_feature = channel_attention(input_feature, 'ch_at', ratio)
		attention_feature = spatial_attention(attention_feature, 'sp_at')
		print ("CBAM Hello")
		
	return attention_feature

def resize_spatial_attention(input_feature, stride, stage_numb, name):
	kernel_size = 5
	kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
	with tf.variable_scope(name):
		avg_pool = tf.reduce_mean(input_feature, axis=[3], keepdims=True)
		assert avg_pool.get_shape()[-1] == 1
		max_pool = tf.reduce_max(input_feature, axis=[3], keepdims=True)
		assert max_pool.get_shape()[-1] == 1
		concat = tf.concat([avg_pool,max_pool], 3)
		assert concat.get_shape()[-1] == 2
		
		B, H, W, C = concat.get_shape() #get original size
		
		if stage_numb == 1:
			concat = avr_pool(concat, (3, 3), stride) #decrease size
			concat = avr_pool(concat, (3, 3), stride)
			concat = avr_pool(concat, (3, 3), stride)
		elif stage_numb == 2:
			concat = avr_pool(concat, (3, 3), stride)
			concat = avr_pool(concat, (3, 3), stride)
		elif stage_numb == 3:
			concat = avr_pool(concat, (3, 3), stride)
		else:
			concat = concat
		
		concat = tf.layers.conv2d(concat, filters=1, kernel_size=[kernel_size,kernel_size], strides=[1,1],padding="same", activation=None, kernel_initializer=kernel_initializer, use_bias=False, name='conv')
	
		assert concat.get_shape()[-1] == 1
		concat = tf.sigmoid(concat, 'sigmoid')
		
		if stage_numb == 4:
			concat = concat
		else:
			concat = tf.image.resize_nearest_neighbor(concat, (H, W))
	
	return input_feature * concat

################################################################################# Blocks #######################################################


def Only_Resize_Attention_residual_block(name, feature, ch_out, fir_stride, stage_numb, is_training, reuse = False):
	with tf.name_scope(name):
		feature_tmp = conv('conv1', feature, ch_out, fir_stride)
		feature_tmp = Batch_Normalization(feature_tmp, training=is_training, scope=name+'/bn1')
		feature_tmp = relu(feature_tmp)
		feature_tmp = conv('conv2', feature_tmp, ch_out, 1)
		feature_tmp = Batch_Normalization(feature_tmp, training=is_training, scope=name+'/bn2')
		feature_tmp = resize_spatial_attention(feature_tmp, 2, stage_numb, name+'/resize_att')
		
		if fir_stride == 2:
			feature_out = relu(feature_tmp)
		else:
			feature_out = relu(feature_tmp + feature)

	return feature_out



