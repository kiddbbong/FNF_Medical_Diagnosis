#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import tensorflow as tf
import PIL
from PIL import Image
import random


# In[2]:


Train_Data = '/home/kiddyu1991/Data/Pelvic_Data/hanyang_train/'
Valid_Data = '/home/kiddyu1991/Data/Pelvic_Data/hanyang_valid/'
Test_Data = '/home/kiddyu1991/Data/Pelvic_Data/hanyang_test/'
External_Data = '/home/kiddyu1991/Data/Pelvic_Data/Seoul_Univ/'

if Test_Data.split('/')[-2].split('_')[0] == 'hanyang':
	Test_A_N = 58
	Train_A_N = 470
elif Test_Data.split('/')[-2].split('_')[0] == 'seoul':
	Test_A_N = 51
	Train_A_N = 415
elif Test_Data.split('/')[-2].split('_')[0] == 'Seoul':
	Test_A_N = 519
	Train_A_N = 470

print('DATA', Test_Data.split('/')[-2].split('_')[0])
print('#####[numb of type]#####', Test_A_N, Train_A_N)

# In[3]:

def get_all_img_paths(path_root, min_h=0, min_w=0):
	paths = []
	for (dirpath, dirnames, filenames) in os.walk(path_root):
		filenames = [f for f in filenames if not f[0] == '.']
		dirnames[:] = [d for d in dirnames if not d[0] == '.']
		for file in filenames:
			if (file.endswith(tuple(['.bmp', '.jpg', '.png']))):
				path = os.path.join(dirpath, file)
				img = Image.open(path)
				img_size = img.size
				h = img_size[1]
				w = img_size[0]
				if(min_h <= h and min_w <= w):
					paths.append(path)
	return paths

# In[5]:

def split_data(img_list, split_numb):
	split_list = []
	normal_list = []
	diff_list = []
	easy_list = []
	
	for i in range(len(img_list)):
		img_tag = img_list[i].split('/')[6]
		if img_tag == 'Normal':
			normal_list.append(img_list[i])
		elif img_tag == 'Abnormal_Diff':
			diff_list.append(img_list[i])
		else:
			easy_list.append(img_list[i])
			
	normal_numb = len(normal_list)
	diff_numb = len(diff_list)
	easy_numb = len(easy_list)
	normal_list = normal_list[0:int(normal_numb*split_numb/4)]
	diff_list = diff_list[0:int(diff_numb*split_numb/4)]
	easy_list = easy_list[0:int(easy_numb*split_numb/4)]
	
	split_list.extend(normal_list)
	split_list.extend(diff_list)
	split_list.extend(easy_list)
	split_list.sort()
	
	for j in range(len(split_list)):
		img_tag = split_list[j].split('/')[6]
		if img_tag == 'Normal':
			normal_numb = j
			break
	
	return split_list, normal_numb


def label_crop(img_path):
	img_tag = img_path.split('/')[6]
	
	if img_tag == 'Normal':
		img_label = [1, 0]
	else :
		img_label = [0, 1]
		
	img_label = np.expand_dims(img_label, axis=0)
		
	return img_label

def load_test_img(width, height):
	valid_path_list = get_all_img_paths(Valid_Data)
	test_path_list = get_all_img_paths(Test_Data)
	external_path_list = get_all_img_paths(External_Data)
	valid_path_list.sort()
	test_path_list.sort()
	external_path_list.sort()
		
	#### Image Value ####
	valid_value_list = []
	test_value_list = []
	external_value_list = []
	#### Label Value ####
	valid_label_list = []
	test_label_list = []
	external_label_list = []
		
	for i in range(len(valid_path_list)):
		img = Image.open(valid_path_list[i])
		label = label_crop(valid_path_list[i])
		w, h = img.size
		mid_h = int(h//2)
		area_h = int((w//2))
		img = img.crop((0, mid_h - area_h, w, mid_h + area_h))
		# Here Is CONVERT
		if img.mode != 'RGB':
			np_img = np.asarray(img)
			img_1 = np.expand_dims(np_img, axis=2)
			img_2 = np.expand_dims(np_img, axis=2)
			img_3 = np.expand_dims(np_img, axis=2)
			second_img = np.concatenate([img_1, img_2], axis=2)
			last_img = np.concatenate([second_img, img_3], axis=2)
			conv_img = Image.fromarray(last_img)
			img = conv_img.resize((width, height))
		else:
			img = img.resize((width, height))
			
		valid_value_list.append(img)
		valid_label_list.append(label)


	for j in range(len(test_path_list)):
		img = Image.open(test_path_list[j])
		label = label_crop(test_path_list[j])
		w, h = img.size
		mid_h = int(h//2)
		area_h = int((w//2))
		img = img.crop((0, mid_h - area_h, w, mid_h + area_h))
		# Here Is CONVERT
		if img.mode != 'RGB':
			img = np.asarray(img)
			img_1 = np.expand_dims(img, axis=2)
			img_2 = np.expand_dims(img, axis=2)
			img_3 = np.expand_dims(img, axis=2)
			second_img = np.concatenate([img_1, img_2], axis=2)
			last_img = np.concatenate([second_img, img_3], axis=2)
			conv_img = Image.fromarray(last_img)
			img = conv_img.resize((width, height))
		else:
			img = img.resize((width, height))
		
		test_value_list.append(img)
		test_label_list.append(label)
		
	for k in range(len(external_path_list)):
		img = Image.open(external_path_list[k])
		label = label_crop(external_path_list[k])
		w, h = img.size
		mid_h = int(h//2)
		area_h = int((w//2))
		img = img.crop((0, mid_h - area_h, w, mid_h + area_h))
		# Here Is CONVERT
		if img.mode != 'RGB':
			img = np.asarray(img)
			img_1 = np.expand_dims(img, axis=2)
			img_2 = np.expand_dims(img, axis=2)
			img_3 = np.expand_dims(img, axis=2)
			second_img = np.concatenate([img_1, img_2], axis=2)
			last_img = np.concatenate([second_img, img_3], axis=2)
			conv_img = Image.fromarray(last_img)
			img = conv_img.resize((width, height))
		else:
			img = img.resize((width, height))
		
		external_value_list.append(img)
		external_label_list.append(label)
		
	return valid_path_list, test_path_list, external_path_list, valid_value_list, test_value_list, external_value_list, valid_label_list, test_label_list, external_label_list

def sub_load_img(width, height, split_numb, SPLIT=False):
	train_path_list = get_all_img_paths(Train_Data)
	valid_path_list = get_all_img_paths(Valid_Data)
	train_path_list.sort()
	valid_path_list.sort()
	
	if SPLIT==True:
		train_path_list, normal_numb = split_data(train_path_list, split_numb)
		valid_path_list, _ = split_data(valid_path_list, split_numb)
		
	else:
		train_path_list, normal_numb = split_data(train_path_list, 4)
		valid_path_list, _ = split_data(valid_path_list, 4)
		
	
	train_value_list = []
	valid_value_list = []
	
	train_label_list = []
	valid_label_list = []
	
	for i in range(len(train_path_list)):
		img = Image.open(train_path_list[i])
		label = label_crop(train_path_list[i])
		w, h = img.size
		mid_h = int(h//2)
		area_h = int((w//2))
		img = img.crop((0, mid_h - area_h, w, mid_h + area_h))
		# Here Is CONVERT
		if img.mode != 'RGB':
			np_img = np.asarray(img)
			img_1 = np.expand_dims(np_img, axis=2)
			img_2 = np.expand_dims(np_img, axis=2)
			img_3 = np.expand_dims(np_img, axis=2)
			second_img = np.concatenate([img_1, img_2], axis=2)
			last_img = np.concatenate([second_img, img_3], axis=2)
			conv_img = Image.fromarray(last_img)
			img = conv_img.resize((width, height))
		else:
			img = img.resize((width, height))
			
		train_value_list.append(img)
		train_label_list.append(label)
		
	for j in range(len(valid_path_list)):
		img = Image.open(valid_path_list[j])
		label = label_crop(valid_path_list[j])
		w, h = img.size
		mid_h = int(h//2)
		area_h = int((w//2))
		img = img.crop((0, mid_h - area_h, w, mid_h + area_h))
		# Here Is CONVERT
		if img.mode != 'RGB':
			np_img = np.asarray(img)
			img_1 = np.expand_dims(np_img, axis=2)
			img_2 = np.expand_dims(np_img, axis=2)
			img_3 = np.expand_dims(np_img, axis=2)
			second_img = np.concatenate([img_1, img_2], axis=2)
			last_img = np.concatenate([second_img, img_3], axis=2)
			conv_img = Image.fromarray(last_img)
			img = conv_img.resize((width, height))
		else:
			img = img.resize((width, height))
			
		valid_value_list.append(img)
		valid_label_list.append(label)
		
	return train_path_list, valid_path_list, train_value_list, valid_value_list, train_label_list, valid_label_list, normal_numb
	
# In[6]:
def load_img(width, height, split_numb, SPLIT=False):
	train_path_list = get_all_img_paths(Train_Data)
	valid_path_list = get_all_img_paths(Valid_Data)
	test_path_list = get_all_img_paths(Test_Data)
	external_path_list = get_all_img_paths(External_Data)
	train_path_list.sort()
	valid_path_list.sort()
	test_path_list.sort()
	external_path_list.sort()
	
	if SPLIT==True:
		train_path_list, normal_numb = split_data(train_path_list, split_numb)
		valid_path_list, _ = split_data(valid_path_list, split_numb)
		test_path_list, _ = split_data(test_path_list, split_numb)
		
	else:
		train_path_list, normal_numb = split_data(train_path_list, 4)
		valid_path_list, _ = split_data(valid_path_list, 4)
		test_path_list, _ = split_data(test_path_list, 4)
		
	#### Image Value ####
	train_value_list = []
	valid_value_list = []
	test_value_list = []
	external_value_list = []
	#### Label Value ####
	train_label_list = []
	valid_label_list = []
	test_label_list = []
	external_label_list = []

	for i in range(len(train_path_list)):
		img = Image.open(train_path_list[i])
		label = label_crop(train_path_list[i])
		w, h = img.size
		mid_h = int(h//2)
		area_h = int((w//2))
		img = img.crop((0, mid_h - area_h, w, mid_h + area_h))
		# Here Is CONVERT
		if img.mode != 'RGB':
			np_img = np.asarray(img)
			img_1 = np.expand_dims(np_img, axis=2)
			img_2 = np.expand_dims(np_img, axis=2)
			img_3 = np.expand_dims(np_img, axis=2)
			second_img = np.concatenate([img_1, img_2], axis=2)
			last_img = np.concatenate([second_img, img_3], axis=2)
			conv_img = Image.fromarray(last_img)
			img = conv_img.resize((width, height))
		else:
			img = img.resize((width, height))
			
		train_value_list.append(img)
		train_label_list.append(label)
		
		
	for j in range(len(valid_path_list)):
		img = Image.open(valid_path_list[j])
		label = label_crop(valid_path_list[j])
		w, h = img.size
		mid_h = int(h//2)
		area_h = int((w//2))
		img = img.crop((0, mid_h - area_h, w, mid_h + area_h))
		# Here Is CONVERT
		if img.mode != 'RGB':
			np_img = np.asarray(img)
			img_1 = np.expand_dims(np_img, axis=2)
			img_2 = np.expand_dims(np_img, axis=2)
			img_3 = np.expand_dims(np_img, axis=2)
			second_img = np.concatenate([img_1, img_2], axis=2)
			last_img = np.concatenate([second_img, img_3], axis=2)
			conv_img = Image.fromarray(last_img)
			img = conv_img.resize((width, height))
		else:
			img = img.resize((width, height))
			
		valid_value_list.append(img)
		valid_label_list.append(label)


	for k in range(len(test_path_list)):
		img = Image.open(test_path_list[k])
		label = label_crop(test_path_list[k])
		w, h = img.size
		mid_h = int(h//2)
		area_h = int((w//2))
		img = img.crop((0, mid_h - area_h, w, mid_h + area_h))
		# Here Is CONVERT
		if img.mode != 'RGB':
			img = np.asarray(img)
			img_1 = np.expand_dims(img, axis=2)
			img_2 = np.expand_dims(img, axis=2)
			img_3 = np.expand_dims(img, axis=2)
			second_img = np.concatenate([img_1, img_2], axis=2)
			last_img = np.concatenate([second_img, img_3], axis=2)
			conv_img = Image.fromarray(last_img)
			img = conv_img.resize((width, height))
		else:
			img = img.resize((width, height))
		
		test_value_list.append(img)
		test_label_list.append(label)
		
	for z in range(len(external_path_list)):
		img = Image.open(external_path_list[z])
		label = label_crop(external_path_list[z])
		w, h = img.size
		mid_h = int(h//2)
		area_h = int((w//2))
		img = img.crop((0, mid_h - area_h, w, mid_h + area_h))
		# Here Is CONVERT
		if img.mode != 'RGB':
			img = np.asarray(img)
			img_1 = np.expand_dims(img, axis=2)
			img_2 = np.expand_dims(img, axis=2)
			img_3 = np.expand_dims(img, axis=2)
			second_img = np.concatenate([img_1, img_2], axis=2)
			last_img = np.concatenate([second_img, img_3], axis=2)
			conv_img = Image.fromarray(last_img)
			img = conv_img.resize((width, height))
		else:
			img = img.resize((width, height))
		
		external_value_list.append(img)
		external_label_list.append(label)

	return train_path_list, valid_path_list, test_path_list, external_path_list, train_value_list, valid_value_list, test_value_list, external_value_list, train_label_list, valid_label_list, test_label_list, external_label_list, normal_numb


def get_batch(BATCH_SIZE, PATH_LIST, VALUE_LIST, LABEL_LIST, normal_numb, test_idx, TEST):
	for i in range(BATCH_SIZE):
		
		if TEST == True:
			
			img_path = PATH_LIST[test_idx]
			img = VALUE_LIST[test_idx]
			label = LABEL_LIST[test_idx]
			
			img = img.transpose(method=PIL.Image.ROTATE_90)
			
			w, h = img.size
			img = img.crop((25, 25, w - 25, h - 25))

		else:
			choice_idx = random.randint(0, 1)
			if choice_idx == 0:
				idx = random.randint(0, normal_numb - 1)
			else:
				idx = random.randint(normal_numb, len(PATH_LIST) - 1)
				
			img_path = PATH_LIST[idx]
			img = VALUE_LIST[idx]
			label = LABEL_LIST[idx]

			origin_w, origin_h = img.size
			crop_int = int(origin_w // 10)
			
			start_h = random.randint(0, crop_int)
			start_w = random.randint(0, crop_int)

			img = img.crop((start_w, start_h, start_w + origin_w - crop_int, start_h + origin_w - crop_int))

			# change if square
			flip_idx = random.randint(0, 3)
		
			if flip_idx == 1:
				img = img.transpose(method=PIL.Image.FLIP_LEFT_RIGHT)
			elif flip_idx == 2:
				img = img.transpose(method=PIL.Image.FLIP_TOP_BOTTOM)
			elif flip_idx == 3:
				img = img.transpose(method=PIL.Image.TRANSPOSE)
			else:
				img = img
			
			# change if square
			rotate_idx = random.randint(0, 3)
		
			if rotate_idx == 1:
				img = img.transpose(method=PIL.Image.ROTATE_180)
			elif rotate_idx == 2:
				img = img.transpose(method=PIL.Image.ROTATE_90)
			elif rotate_idx == 3:
				img = img.transpose(method=PIL.Image.ROTATE_270)
			else:
				img = img
			
		np_img = np.asarray(img)

		img_data = np.expand_dims(np_img, axis=0)
		
		if i == 0:
			batch_img = img_data
			batch_label = label
		else:
			batch_img = np.concatenate([batch_img, img_data], axis=0)
			batch_label = np.concatenate([batch_label, label], axis=0)
			
	return batch_img, batch_label, img_path


# In[ ]:




