#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ASR.file_function.file_func import *
from ASR.file_function.logic_func import *
import config

import numpy as np
from tqdm import tqdm


class DataSpeech():
	'''
	Output pinyin ; Mode A/B
	'''
	
	def __init__(self, path, type_, performance=False, **kwargs):
		'''
		path：folder prefix
		'''
				
		self.datapath = path
		self.type_ = type_ # Three types: {train、dev、test}
		assert(self.type_ in ['train','dev','test']),"data generator label not correct"

		self.slash = config.slash()
				
		self.list_symbol = GetSymbolList(config.pinyin_dict)
		self.output_mode = config.OUTPUT_MODE

		self.SymbolNum = len(self.list_symbol)
		
		self.LoadDataList()
		self.LoadNoiseList()

		self.feature_method = config.FEATURE_EXTRACTION
		self.feature_length = config.FEATURE_LENGTH
		self.feature_method_2 = config.FEATURE_EXTRACTION_2
		self.feature_length_2 = config.FEATURE_LENGTH_2

		self.bg = np.load(config.bg_folder+config.FEATURE_BG)
		self.bg2 = np.load(config.bg_folder+config.FEATURE_BG_2)
	
	def LoadDataList(self):

		folder_ls = config.DATASETS
		filename_ls = []

		for f in folder_ls:
			filename_ls.append(f'{self.datapath+f+self.slash+self.type_}.json')

		self.list_all_json,self.dataset_range = get_json_all(filename_ls)
		
		self.DataNum = self.GetDataNum()
		self.min_batch_size = len(self.dataset_range)

	def LoadNoiseList(self):
		self.list_noise = []
		for i in range(33):
			self.list_noise.append(config.data_path+f'noise/noise_{i}.wav')

	def GetDataNum(self):
		return len(self.list_all_json)

	def GetDataHeat(self,ran_num):
			
		filename = self.list_all_json[ran_num]['r_path']
		list_symbol = self.list_all_json[ran_num]['pinyin']
		
		filename = config.path_checker(filename)
		
		noise_filename = ''
		if (config.ADD_NOISE) and (ran_num%2 == 0):
			rand_noise = np.random.randint(0,33)
			noise_filename = self.list_noise[rand_noise]
		wavsignal,fs = read_wav_data_2((self.datapath + filename),config.AUDIO_SPEED,config.SPEED_CHANGE,noise_filename)
		
		feat_out=[]
		for i in list_symbol:
			if(''!=i):
				n=self.SymbolToNum(i)
				feat_out.append(n)
		
		## handle error (that pinyin not in dict)
		if (-1 in feat_out):
			return [(-1,),(-1,)],-1
		
		data_input = self.feature_method.feature(wavsignal,fs)
		data_input = data_input.reshape(data_input.shape[0],data_input.shape[1],1)

		if config.DOUBLE_INPUT:
			data_input_2 = self.feature_method_2.feature(wavsignal,fs)
			data_input_2 = data_input_2.reshape(data_input_2.shape[0],data_input_2.shape[1],1)
		else:
			data_input_2 = None

		if (data_input.shape[0] > config.AUDIO_LENGTH):
			return [(-1,),(-1,)],-1
				
		data_label = np.array(feat_out)

		if len(data_label) > config.LABEL_MAX_STRING_LENGTH:
			return [(-1,),(-1,)],-1

		return [data_input,data_input_2], data_label
	
	def data_generator_double(self,batch_size=6, audio_length = config.AUDIO_LENGTH, go_backward=False, **kwargs):

		assert config.DOUBLE_INPUT, 'config.DOUBLE_INPUT should be True'

		if batch_size > self.min_batch_size:
			pass
		else:
			batch_size = self.min_batch_size

		labels = np.zeros((batch_size,1), dtype = np.float)
		
		while True:
			X = np.zeros((batch_size, audio_length, self.feature_length, 1), dtype = np.float)
			X2 = np.zeros((batch_size, audio_length, self.feature_length_2, 1), dtype = np.float)
			y = np.zeros((batch_size, 64), dtype=np.int16)
			
			input_length = []
			label_length = []

			count = 0
			start = 0
			while(count < batch_size):
				# ensure every batch contain at least one record from different dataset 
				if count < self.min_batch_size:
					ran_num = np.random.randint(start,self.dataset_range[count])
				else:
					ran_num = np.random.randint(0,self.DataNum)

				data_input_ls, data_labels = self.GetDataHeat(ran_num)
				data_input = data_input_ls[0]
				data_input_2 = data_input_ls[1]

				if (-1 in data_input) or (-1 in data_input_2): #handle error
					continue

				X[count]=self.bg
				X2[count]=self.bg2
				input_length.append(config.OUTPUT_LENGTH)
				
				if go_backward == False:
					X[count,0:len(data_input)] = data_input
					X2[count,0:len(data_input)] = data_input_2

				elif go_backward == True:
					X[count,-(len(data_input)):] = data_input
					X2[count,-(len(data_input)):] = data_input_2

				if self.feature_method.USE_CMVN == True:
					X[count] = CMVN(X[count])
				if self.feature_method_2.USE_CMVN == True:
					X2[count] = CMVN(X2[count])

				y[count,0:len(data_labels)] = data_labels
				
				label_length.append([len(data_labels)])

				if count < self.min_batch_size:
					start = self.dataset_range[count]
				count += 1
			
			label_length = np.matrix(label_length)
			input_length = np.array([input_length]).T

			yield [X, X2, y, input_length, label_length ], labels

	def data_generator_single(self,batch_size=6, audio_length = config.AUDIO_LENGTH, go_backward=False, **kwargs):

		if batch_size > self.min_batch_size:
			pass
		else:
			batch_size = self.min_batch_size

		labels = np.zeros((batch_size,1), dtype = np.float)
		
		while True:
			X = np.zeros((batch_size, audio_length, self.feature_length, 1), dtype = np.float)
			y = np.zeros((batch_size, 64), dtype=np.int16)
			
			input_length = []
			label_length = []

			count = 0
			start = 0
			while(count < batch_size):
				# ensure every batch contain at least one record from different dataset 
				if count < self.min_batch_size:
					ran_num = np.random.randint(start,self.dataset_range[count])
				else:
					ran_num = np.random.randint(0,self.DataNum)

				data_input_ls, data_labels = self.GetDataHeat(ran_num)
				data_input = data_input_ls[0]

				if (-1 in data_input): #handle error
					continue

				X[count]=self.bg
				input_length.append(config.OUTPUT_LENGTH)
				
				if go_backward == False:
					X[count,0:len(data_input)] = data_input

				elif go_backward == True:
					X[count,-(len(data_input)):] = data_input

				if self.feature_method.USE_CMVN == True:
					X[count] = CMVN(X[count])

				y[count,0:len(data_labels)] = data_labels
				
				label_length.append([len(data_labels)])

				if count < self.min_batch_size:
					start = self.dataset_range[count]
				count += 1
			
			label_length = np.matrix(label_length)
			input_length = np.array([input_length]).T

			yield [X, y, input_length, label_length ], labels

	def SymbolToNum(self,symbol):
		try:
			if(symbol != ''):
				return self.list_symbol.index(symbol)
		except ValueError: # not in list
			return -1

	def test_length(self,threshold=1600):

		smaller_than_or_equal_to = 0
		for i in tqdm(range(self.DataNum)):
			seq = self.GetDataHeat(i)[0]
			if (len(seq) <= threshold):
				smaller_than_or_equal_to += 1
			seq = None
		
		print(f'[result] There are {smaller_than_or_equal_to} audio smaller than {threshold} frame')

class DataSpeech2(DataSpeech):
	'''
	Output character instead of pinyin ; Mode C
	'''
	
	def __init__(self, path, type_, **kwargs):
		'''
		path：folder prefix
		'''
				
		self.datapath = path
		self.type_ = type_ # Three types: {train、dev、test}
		assert(self.type_ in ['train','dev','test']),"data generator label not correct"

		self.slash = config.slash()
				
		self.list_word = GetWordList(config.word_dict)
		self.output_mode = config.OUTPUT_MODE
		
		self.LoadDataList()
		self.LoadNoiseList()

		self.feature_method = config.FEATURE_EXTRACTION
		self.feature_length = config.FEATURE_LENGTH
		self.feature_method_2 = config.FEATURE_EXTRACTION_2
		self.feature_length_2 = config.FEATURE_LENGTH_2

		self.bg = np.load(config.bg_folder+config.FEATURE_BG)
		self.bg2 = np.load(config.bg_folder+config.FEATURE_BG_2)
	
	def GetDataHeat(self,ran_num):
			
		filename = self.list_all_json[ran_num]['r_path']
		list_word = self.list_all_json[ran_num]['word']
		list_word = list(list_word)
		
		filename = config.path_checker(filename)
		
		noise_filename = ''
		if (config.ADD_NOISE) and (ran_num%2 == 0):
			rand_noise = np.random.randint(0,33)
			noise_filename = self.list_noise[rand_noise]
		wavsignal,fs=read_wav_data_2((self.datapath + filename),config.AUDIO_SPEED,config.SPEED_CHANGE,noise_filename)
		
		feat_out=[]
		for i in list_word:
			if(''!=i):
				n=self.WordToNum(i)
				feat_out.append(n)
		
		## handle error (that pinyin not in dict)
		if (-1 in feat_out):
			return [(-1,),(-1,)],-1
		
		data_input = self.feature_method.feature(wavsignal,fs)
		data_input = data_input.reshape(data_input.shape[0],data_input.shape[1],1)
		if config.DOUBLE_INPUT:
			data_input_2 = self.feature_method_2.feature(wavsignal,fs)
			data_input_2 = data_input_2.reshape(data_input_2.shape[0],data_input_2.shape[1],1)
		else:
			data_input_2 = None

		if (data_input.shape[0] > config.AUDIO_LENGTH):
			return [(-1,),(-1,)],-1
				
		data_label = np.array(feat_out)

		if len(data_label) > config.LABEL_MAX_STRING_LENGTH:
			return [(-1,),(-1,)],-1

		return [data_input,data_input_2], data_label

	def WordToNum(self,word):
		try:
			if(word != ''):
				return self.list_word.index(word)
		except ValueError: # not in list
			return -1

	def data_generator_language(self,batch_size=6, audio_length=config.AUDIO_LENGTH, go_backward=False, **kwargs):

		self.epsilon = config.EPSILON
		
		if batch_size > self.min_batch_size:
			pass
		else:
			batch_size = self.min_batch_size
		
		while True:
			X = np.zeros((batch_size, audio_length, self.feature_length, 1), dtype = np.float)
			X2 = np.zeros((batch_size, audio_length, self.feature_length_2, 1), dtype = np.float)
			y = np.zeros((batch_size,config.LABEL_MAX_STRING_LENGTH,4788))
			y.fill((self.epsilon/(len(self.list_word)-1)))
			
			count = 0
			start = 0
			while(count < batch_size):
				# ensure every batch contain at least one record from different dataset 
				if count < self.min_batch_size:
					ran_num = np.random.randint(start,self.dataset_range[count])
				else:
					ran_num = np.random.randint(0,self.DataNum)

				data_input_ls, data_labels = self.GetDataHeat(ran_num)
				data_input = data_input_ls[0]
				data_input_2 = data_input_ls[1]

				if (-1 in data_input) or (-1 in data_input_2): #handle error
					continue

				X[count]=self.bg
				X2[count]=self.bg2
				
				if go_backward == False:
					X[count,0:len(data_input)] = data_input
					X2[count,0:len(data_input)] = data_input_2

				elif go_backward == True:
					X[count,-(len(data_input)):] = data_input
					X2[count,-(len(data_input)):] = data_input_2

				if self.feature_method.USE_CMVN == True:
					X[count] = CMVN(X[count])
				if self.feature_method_2.USE_CMVN == True:
					X2[count] = CMVN(X2[count])

				for idx,num in enumerate(data_labels):
					y[count,idx,num] = (1-self.epsilon)
				y[count,(len(data_labels)):,self.list_word.index('_')] = (1-self.epsilon)
				
				if count < self.min_batch_size:
					start = self.dataset_range[count]
				count += 1
			
			if config.DOUBLE_INPUT:
				yield [X, X2], y
			else:
				yield X, y

def test_showOutput(ds,b_size=6):
	'''
	ds = 1: Mode A

	ds = 2: Mode C
	'''
	path=config.data_path

	if ds==1:
		d = DataSpeech(path,'test',performance=True)
	elif ds==2:
		d = DataSpeech2(path,'test')
	else:
		raise Exception()
	print(f'-----dataset_range:{len(d.dataset_range)}')

	print('-----data_generator_single:')
	dg = d.data_generator_single(b_size)
	data = next(dg)[0]
	print(f'-----btach size: {len(data[0])}')
	print('X:')
	print(data[0][0].reshape(-1))
	print('Y:')
	print(data[1][0])

	print('-----data_generator_double:')
	dg = d.data_generator_double(b_size)
	data = next(dg)[0]
	print(f'-----btach size: {len(data[0])}')
	print('X:')
	print(data[0][0].reshape(-1))
	print('X2:')
	print(data[1][0].reshape(-1))
	print('Y:')
	print(data[2][0])

def test_dataset_length():
	path = config.data_path
	d = DataSpeech(path,'test')
	d.test_length()
