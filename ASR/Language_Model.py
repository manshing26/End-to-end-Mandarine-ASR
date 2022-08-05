#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import config
from ASR.file_function.logic_func import *
from ASR.file_function.file_func import *
from ASR.PINYIN_model_architecture import gru_block
from ASR.pinyin_dict import Pinyin_dict
from ASR.readdata_json import DataSpeech2 as DSC

import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tqdm import tqdm

class DL_language_model():

	def __init__(self):

		self.folder_name = config.DL_language_model_folder + 'base/'
		self.max_input_length = config.LABEL_MAX_STRING_LENGTH
		self.num_pinyin = config.OUTPUT_SIZE
		self.list_symbol = GetSymbolList(config.pinyin_dict)
		self.list_word = GetWordList(config.word_dict)
		self.checkDir()
		self.total_trained_step = 0
		self.slash = slash()

		self.model = self.build_model()

	def checkDir(self):

		if (os.path.isdir(self.folder_name) == False): # not exist -> False
			os.mkdir(self.folder_name)
			print(f'[info] Created new folder')
			return False

		else: # exist -> True
			return True

	def change_dir(self,new_ver:str):
		self.ver = new_ver
		self.SM_MODEL_NAME = f"PINYIN_{self.ver}_base"
		self.SM_MODEL_PATH = config.speech_model_folder + self.ver + self.slash
		assert os.path.isdir((self.SM_MODEL_PATH + self.SM_MODEL_NAME)), 'PINYIN model not found'

		self.folder_name = config.DL_language_model_folder + f'{new_ver}/'
		have_record = self.checkDir()

		if have_record:
			try:
				self.load_model()
			except Exception:
				pass

	def get_asr_model(self):
		self.asr_base_model = tf.keras.models.load_model(self.SM_MODEL_PATH + self.SM_MODEL_NAME)

	def build_model(self):
		
		input_layer = tf.keras.layers.Input(shape=(self.max_input_length,))

		embed_layer = tf.keras.layers.Embedding(self.num_pinyin,256)(input_layer)

		gru_1 = gru_block(embed_layer,256)
		gru_2 = gru_block(gru_1,256)
		gru_3 = gru_block(gru_2,256)

		dense_1 = tf.keras.layers.Dense(512)(gru_3)
		dense_2 = tf.keras.layers.Dense(512)(dense_1)
		dense_3 = tf.keras.layers.Dense(4788)(dense_2)
		y_pred = tf.keras.layers.Activation('softmax')(dense_3)

		model_L = tf.keras.models.Model(inputs=input_layer,outputs=y_pred)
		model_L.compile('adam','categorical_crossentropy')
		model_L.summary()
		return model_L

	def train_model(self):

		step = 1000

		yielddata = Data_generator_word(type_='train').data_generator(8)

		step_count = self.total_trained_step
		while(True):
			try:
				print(f'[info] Accumulated step: {step_count}')
				self.model.fit_generator(yielddata,step)
				step_count += step
				self.total_trained_step += step
				self.save_model()

			except KeyboardInterrupt:
				print('[info] Keyboard Interrupt')
				self.save_model()
				saveE = input('[Option] Save Entire model? [y/n]: ')
				if (saveE.lower() == 'y'):
					self.saveEntire()
					print('[info] Saved')
				break

	def fine_tune(self):

		# load_base_model
		self.get_asr_model()

		yielddata = Data_generator_word(type_='train').data_generator_audio_input(self.asr_base_model,8)

		step_count = self.total_trained_step
		while(True):
			try:
				print(f'[info] Accumulated step: {step_count}')
				self.model.fit_generator(yielddata,500)
				step_count += 500
				self.total_trained_step += 500
				self.save_model()

			except KeyboardInterrupt:
				print('[info] Keyboard Interrupt')
				self.save_model()
				saveE = input('[Option] Save Entire model? [y/n]: ')
				if (saveE.lower() == 'y'):
					self.saveEntire()
					print('[info] Saved')
				break

	def save_model(self):
		save_dest = self.folder_name + 'language_model'
		self.model.save_weights(save_dest)

		f = open(f'{self.folder_name}total_trained_step.txt','w')
		f.writelines(f"\nTotal step trained:{self.total_trained_step}")
		f.close()

	def load_model(self):
		load_dest = self.folder_name + 'language_model'
		self.model.load_weights(load_dest)

		self.total_trained_step = 0
		if os.path.isfile(f'{self.folder_name}total_trained_step.txt'):
			f = open(f'{self.folder_name}total_trained_step.txt','r')
			text = f.read()
			text_list = text.split(':')
			print(f'[info] Successfully load the model (weight) with total trained step {text_list[1]}')
			try:
				self.total_trained_step += int(text_list[1])
			except Exception:
				pass
			f.close()

	def saveEntire(self):
		save_dest = self.folder_name + f'language_model'
		self.model.save(save_dest,save_format='tf')

	def loadEntire(self):
		self.model = tf.keras.models.load_model(self.folder_name + f'language_model')

	def Predict(self,pinyin_list):

		pinyin_list_num = [to_list_index(pinyin,self.list_symbol) for pinyin in pinyin_list]

		x = np.zeros((1,64,),dtype=np.int32)
		x[0,:] = self.list_symbol.index('_')
		x[0,0:(len(pinyin_list_num))] = pinyin_list_num

		result = self.model.predict(x)[0]
		idx = np.argmax(result,axis=1)
		sentence = [self.list_word[i] for i in idx]
		sentence = [sen for sen in sentence if sen != '_']

		return (''.join(sentence))

	def test_accuracy(self,epoch=10,batch_size=5): #accuracy of language model only, not included the performance of pinyin model

		if 'base/' in self.folder_name:
			yielddata = Data_generator_word(type_='test').data_generator(batch_size)
		else:
			self.get_asr_model()
			yielddata = Data_generator_word(type_='test').data_generator_audio_input(self.asr_base_model,batch_size,real_data=True)

		total_score = 0

		for i in range(epoch):
			print(f'[info] Epoch {i+1}')
			data = next(yielddata)
			y_pred = self.model.predict(data[0])
			y_pred = np.argmax(y_pred,axis=2)
			y_true = np.argmax(data[1],axis=2)

			for j in range(batch_size):
				y_true_input = [str(word) for word in y_true[j] if word != self.list_word.index('_')]
				y_pred_input = [str(word) for word in y_pred[j] if word != self.list_word.index('_')]
				total_score += WordErrorRate(y_true_input,y_pred_input)

		total_score = (total_score/epoch)/batch_size
		print(f'\n[result] WER of language model: {total_score}')
		print(f'[result] Accuracy of language model: {(1-total_score)*100}%')

###########################################################################################################

class Data_generator_word():

	def __init__(self,type_='train'):
		self.type_ = type_
		self.datapath = config.data_path
		self.slash = config.slash()
		self.list_symbol = GetSymbolList(config.dict_folder+'dict.txt')
		self.list_word = GetWordList(config.word_dict)
		self.NUM_LIST = {'0':'零','1':'一','2':'二','3':'三','4':'四','5':'五','6':'六','7':'七','8':'八','9':'九'}
		self.UNIT = ['','十','百','千','万']

		self.epsilon = config.EPSILON # label smoothing

		self.LoadDataList()

	def LoadDataList(self):

		## open-source data
		folder_ls = config.DATASETS
		filename_ls = []

		for f in folder_ls:
			filename_ls.append(f'{self.datapath+f+self.slash+self.type_}.json')

		self.list_all_json,self.dataset_range = get_json_all(filename_ls)
		
		self.DataNum = self.GetDataNum()
		self.min_batch_size = len(self.dataset_range)
		print('[info] loaded open-source data')

		## yyihu data
		self.yh_sentence = []
		df_yh = pd.read_csv(f'{self.datapath}yyihu/metadata.csv')
		label = [json.loads(l) for l in df_yh.label]
		for l in label:
			for segment in l:
				self.yh_sentence.append(segment["Text"])
		print('[info] loaded yyihu data')

		# server data
		with open(f'{self.datapath}server/server_label.txt','rb') as f:
			server_label = pickle.load(f)
		for s in server_label:
			self.yh_sentence.append(s[1]['Text'])
		print('[info] loaded server data')

		self.yh_sentence = list(set(self.yh_sentence))

	def GetDataNum(self):
		return len(self.list_all_json)

	def SymbolToNum(self,symbol):
		try:
			return self.list_symbol.index(symbol)
		except ValueError: # not in list error
			return -1

	def word_to_num(self,word):
		try:
			return self.list_word.index(word)
		except ValueError:
			return -1

	def label_formatter(self,sentence):
		'''
		for changing arabic number
		'''
		pattern = re.compile(r"，|。|？| ")
		sentence = re.sub(pattern,'',sentence)

		return_ls = list(sentence)
		num = []
		for idx,ele in enumerate(return_ls):
			if ele in self.NUM_LIST.keys():
				num.append(idx)
		
		# No need to change, return
		if len(num)==0:
			return sentence
		
		# form index set
		# [1,2,3,7,8,10] --> [[1,2,3],[7,8],[10]]
		temp=[]
		tu=[]
		for i in num:
			if (len(tu)==0):
				tu.append(i)
			elif (tu[-1]==i-1): #continuous
				tu.append(i)
			else:
				temp.append(tuple(tu))
				tu.clear()
				tu.append(i)

		if len(tu)!=0: # append last element in tu
			temp.append(tuple(tu))

		percent=''
		# Change to words
		for t in reversed(temp):

			word=return_ls[t[0]:t[-1]+1]
			percentage = ('%' in word) # check percentage
			if percentage:
				word.remove('%')
				percent='百份之'

			# Add unit
			unit = len(word)
			for i,w in enumerate(word):
				if w != '0' and (len(word)<=len(self.UNIT)): # more than 100k
					word[i] = ((self.NUM_LIST[w]+self.UNIT[unit-1]))
				else:
					word[i] = (self.NUM_LIST[w])
				unit -= 1

			# remove unnecessary zero
			while(True):
				if word[-1] == self.NUM_LIST['0'] and (len(word)-1)!=0 :
					word.pop(-1)
				else:
					break

			return_ls[t[0]:t[-1]+1] = percent+''.join(word)
			
		return (''.join(return_ls))
	
	def test_valid(self):

		valid = 0

		print('Calculating valid data...')
		for i in tqdm(self.yh_sentence):
			sen = self.label_formatter(i)
			sentence_num = [self.word_to_num(word) for word in sen]
			if(-1 not in sentence_num):
				valid += 1

		print(f'All Data:{len(self.yh_sentence)}')
		print(f'Valid data:{valid}')
	
	def data_generator(self,batch_size=8):

		p_dict = Pinyin_dict()

		while(True):
			x = np.zeros((batch_size,config.LABEL_MAX_STRING_LENGTH))
			y = np.zeros((batch_size,config.LABEL_MAX_STRING_LENGTH,4788))
			# y = np.ndarray(shape=(batch_size,config.LABEL_MAX_STRING_LENGTH,4788))
			y.fill((self.epsilon/(len(self.list_word)-1)))

			i = 0
			while(i<batch_size):
				if (i%2) == 0: # open-source
					rand = np.random.randint(self.DataNum)
					
					wav_data = self.list_all_json[rand]

					## extract pinyin from dataset
					pinyin_list = wav_data['pinyin']
					pinyin_list_num = [self.SymbolToNum(pinyin) for pinyin in pinyin_list]

					## handle chinese character
					sentence = wav_data['word']
					sentence_num = [self.word_to_num(word) for word in sentence]

				else: #yyihu
					rand = np.random.randint(len(self.yh_sentence))
					sentence = self.label_formatter(self.yh_sentence[rand])

					pinyin_list = p_dict.check(sentence)
					pinyin_list_num = [self.SymbolToNum(pinyin) for pinyin in pinyin_list]

					sentence_num = [self.word_to_num(word) for word in sentence]

				if(-1 in pinyin_list_num):
					continue
				if (len(pinyin_list_num)) > config.LABEL_MAX_STRING_LENGTH:
					continue
				if(-1 in sentence_num):
					continue
				if(len(sentence_num)) > config.LABEL_MAX_STRING_LENGTH:
					continue

				x[i,:] = self.list_symbol.index('_')
				x[i,0:(len(pinyin_list_num))] = pinyin_list_num

				for nw in enumerate(sentence_num):
					y[i,nw[0],nw[1]] = (1-self.epsilon)
				y[i,(len(sentence_num)):,self.list_word.index('_')] = (1-self.epsilon)

				i += 1

			yield (x,y)

	def data_generator_audio_input(self,base_model,batch_size=5,real_data=False):

		real_data_gen = DSC(self.datapath,type_=self.type_).data_generator_language(batch_size=batch_size)
		self_gen = self.data_generator(batch_size=batch_size)

		flag = True
		while(True):
			if flag:
				if not real_data: # all real data
					flag = False
				valid = True
				input_, y = next(real_data_gen)
				actual_batch_size = y.shape[0]

				X = np.zeros((actual_batch_size,config.LABEL_MAX_STRING_LENGTH))

				base_pred = base_model.predict(x=input_)
				base_pred = base_pred[:,:,:]

				in_len = np.zeros((actual_batch_size),dtype=np.int32)
				in_len[:] = config.OUTPUT_LENGTH

				r = K.ctc_decode(base_pred, in_len, greedy = True)
				r1 = K.get_value(r[0][0])

				for i in range(actual_batch_size):
					X[i,:] = self.list_symbol.index('_')
					x = [num for num in r1[i] if num != -1] # exclude -1
					if len(x) > config.LABEL_MAX_STRING_LENGTH:
						valid = False
						break
					X[i,0:len(x)] = x
				
				if valid:
					yield X,y

			else:
				flag = True
				yield next(self_gen)
