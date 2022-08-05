#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Speech model
"""
import os
import time
import config

from ASR.file_function.file_func import *
from ASR.file_function.logic_func import *
from ASR.readdata_json import DataSpeech, DataSpeech2

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

ModelName='PINYIN'
version=config.VERSION

class ModelSpeech():

	def __init__(self,init=True,lr=0.001):
		assert config.OUTPUT_MODE in ['A','C']
		
		self.MODEL_NAME = ModelName
		self.version = version
		self.slash = config.slash()
		self.CheckDir()
		self.MAX_OUTPUT_SIZE = config.OUTPUT_SIZE
		self.label_max_string_length = config.LABEL_MAX_STRING_LENGTH
		self.AUDIO_LENGTH = config.AUDIO_LENGTH
		self.feature_method = config.FEATURE_EXTRACTION
		self.feature_length = config.FEATURE_LENGTH
		self.feature_method_2 = config.FEATURE_EXTRACTION_2
		self.feature_length_2 = config.FEATURE_LENGTH_2
		self.datapath = config.data_path
		self.model_filename = f"{self.MODEL_NAME}_{self.version}"
		self.base_model_filename = f"{self.MODEL_NAME}_{self.version}_base"
		self.total_trained_step = 0
		self.LEARNING_RATE = lr

		self._model, self.base_model = self.CreateModel() 
			
	def CheckDir(self):
		self.folder_name = config.speech_model_folder + f'{self.version}'
		if (os.path.isdir(self.folder_name) == False):
			os.mkdir(self.folder_name)
			print(f'[info] Created new folder')
		self.folder_name = self.folder_name + self.slash

		if (os.path.isfile(self.folder_name+'record_config.py') == False): # add record_config.py
			os.system(f'cp {config.abs_path_prefix}config.py {self.folder_name}record_config.py')

	def CreateModel(self):
		'''
		_model --> end to end (for CTC loss);
		base_model --> y_pred (for prediction)
		'''
		
		model,model_data = config.architecture(self.AUDIO_LENGTH,self.feature_length,self.MAX_OUTPUT_SIZE,self.label_max_string_length)
		opt = Adam(learning_rate = self.LEARNING_RATE, beta_1 = 0.9, beta_2 = 0.999, decay = 0.0, epsilon = 10e-8) # epsilon --> prevent divided by zero, replacing a very small value
		model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer = opt)
		model.summary()

		print('[info] Create Model Successful, Compiles Model Successful')
		return model, model_data	
		
	def TrainModel(
		self, epoch = 1, save_step = 50, 
		batch_size = 4, mode='train',
		max_step = 1000000
		):
		'''
		Train the model
		'''
		rest_count = 0
		# readdata
		if config.OUTPUT_MODE == "A":
			data=DataSpeech(self.datapath, type_=mode)
		elif config.OUTPUT_MODE == "C":
			data=DataSpeech2(path=self.datapath,type_=mode)
				
		yielddatas = data.data_generator_single(batch_size, self.AUDIO_LENGTH,go_backward=False)
		if config.DOUBLE_INPUT == True:
			yielddatas = data.data_generator_double(batch_size, self.AUDIO_LENGTH,go_backward=False)
		
		for epoch in range(epoch):
			print('[message] epoch %d' % epoch)
			while True:
				try:
					print('[message] Accumulated steps: %d+'%(self.total_trained_step))
					
					self._model.fit_generator(yielddatas, save_step)
					self.total_trained_step += save_step
				except StopIteration:
					print('[error] generator error. please check data format.')
					break

				except KeyboardInterrupt:
					save_entire = input("\n[save] Save Entire Model? [Y/n]:")
					if (save_entire.lower()=='y'):
						self.SaveEntireModel()
					self.SaveModel_Weight(step=self.total_trained_step)
					break
				
				self.SaveModel_Weight(step=self.total_trained_step)

				# break
				rest_count += save_step
				if (rest_count % 15000 == 0):
					print("\n[info] Rest after 15000 steps")
					time.sleep(150)

				if (rest_count > max_step):
					break

	def LoadModel_Weight(self):
		'''
		Load model
		'''
		prefix = self.folder_name
		self._model.load_weights((prefix + self.model_filename))
		self.base_model.load_weights((prefix + self.base_model_filename))

		if os.path.isfile(f'{prefix}{self.version}_Lastest_update_weight.txt'):
			f = open(f'{prefix}{self.version}_Lastest_update_weight.txt','r')
			text = f.read()
			text_list = text.split('\n')
			print(f'[info] Successfully load the model (weight) built in {text_list[0]} with {text_list[1]}')
			try:
				self.total_trained_step += int((text_list[1].split())[-1])
			except Exception:
				pass
			f.close()

	def LoadModel_transfer(self,ver=''):
		'''
		Inherit entire model from another version
		'''
		if (ver == ''):
			return

		prefix = config.speech_model_folder + f'{ver}' + config.slash()
		self._model.load_weights((prefix + f"{self.MODEL_NAME}_{ver}"))
		self.base_model.load_weights((prefix + f"{self.MODEL_NAME}_{ver}_base"))
		
		print(f'[info] Successfully load the model from {ver}')

	def SaveModel_Weight(self,prefix = config.speech_model_folder,step = 0):
		'''
		Save model
		'''
		prefix = self.folder_name
		self._model.save_weights((prefix + self.model_filename))
		self.base_model.save_weights((prefix + self.base_model_filename))

		from datetime import datetime
		f = open(f'{prefix}{self.version}_Lastest_update_weight.txt','w')
		f.writelines(str(datetime.today()))
		f.writelines(f"\nTotal step trained: {step}")
		f.close()
	
	def SaveEntireModel(self):
		'''
		Save the entire model
		'''
		prefix = self.folder_name
		self.base_model.save((prefix+f'{self.MODEL_NAME}_{self.version}_base'),save_format='tf')

	def LoadEntireModel(self):
		'''
		Load the entire model
		'''
		prefix = self.folder_name
		self._model = load_model((prefix+f'{self.MODEL_NAME}_{self.version}'))
		self.base_model = load_model((prefix+f'{self.MODEL_NAME}_{self.version}_base'))

		opt = Adam(learning_rate = self.LEARNING_RATE, beta_1 = 0.9, beta_2 = 0.999, decay = 0.0, epsilon = 10e-8) # epsilon --> prevent divided by zero, replacing a very small value
		self._model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer = opt)

		print('[info] Load Entire Model Successful, Compiles Model Successful')
	
	def export_CNN_component(self,folder,type_,bottom="the_input",top="post_relu"):
		'''
		Export the trained resnet_50 part for further usage
		'''
		from tensorflow.keras.models import Model

		sharing_input = self.base_model.get_layer(bottom).get_output_at(0)
		sharing_output = self.base_model.get_layer(top).get_output_at(0)

		cnn = Model(sharing_input,sharing_output)
		cnn.save_weights(f'{folder}{type_}/{type_}')

	@property
	def model(self):

		return self._model
