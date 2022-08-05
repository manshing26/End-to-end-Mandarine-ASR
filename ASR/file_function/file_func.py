#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Provide function for handling the audio file
'''
import os
import numpy as np
import json
from typing import Union, List
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

class AudioDecodeError(CouldntDecodeError):
	pass

def formatter(filename:str,set_rate:bool=True):
	'''
	formatter(filename,set_rate=True) --> return boolean

	Change the format to wav
	'''
	if not os.path.isfile(filename):
		return False

	from pydub import AudioSegment
	path = os.path.dirname(filename)
	name = os.path.basename(filename)
	name,init_format = name.split('.')[0],name.split('.')[1]

	sound = AudioSegment.from_file(filename,format=init_format)

	# change frame rate
	if set_rate:
		sound.set_frame_rate(16000)

	# export
	try:
		sound.export((path+slash()+name+'.wav'),format='wav')
		return True
	except Exception:
		return False

def read_wav_data(filename:str, num_channel:bool = False):
	'''
	For asr.py
	'''
	try:
		wav = AudioSegment.from_file(filename)
		num_channel=wav.channels
		fs = wav.frame_rate

		str_data = wav.raw_data
		wave_data = np.fromstring(str_data, dtype = np.short)
		wave_data.shape = -1, num_channel # reshape
		wave_data = wave_data.T # transform
		
		return wave_data,fs,num_channel
	
	except CouldntDecodeError:
		raise AudioDecodeError

	except FileNotFoundError:
		raise FileNotFoundError

def read_wav_data_2(filename:str,speed:Union[int,float]=1,mode:str='constant',noise_filename:str = ''):
	'''
	For training / readdata
	'''
	try:
		wav = AudioSegment.from_file(filename)
		
	except CouldntDecodeError:
		raise AudioDecodeError

	except FileNotFoundError:
		raise FileNotFoundError

	num_channel=wav.channels
	framerate=wav.frame_rate

	if mode == 'range':
		speed_ls = [speed-0.1, speed, speed+0.1]
		speed = speed_ls[np.random.randint(3)]
		wav = wav._spawn(wav.raw_data,overrides={'frame_rate':int(framerate*speed)})
		wav = wav.set_frame_rate(16000)

	if noise_filename != '':
		try:
			wav = noise_overlay(wav,noise_filename)
		except Exception:
			pass

	str_data = wav.raw_data
	wave_data = np.fromstring(str_data, dtype = np.short)
	if mode == 'range':
		wave_data = wave_data * np.random.uniform(0.5,2) # Volumn change
	wave_data.shape = -1, num_channel # reshape
	wave_data = wave_data.T # transform
	
	return wave_data, wav.frame_rate

def noise_overlay(wav,noise_filename):
	noise = AudioSegment.from_file(noise_filename)
	noise_vol = np.random.randint(15,25)
	noise = noise-noise_vol
	overlay = wav.overlay(noise, position=0)
	return overlay

def merge_dict(dict1,dict2):
	return {**dict1,**dict2}

def GetSymbolList(datapath:str):
	'''
	GetSymbolList(datapath) --> [pinyin]

	return a list included all pinyins
	'''		

	assert (os.path.isfile(datapath)), 'File not found'

	txt_file=open(datapath,'r',encoding='UTF-8') # Read file
	txt_text=txt_file.read()
	txt_lines=txt_text.split('\n')
	list_symbol=[]
	for i in txt_lines:
		if(i!=''):
			txt_l=i.split('\t')
			list_symbol.append(txt_l[0])
	txt_file.close()
	list_symbol.append('_')
	return list_symbol

def GetSymbolWordDict(datapath:str):
	'''
	GetSymbolWordDict(datapath) --> {pinyin:[words]}
	
	return a dict with all pinyins as keys and their related words
	'''
	if(datapath != ''):
		if(datapath[-1]!='/' or datapath[-1]!='\\'):
			datapath = datapath + '/'

	if os.path.isfile((datapath+'dict.txt')) == False: # File not exist
		return -1

	txt_file=open(datapath + 'dict.txt','r',encoding='UTF-8')
	txt_text=txt_file.read()
	txt_lines=txt_text.split('\n')
	symbol_word_dict={}
	for i in txt_lines:
		if(i!=''):
			txt_l=i.split('\t')
			symbol_word_dict[txt_l[0]] = [word for word in txt_l[1]]
	txt_file.close()
	symbol_word_dict['_'] = []
	return symbol_word_dict

def GetWordList(datapath:str):
	
	assert (os.path.isfile(datapath)), 'File not found'

	txt_file=open(datapath,'r',encoding='UTF-8') # Read file
	txt_text=txt_file.read()
	txt_lines=txt_text.split('\n')
	list_word=[]
	for i in txt_lines:
		if(i!=''):
			list_word.append(i)
	txt_file.close()
	list_word.append('_')
	return list_word

def path_checker(path:str):
    from platform import system
    system = system()
    if (system == "Windows"):
        path = path.replace("/","\\")
    elif (system == "Linux"):
        path = path.replace("\\","/")
    return path

def slash():
    from platform import system
    system = system()
    if (system == "Windows"):
        return "\\"
    elif (system == "Linux"):
        return "/"
    return "/"

def output_auto(mode:str):
	
	mode_all = ["A","B","C"]

	assert(mode in mode_all), 'wrong mode in config.py'
	if mode == "A":
		return (1424,"dict.txt")
	if mode == "B":
		return (409,"dict_2.txt")
	if mode == "C":
		return (4788,"word2.txt")

def select_bg(feature_length:int,feature_length_2:int):
	mapping = {
		'60':'bg.npy',
		'39':'bg_mfcc.npy',
		'13':'bg_mfcc_slim.npy'
	}

	try:
		return mapping[str(feature_length)],mapping[str(feature_length_2)]
		
	except KeyError:
		raise KeyError('feature_length or feature_length_2 not valid')

def output_bg(feature_method,source:str,dest:str,start:int=0,end:Union[int,None]=None):
	
	assert os.path.isfile(source), 'File not found'
	signal,fs,num_channel = read_wav_data(source)

	for c in range(num_channel):

		if end == None:
			feat = feature_method(signal,fs,channel=c)[start:,:]
		else:
			feat = feature_method(signal,fs,channel=c)[start:end,:]
		
		feat = feat.reshape(feat.shape[0],feat.shape[1],1)

		dest_c = dest+f'_{c}'
		print(feat.shape)
		np.save(dest_c,feat)

		assert os.path.isfile(dest_c+'.npy'), 'Generated file not found'
		test_load = np.load(dest_c+'.npy')
		assert (not(False in (test_load==feat))), 'Saved .npy file is broken'

def get_json_all(filename_ls:List[str]):

	list_all_json = []
	dataset_range = []
	for filename in filename_ls:

		assert os.path.isfile(filename), f'json file not found: {filename}'
		
		with open(filename,'r') as f:
			dataset = json.loads(f.read())
			list_all_json.extend(dataset)
			dataset_range.append(len(list_all_json))

	return list_all_json, dataset_range