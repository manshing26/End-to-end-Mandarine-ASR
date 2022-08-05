#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Some logicial function
'''
import numpy as np
from python_speech_features import mfcc, delta, logfbank

def WordErrorRate(y_true, y_pred):
	from jiwer import wer
	'''
	type: list/str
	'''
	if type(y_pred)==str:
		y_pred=[s for s in y_pred]
	if type(y_true)==str:
		y_true=[s for s in y_true]
	return wer(y_true,y_pred)

def LenErrorRate(y_true,y_pred):

	len_error = abs(len(y_true)-len(y_pred))

	return len_error/len(y_true)

class GetMfcc():
	USE_CMVN = True
	FEATURE_LENGTH = 39

	@staticmethod
	def feature(wavsignal,fs,channel=0): #39
		
		feat_mfcc=mfcc(wavsignal[channel],fs)
		feat_mfcc_d=delta(feat_mfcc,2)
		feat_mfcc_dd=delta(feat_mfcc_d,2)
		
		wav_feature = np.column_stack((feat_mfcc, feat_mfcc_d, feat_mfcc_dd))
		return wav_feature

class GetMfcc_slim():
	USE_CMVN = True
	FEATURE_LENGTH = 13

	@staticmethod
	def feature(wavsignal,fs,channel=0): #13
		
		feat_mfcc = mfcc(wavsignal[channel],fs)
		
		return feat_mfcc

class GetLogfbank():
	USE_CMVN = False
	FEATURE_LENGTH = 60

	@staticmethod
	def feature(wavsignal,fs,nfilt=60, channel=0): #60

		feat_logfbank_L=logfbank(wavsignal[channel],fs,nfilt=nfilt,nfft=512) # Left channel / mono channel

		return feat_logfbank_L

def CMVN(data):
	'''
	data.shape -> (audio_length,feature_method_width)
	'''
	epsilon = 1e-10
	mean_ = np.mean(data,axis=(0,))
	stddev = np.std(data,axis=(0,))

	return (data[:,:] - mean_) / np.maximum(stddev,epsilon)

def getInputLength(data_input, down_scale=8, maxOutput=200):
	frame = data_input.shape[0] / down_scale
	if frame%1 != 0:
		frame += 2
	if frame > maxOutput:
		frame == maxOutput
	return int(frame)

def to_list_index(find_element,list_to_be_check):
	try:
		return list_to_be_check.index(find_element)
	except ValueError: # not in the list
		return -1
