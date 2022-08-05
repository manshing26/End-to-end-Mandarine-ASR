#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Speech model training script
"""
import click
import config
import tensorflow as tf

from ASR.PINYIN_model import ModelSpeech

gpu = config.GPU

if gpu:
	configP = tf.compat.v1.ConfigProto()
	configP.gpu_options.allow_growth=True
	sess = tf.compat.v1.Session(config=configP)

@click.command()
@click.option('-t','--transfer_model',help='Version of model to be transferd', type=str)
@click.option('-l','--load_weight',help='Load the saved weight',type=bool, default=False, show_default=True)
@click.option('-r','--learning_rate',help='Model learning rate', type=float, default=0.0001, show_default=True)
def main(learning_rate,transfer_model,load_weight):

	ms = ModelSpeech(lr=learning_rate)

	if transfer_model != None and load_weight:
		raise Exception('[message] Cannot transfer weight and load weight from saved weight concurrently')

	if transfer_model != None:
		ms.LoadModel_transfer(ver=transfer_model)

	elif load_weight:
		ms.LoadModel_Weight()

	else: # control in CLI

		load = input("\n[option] Load saved weight? [Y/n/t]:")

		if (load.lower()=='y'): #Y
			ms.LoadModel_Weight()

		elif (load.lower()=='t'):
			ver_input = input("\n[option] Version of model to be transferd [e.g: verR0.00c]:")
			ms.LoadModel_transfer(ver=ver_input)

		else: # n
			load_f = input("\n[option] Continue without loading the weight? Saved weight will be coved if exists [Y/n]:")

			if (load_f.lower()=='y'):
				pass
			else:
				exit()

	ms.TrainModel(epoch = 1, batch_size = config.BATCH_SIZE, save_step = config.SAVE_STEP, mode='train') # train, dev, test

if __name__=="__main__":
	main()