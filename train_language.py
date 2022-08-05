#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Speech model training script
"""
import click
import config
import tensorflow as tf
import tensorflow.keras.backend as K

from ASR.Language_Model import DL_language_model

gpu = config.GPU

if gpu:
	configP = tf.compat.v1.ConfigProto()
	configP.gpu_options.allow_growth=True
	sess = tf.compat.v1.Session(config=configP)

def get_ver(dll:DL_language_model,forced_change:bool):
	ver = input('[Input] version name [ver/<empty>]: ')
	if ver != '':
		dll.change_dir(ver)
	elif (ver == '') and (not forced_change):
		return
	else:
		exit(0)

@click.command()
@click.option('-e','--epoch',help='Epoch of test',type=int, default=200, show_default=True)
def main(epoch):

	dll = DL_language_model()
	load = input('[Option] Load weight? [y/n]: ')
	if (load.lower() == 'y'):
		try:
			dll.load_model()
			print('[info] Load weight success')

		except FileNotFoundError:
			print('[error] No weight data is found')

	opt = input('[Option] What you want do do? [train/test/ft]: ')
	if (opt.lower() == 'train'):
		dll.train_model()

	elif (opt.lower() == 'test'):
		get_ver(dll,forced_change=False)
		dll.test_accuracy(epoch)
		K.clear_session()

	elif (opt.lower() == 'ft'):
		get_ver(dll,forced_change=True)
		dll.fine_tune()

	else:
		print('[info] invalid input, program exit')
		
if __name__=="__main__":
	main()