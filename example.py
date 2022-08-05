#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script for using the trained ASR model
"""

import click
import config
import tensorflow as tf

from ASR.asr import asr, asr_double, asr_e2e, asr_e2e_double
from ASR.file_function.file_func import AudioDecodeError
from tensorflow.keras import backend as K

gpu = config.GPU

if gpu:
	configP = tf.compat.v1.ConfigProto()
	configP.gpu_options.allow_growth=True
	sess = tf.compat.v1.Session(config=configP)

@click.command()
@click.option('-p','--path', help='Audio file path', default=config.test_audio, show_default=True)
@click.option('-v','--version', help='Model version', default=config.VERSION, show_default=True)
def main(path,version): # ASR end-to-end version

    try:
        if config.OUTPUT_MODE == 'A':

            if not config.DOUBLE_INPUT: # Single input
                asr_model = asr(version)
            else : # Double input
                asr_model = asr_double(version)

        elif config.OUTPUT_MODE == 'C':

            if not config.DOUBLE_INPUT: # Single input
                asr_model = asr_e2e(version)
            else : # Double input
                asr_model = asr_e2e_double(version)

        r = asr_model.RecognizeSpeech_FromFile(path)
            
        print('[ASR] Sentence:\n',r)

    except AudioDecodeError:
        print('Audio decode error')

    except FileNotFoundError:
        print('File not found error')

    finally:
        K.clear_session()

if __name__ == "__main__":
    main()